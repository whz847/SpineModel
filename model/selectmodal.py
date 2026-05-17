import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LabelAwareModalityAttention(nn.Module):
    def __init__(self, feature_dim=1024, attention_dim=512, num_modalities=6, num_diseases=25):
        super().__init__()
        self.feature_dim = feature_dim
        self.attention_dim = attention_dim
        self.num_modalities = num_modalities
        self.num_diseases = num_diseases

        # Learnable projection layers
        self.W_q = nn.Linear(feature_dim, attention_dim)   # Text -> query
        self.W_k = nn.Linear(feature_dim, attention_dim)   # Image -> key
        self.W_v = nn.Linear(feature_dim, feature_dim)     # Image -> value

        # Prior-related parameters
        self.gamma = 1.0
        self.epsilon = 1e-8

    def forward(self, final_features, lp_features, expert_prior_matrix=None, lambda_prior=None, labels=None, is_train=True):
        """
        Args:
            final_features: [B, M, D] - 图像特征（M个模态）
            lp_features: [2*K, D] - 文本特征（K个疾病的阴阳描述）
            expert_prior_matrix: [K, M] - 专家先验矩阵（numpy 或 tensor）
            lambda_prior: [K, M] - 正则化强度（numpy 或 tensor）
            labels: [B, K] - 多标签真值（仅训练时使用）
            is_train: bool

        Returns:
            selected_modality_indices: [B, K, 2] - 每个样本每个疾病的 top-2 模态索引（仅用于分析/可视化）
            attention_weights: [B, K, M] - 注意力权重 β_{k,m}^{(i)}
            prior_loss: scalar tensor - 正则化损失
            fused_disease_representations: [B, K, D] - 融合后的疾病表征
        """
        device = final_features.device
        B, M, D = final_features.shape

        # 确定疾病数量 K
        if expert_prior_matrix is not None:
            K = expert_prior_matrix.shape[0]
        else:
            K = lp_features.shape[0] // 2

        # 提取阳性文本特征（每个疾病一个）
        pos_features = lp_features[:K]  # [K, D]

        # 投影到注意力空间
        Q = self.W_q(pos_features)                     # [K, da]
        K_img = self.W_k(final_features)               # [B, M, da]
        V_img = self.W_v(final_features)               # [B, M, D]

        # 计算 logits: [B, K, M]
        # (K, da) @ (B, da, M) -> (B, K, M)
        logits = torch.einsum('kd,bmd->bkm', Q, K_img) / (self.attention_dim ** 0.5)

        # 添加专家先验偏置
        if expert_prior_matrix is not None:
            if isinstance(expert_prior_matrix, np.ndarray):
                expert_prior_tensor = torch.tensor(expert_prior_matrix, dtype=torch.float32, device=device)
            else:
                expert_prior_tensor = expert_prior_matrix.to(device)
            # [K, M]
            prior_bias = self.gamma * (
                torch.log(expert_prior_tensor + self.epsilon) -
                torch.log(torch.tensor(0.5, device=device) + self.epsilon)
            )
            logits = logits + prior_bias.unsqueeze(0)  # [B, K, M]

        # 注意力权重：β_{k,m}^{(i)} = softmax_m(logit_{k,m}^{(i)})
        attention_weights = F.softmax(logits, dim=-1)  # [B, K, M]

        # 融合表示：H̃_k^{(i)} = Σ_m β_{k,m}^{(i)} * V(F_{i,m})
        fused_disease_representations = torch.einsum('bkm,bmd->bkd', attention_weights, V_img)  # [B, K, D]

        # 模态选择：每个 (i,k) 选 top-2 模态索引（用于分析或后续使用）
        selected_modality_indices = torch.topk(attention_weights, k=2, dim=-1).indices  # [B, K, 2]

        # 正则化损失（仅训练时计算）
        prior_loss = torch.tensor(0.0, device=device)
        if is_train and labels is not None:
            # labels: [B, K]
            assert labels.shape == (B, K), f"Label shape {labels.shape} != ({B}, {K})"

            # 将 labels 扩展为 [B, K, 1] 用于 mask
            label_mask = labels.unsqueeze(-1)  # [B, K, 1]

            # 只对正样本（label=1）计算平均注意力
            # masked_beta: [B, K, M], zero out negative samples
            masked_beta = attention_weights * label_mask.float()  # [B, K, M]

            # 对每个疾病 k，计算在所有正样本上的平均 β
            sum_beta = masked_beta.sum(dim=0)  # [K, M]
            count_pos = label_mask.float().sum(dim=0).squeeze(-1)  # [K]

            # 避免除零
            avg_beta = torch.zeros_like(sum_beta)
            valid = count_pos > 0
            avg_beta[valid] = sum_beta[valid] / count_pos[valid].unsqueeze(-1)

            # 获取专家先验和 lambda
            if isinstance(expert_prior_matrix, np.ndarray):
                P = torch.tensor(expert_prior_matrix, dtype=torch.float32, device=device)
            else:
                P = expert_prior_matrix.to(device)

            if lambda_prior is None:
                Lambda = torch.ones_like(P)
            else:
                if isinstance(lambda_prior, np.ndarray):
                    Lambda = torch.tensor(lambda_prior, dtype=torch.float32, device=device)
                else:
                    Lambda = lambda_prior.to(device)

            # 正则化损失：Σ λ_{k,m} (β̄_{k,m} - P_{k,m})²
            prior_loss = torch.sum(Lambda * (avg_beta - P) ** 2)

        return selected_modality_indices, attention_weights, prior_loss, fused_disease_representations  #[8, 25, 2], [8, 25, M], scalar, [8, 25, D]


# Example usage (optional)
if __name__ == "__main__":
    final_features = torch.randn(8, 6, 1024)
    lp_features = torch.randn(50, 1024)  # 25 classes × positive/negative

    # Extract expert prior matrix from table (only using SagT1, SagT2, AxT2, T2FS columns)
    # Order is [AX, FS, T1, T2, AP, LAP]
    expert_prior_matrix = np.array([
        [0.5, 1.0, 0.5, 0.5, 1.0, 1.0],  # Acute Vertebral Compression Fracture
        [0.0, 0.0, 0.5, 0.5, 1.0, 1.0],  # Chronic Vertebral Compression Fracture
        [0.5, 1.0, 0.5, 0.5, 1.0, 1.0],  # Burst Fracture
        [0.5, 0.0, 1.0, 0.5, 0.5, 0.5],  # Pars Interarticularis Defect
        [0.5, 1.0, 0.5, 0.5, 1.0, 1.0],  # Transverse Process Fracture
        [0.0, 1.0, 0.5, 1.0, 0.0, 0.0],  # Posterior Vertebral Wall High Signal
        [1.0, 1.0, 0.5, 1.0, 0.0, 0.0],  # Lumbar Disc Herniation
        [1.0, 0.5, 0.5, 1.0, 0.5, 0.5],  # Lumbar Spine Canal Stenosis
        [1.0, 0.5, 0.5, 1.0, 0.0, 0.0],  # Lateral Recess Stenosis
        [0.5, 0.5, 0.5, 1.0, 0.0, 0.0],  # Foraminal Stenosis
        [0.0, 0.0, 0.5, 0.5, 1.0, 1.0],  # Lumbar Spondylolisthesis
        [0.0, 0.0, 0.5, 0.5, 1.0, 0.5],  # Lumbar Scoliosis
        [0.5, 0.5, 0.5, 1.0, 0.5, 0.5],  # Adjacent Segment Disease
        [0.0, 0.5, 1.0, 1.0, 0.5, 0.5],  # Schmorl's Node
        [0.5, 0.0, 0.5, 0.5, 1.0, 1.0],  # Kummell Disease
        [0.5, 1.0, 1.0, 1.0, 0.5, 0.5],  # Modic Endplate Changes Modic
        [1.0, 1.0, 0.5, 1.0, 0.5, 0.5],  # Spondylitis
        [1.0, 1.0, 0.5, 1.0, 0.5, 0.5],  # Spinal Infection
        [1.0, 1.0, 0.5, 1.0, 0.0, 0.0],  # Subcutaneous Fasciitis
        [0.5, 1.0, 1.0, 1.0, 0.5, 0.5],  # Multiple Myeloma
        [0.5, 1.0, 1.0, 1.0, 0.5, 0.5],  # Vertebral Body Tumor
        [1.0, 0.5, 0.5, 1.0, 0.0, 0.0],  # Intradural Spinal Tumor
        [0.5, 0.5, 0.5, 1.0, 1.0, 1.0],  # Postoperative Internal Fixation
        [0.5, 0.5, 0.5, 1.0, 0.5, 0.5],  # Non-instrumented Postoperative Spine
        [0.5, 1.0, 1.0, 0.5, 1.0, 1.0],  # Postoperative Percutaneous Kyphoplasty
    ])  # [25, 6]

    # Set λ_k,m (example)
    lambda_prior = np.ones((25, 6)) * 0.1  # Adjustable

    # Create model
    model = LabelAwareModalityAttention(feature_dim=1024, attention_dim=512, num_modalities=6, num_diseases=25)

    # Create labels (example)
    labels = torch.randint(0, 2, (8, 25))

    # Forward pass
    selected_features, attn_weights, prior_loss, fused_disease_representations = model(
        final_features, lp_features,
        expert_prior_matrix=expert_prior_matrix,
        lambda_prior=lambda_prior,
        labels=labels,
        is_train=True
    )

    print("Selected Features Shape:", selected_features.shape)  # [B, 2*D]
    print("Attention Weights Shape:", attn_weights.shape)  # [B, M]
    print("Prior Loss:", prior_loss.item())

    # Check if selected modalities are correct
    print("Sample attention weights for first sample:", attn_weights[0])
    _, top_indices = torch.topk(attn_weights[0], 2)
    print("Top 2 modalities for first sample's disease 0:", selected_features[0, 0])