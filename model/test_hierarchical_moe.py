import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


# ============================================================================
# 配置类
# ============================================================================

class ModelConfig:
    """模型配置"""

    def __init__(
            self,
            expert_dim: int = 512,
            num_experts: int = 4,
            disease_embed_dim: int = 1024,  # 修改为1024维
            hidden_dim: int = 256,
            num_attention_heads: int = 8,
            dropout: float = 0.1,
            use_residual: bool = True,
            use_cross_expert_attention: bool = True,
            modality_feature_dim: int = 1024  # 修改为单个维度参数
    ):
        self.expert_dim = expert_dim
        self.num_experts = num_experts
        self.disease_embed_dim = disease_embed_dim
        self.hidden_dim = hidden_dim
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.use_residual = use_residual
        self.use_cross_expert_attention = use_cross_expert_attention
        self.modality_feature_dim = modality_feature_dim  # 单个模态特征维度


class ExpertConfig:
    """专家配置 - 用于解释和分析"""

    def __init__(
            self,
            expert_names: Optional[List[str]] = None,
            expert_specializations: Optional[Dict[int, List[str]]] = None
    ):
        # 专家名称
        if expert_names is None:
            self.expert_names = [
                "Stenosis & Compression",  # 狭窄与压迫
                "Fracture & Trauma",  # 骨折与创伤
                "Tumor & Mass Effect",  # 肿瘤与占位
                "Infection & Inflammation"  # 感染与炎症
            ]
        else:
            self.expert_names = expert_names

        # 专家专门化疾病
        if expert_specializations is None:
            self.expert_specializations = {
                0: ["Acute Vertebral Compression Fracture", "Chronic Vertebral Compression Fracture", "Burst Fracture", "Pars Interarticularis Defect", "Transverse Process Fracture",
                    "Posterior Vertebral Wall High Signal", "Lumbar Disc Herniation"],
                1: ["Lumbar Spinal Canal Stenosis", "Lateral Recess Stenosis", "Foraminal Stenosis", "Lumbar Spondylolisthesis", "Lumbar Scoliosis", "Adjacent Segment Disease"],
                2: ["Schmorl's Node", "Kummell Disease", "Modic Endplate Changes", "Spondylitis", "Spinal Infection", "Subcutaneous Fasciitis"],
                3: ["Multiple Myeloma", "Vertebral Body Tumor", "Intradural Spinal Tumor", "Postoperative Internal Fixation", "Non-instrumented Postoperative Spine", "Postoperative Percutaneous Kyphoplasty"]
            }
        else:
            self.expert_specializations = expert_specializations


# ============================================================================
# 25种脊柱疾病列表
# ============================================================================

SPINE_DISEASES = [
    "Acute Vertebral Compression Fracture",  # 急性椎体压缩骨折
    "Chronic Vertebral Compression Fracture",  # 慢性椎体压缩骨折
    "Burst Fracture",  # 爆裂骨折
    "Pars Interarticularis Defect",  # 峡部裂
    "Transverse Process Fracture",  # 横突骨折
    "Posterior Vertebral Wall High Signal",  # 椎体后壁高信号
    "Lumbar Disc Herniation",  # 腰椎间盘突出
    "Lumbar Spinal Canal Stenosis",  # 腰椎管狭窄
    "Lateral Recess Stenosis",  # 侧隐窝狭窄
    "Foraminal Stenosis",  # 椎间孔狭窄
    "Lumbar Spondylolisthesis",  # 腰椎滑脱
    "Lumbar Scoliosis",  # 腰椎侧弯
    "Adjacent Segment Disease",  # 邻近节段病
    "Schmorl's Node",  # Schmorl结节
    "Kummell Disease",  # Kummell病
    "Modic Endplate Changes",  # Modic终板改变
    "Spondylitis",  # 脊柱炎
    "Spinal Infection",  # 脊柱感染
    "Subcutaneous Fasciitis",  # 皮下筋膜炎
    "Multiple Myeloma",  # 多发性骨髓瘤
    "Vertebral Body Tumor",  # 椎体肿瘤
    "Intradural Spinal Tumor",  # 椎管内肿瘤
    "Postoperative Internal Fixation",  # 术后内固定
    "Non-instrumented Postoperative Spine",  # 非器械术后脊柱
    "Postoperative Percutaneous Kyphoplasty"  # 术后经皮椎体成形
]

# ============================================================================
# 专家先验矩阵 (Expert Prior Matrix)
# ============================================================================

# 临床放射科专家知识：每种疾病对应的模态重要性
# 格式: [AP_XR, LAT_XR, SagT1, SagT2, AxT2, T2_FS]
# 1.0 = 非常重要, 0.5 = 有一定价值, 0.0 = 基本无用

EXPERT_PRIOR_MATRIX = {
    "Acute Vertebral Compression Fracture": [1.0, 1.0, 0.5, 0.5, 0.5, 1.0],
    "Chronic Vertebral Compression Fracture": [1.0, 1.0, 0.5, 0.5, 0.0, 0.0],
    "Burst Fracture": [1.0, 1.0, 0.5, 0.5, 0.5, 1.0],
    "Pars Interarticularis Defect": [0.5, 0.5, 1.0, 0.5, 0.5, 0.0],
    "Transverse Process Fracture": [1.0, 1.0, 0.5, 0.5, 0.5, 1.0],
    "Posterior Vertebral Wall High Signal": [0.0, 0.0, 0.5, 1.0, 0.0, 1.0],
    "Lumbar Disc Herniation": [0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
    "Lumbar Spinal Canal Stenosis": [0.5, 0.5, 0.5, 1.0, 1.0, 0.5],
    "Lateral Recess Stenosis": [0.0, 0.0, 0.5, 1.0, 1.0, 0.5],
    "Foraminal Stenosis": [0.0, 0.0, 0.5, 1.0, 0.5, 0.5],
    "Lumbar Spondylolisthesis": [1.0, 1.0, 0.5, 0.5, 0.0, 0.0],
    "Lumbar Scoliosis": [1.0, 0.5, 0.5, 0.5, 0.0, 0.0],
    "Adjacent Segment Disease": [0.5, 0.5, 0.5, 1.0, 0.5, 0.5],
    "Schmorl's Node": [0.5, 0.5, 1.0, 1.0, 0.0, 0.5],
    "Kummell Disease": [1.0, 1.0, 0.5, 0.5, 0.5, 0.0],
    "Modic Endplate Changes": [0.5, 0.5, 1.0, 1.0, 0.5, 1.0],
    "Spondylitis": [0.5, 0.5, 0.5, 1.0, 1.0, 1.0],
    "Spinal Infection": [0.5, 0.5, 0.5, 1.0, 1.0, 1.0],
    "Subcutaneous Fasciitis": [0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
    "Multiple Myeloma": [0.5, 0.5, 1.0, 1.0, 0.5, 1.0],
    "Vertebral Body Tumor": [0.5, 0.5, 1.0, 1.0, 0.5, 1.0],
    "Intradural Spinal Tumor": [0.0, 0.0, 0.5, 1.0, 1.0, 0.5],
    "Postoperative Internal Fixation": [1.0, 1.0, 0.5, 0.5, 0.5, 0.5],
    "Non-instrumented Postoperative Spine": [0.5, 0.5, 0.5, 1.0, 0.5, 0.5],
    "Postoperative Percutaneous Kyphoplasty": [1.0, 1.0, 1.0, 0.5, 0.5, 1.0]
}


# ============================================================================
# 第一部分：ExpertCell 的四个核心组件
# ============================================================================

class ModalitySpecificExpert(nn.Module):
    """
    模态特定局部专家 - 提取细粒度的模态特定线索

    针对每个专家e和模态m，应用模态特定的MLP提取特征：
    h^{e,local}_m = φ^{local}_{e,m}(F^{(m)}_k)
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, modality_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            modality_features: [batch_size, input_dim]
        Returns:
            [batch_size, output_dim]
        """
        return self.mlp(modality_features)


class ExpertInternalGating(nn.Module):
    """
    专家内部模态门控机制

    计算专家内部的模态门控权重
    """

    def __init__(self, disease_embed_dim: int, num_modalities: int):
        super().__init__()
        self.disease_embed_dim = disease_embed_dim
        self.num_modalities = num_modalities

        # 为模态特征和疾病嵌入创建门控网络
        self.gating_net = nn.Linear(disease_embed_dim + 1024, 1)  # 假设模态特征维度为1024

    def forward(self, disease_embedding: torch.Tensor,
                modality_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            disease_embedding: [class_num, disease_embed_dim]
            modality_features: List of [batch_size, modality_feature_dim]
        Returns:
            [class_num, num_modalities] - 每个类别对每个模态的权重
        """
        class_num = disease_embedding.size(0)
        num_modalities = len(modality_features)
        batch_size = modality_features[0].size(0)

        # 为每个模态计算门控权重
        gating_weights_list = []
        for modality_feature in modality_features:
            # 扩展疾病嵌入以匹配模态特征和批次
            disease_embedding_expanded = disease_embedding.unsqueeze(1).expand(-1, batch_size, -1)  # [class_num, batch_size, disease_embed_dim]
            modality_feature_expanded = modality_feature.unsqueeze(0).expand(class_num, -1, -1)  # [class_num, batch_size, modality_feature_dim]

            # 合并疾病嵌入和模态特征
            combined = torch.cat([disease_embedding_expanded, modality_feature_expanded], dim=-1)  # [class_num, batch_size, disease_embed_dim + modality_feature_dim]

            # 计算门控logits
            gating_logits = self.gating_net(combined).squeeze(-1)  # [class_num, batch_size]

            # 在批次维度上平均，得到每个类对该模态的权重
            gating_weights = torch.mean(gating_logits, dim=-1)  # [class_num]
            gating_weights_list.append(gating_weights)

        # 堆叠所有模态的权重
        gating_weights = torch.stack(gating_weights_list, dim=1)  # [class_num, num_modalities]

        # 在模态维度上进行softmax
        gating_weights = F.softmax(gating_weights, dim=1)  # [class_num, num_modalities]

        return gating_weights


class PathologyGlobalTransformation(nn.Module):
    """
    病理级全局转换

    处理全局疾病上下文，捕获高级病理模式：
    h^e_{global} = Ψ^{global}_e(Ĥ_k)
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 use_residual: bool = True, dropout: float = 0.1):
        super().__init__()
        self.use_residual = use_residual

        self.transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

        if use_residual and input_dim == output_dim:
            self.residual_proj = nn.Identity()
        elif use_residual:
            self.residual_proj = nn.Linear(input_dim, output_dim)
        else:
            self.residual_proj = None

    def forward(self, global_disease_context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            global_disease_context: [batch_size, input_dim]
        Returns:
            [batch_size, output_dim]
        """
        transformed = self.transform(global_disease_context)

        if self.residual_proj is not None:
            residual = self.residual_proj(global_disease_context)
            return transformed + residual
        else:
            return transformed


class CrossExpertAttention(nn.Module):
    """
    跨专家通信 - 通过多头自注意力实现

    建模专家间的交互：
    z^e = MHA_e({h^{e'}_{local}}^E_{e'=1})
    """

    def __init__(self, expert_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=expert_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

    def forward(self, expert_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            expert_tokens: [batch_size, num_experts, expert_dim]
        Returns:
            [batch_size, num_experts, expert_dim]
        """
        attended_tokens, _ = self.multihead_attn(
            query=expert_tokens,
            key=expert_tokens,
            value=expert_tokens
        )
        return attended_tokens


# ============================================================================
# 第二部分：ExpertCell 完整实现
# ============================================================================

class ExpertCell(nn.Module):
    """
    专家细胞 - 整合四个核心组件

    每个ExpertCell捕获一种独特的病理风格，产生表示 ĥ^e ∈ ℝ^d

    组件：
    1. 模态特定局部专家
    2. 专家内部模态门控
    3. 病理级全局转换
    4. 跨专家通信（外部应用）
    """

    def __init__(
            self,
            disease_embed_dim: int,
            expert_dim: int,
            modality_feature_dim: int,
            hidden_dim: int = 256,
            num_attention_heads: int = 8,
            dropout: float = 0.1,
            use_residual: bool = True
    ):
        super().__init__()

        self.expert_dim = expert_dim
        self.modality_feature_dim = modality_feature_dim

        # 1. 模态特定局部专家（为每个模态创建一个）
        self.local_experts = nn.ModuleList([
            ModalitySpecificExpert(
                input_dim=modality_feature_dim,
                hidden_dim=hidden_dim,
                output_dim=expert_dim,
                dropout=dropout
            )
        ])

        # 2. 专家内部模态门控
        self.internal_gating = ExpertInternalGating(
            disease_embed_dim=disease_embed_dim,
            num_modalities=6  # 假设有6个模态
        )

        # 3. 病理级全局转换
        self.global_transform = PathologyGlobalTransformation(
            input_dim=disease_embed_dim,
            hidden_dim=hidden_dim,
            output_dim=expert_dim,
            use_residual=use_residual,
            dropout=dropout
        )

    def forward(
            self,
            global_disease_context: torch.Tensor,
            disease_embedding: torch.Tensor,
            modality_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            global_disease_context: [batch_size, disease_embed_dim]
            disease_embedding: [class_num, disease_embed_dim]
            modality_features: List of [batch_size, modality_feature_dim]
        Returns:
            [batch_size, expert_dim]
        """
        batch_size = global_disease_context.size(0)
        num_modalities = len(modality_features)

        # 1. 应用模态特定局部专家
        local_expert_outputs = []
        for modality_feature in modality_features:
            local_output = self.local_experts[0](modality_feature)  # 使用第一个专家处理所有模态
            local_expert_outputs.append(local_output)

        # 2. 计算专家内部模态门控权重
        gating_weights = self.internal_gating(disease_embedding, modality_features)  # [class_num, num_modalities]

        # 3. 聚合局部专家输出
        h_local = torch.zeros(batch_size, self.expert_dim, device=global_disease_context.device)

        # 为每个模态加权求和
        for i in range(num_modalities):
            # 计算每个类别的平均权重
            avg_weight = torch.mean(gating_weights[:, i])  # 标量
            h_local += avg_weight * local_expert_outputs[i]

        # 4. 应用病理级全局转换（对每个batch样本应用相同的转换）
        h_global = self.global_transform(global_disease_context)

        # 合并局部和全局表示
        expert_output = h_local + h_global

        return expert_output


# ============================================================================
# 第三部分：HierarchicalMoE 实现
# ============================================================================

class DiseaseExpertGating(nn.Module):
    """
    疾病条件专家门控

    计算专家门控权重：
    g^{(d)}_{k,e} = softmax_e(w^T_e e_k)
    """

    def __init__(self, disease_embed_dim: int, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        self.expert_vectors = nn.Parameter(torch.randn(num_experts, disease_embed_dim))
        nn.init.normal_(self.expert_vectors, mean=0.0, std=0.02)

    def forward(self, disease_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            disease_embedding: [class_num, disease_embed_dim]
        Returns:
            [class_num, num_experts]
        """
        logits = torch.matmul(disease_embedding, self.expert_vectors.T)
        gating_weights = F.softmax(logits, dim=-1)
        return gating_weights


class HierarchicalMoE(nn.Module):
    """
    分层多路径跨模态专家混合模型

    实现疾病级专家聚合：
    H_k = Σ^E_{e=1} g^{(d)}_{k,e} * ĥ^e

    Args:
        disease_embed_dim: 疾病嵌入维度 (默认1024)
        expert_dim: 专家输出维度 (默认512)
        num_experts: 专家数量 (默认4)
        hidden_dim: 隐藏层维度 (默认256)
        num_attention_heads: 注意力头数 (默认8)
        dropout: Dropout率 (默认0.1)
        use_residual: 是否使用残差连接 (默认True)
        use_cross_expert_attention: 是否使用跨专家通信 (默认True)
        modality_feature_dim: 模态特征维度 (默认1024)
    """

    def __init__(
            self,
            disease_embed_dim: int = 1024,
            expert_dim: int = 512,
            num_experts: int = 4,
            hidden_dim: int = 256,
            num_attention_heads: int = 8,
            dropout: float = 0.1,
            use_residual: bool = True,
            use_cross_expert_attention: bool = True,
            modality_feature_dim: int = 1024
    ):
        super().__init__()

        self.num_experts = num_experts
        self.expert_dim = expert_dim
        self.use_cross_expert_attention = use_cross_expert_attention

        # 创建多个ExpertCell
        self.expert_cells = nn.ModuleList([
            ExpertCell(
                disease_embed_dim=disease_embed_dim,
                expert_dim=expert_dim,
                modality_feature_dim=modality_feature_dim,
                hidden_dim=hidden_dim,
                num_attention_heads=num_attention_heads,
                dropout=dropout,
                use_residual=use_residual
            ) for _ in range(num_experts)
        ])

        # 疾病条件专家门控
        self.expert_gating = DiseaseExpertGating(
            disease_embed_dim=disease_embed_dim,
            num_experts=num_experts
        )

        # 跨专家通信
        if use_cross_expert_attention:
            self.cross_expert_attention = CrossExpertAttention(
                expert_dim=expert_dim,
                num_heads=num_attention_heads,
                dropout=dropout
            )
        else:
            self.cross_expert_attention = None

        self.fc = nn.Linear(512, 25)

    def forward(
            self,
            global_disease_context: torch.Tensor,
            disease_embedding: torch.Tensor,
            modality_features: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播

        Args:
            global_disease_context: 全局疾病上下文 [batch_size, disease_embed_dim]
            disease_embedding: 疾病嵌入 [class_num, disease_embed_dim]
            modality_features: 模态特征列表，每个元素形状为[batch_size, 1024]

        Returns:
            final_representation: 最终疾病表示 [batch_size, expert_dim]
            expert_info: 包含专家输出和门控权重的字典
        """
        batch_size = global_disease_context.size(0)

        # 1. 应用所有专家细胞
        expert_outputs = []
        for expert_cell in self.expert_cells:
            expert_output = expert_cell(
                global_disease_context=global_disease_context,
                disease_embedding=disease_embedding,
                modality_features=modality_features
            )
            expert_outputs.append(expert_output)

        # 堆叠专家输出: [batch_size, num_experts, expert_dim]
        expert_tokens = torch.stack(expert_outputs, dim=1)

        # 2. 应用跨专家通信
        if self.cross_expert_attention is not None:
            attended_tokens = self.cross_expert_attention(expert_tokens)
            final_expert_outputs = []
            for expert_idx in range(self.num_experts):
                original_output = expert_tokens[:, expert_idx, :]
                attention_output = attended_tokens[:, expert_idx, :]
                final_output = original_output + attention_output
                final_expert_outputs.append(final_output)
        else:
            final_expert_outputs = [expert_tokens[:, i, :] for i in range(self.num_experts)]

        # 3. 计算疾病条件专家门控权重
        # 这里需要根据class_num的维度来获取权重，但我们需要的是batch维度的表示
        # 因此我们将对所有疾病嵌入求平均或选择一个代表性的嵌入
        expert_gating_weights = self.expert_gating(disease_embedding)  # [class_num, num_experts]
        # 为了与原逻辑保持一致，我们取疾病嵌入的平均值来获得类似batch的表示
        avg_disease_embedding = torch.mean(disease_embedding, dim=0, keepdim=True).expand(batch_size, -1)  # [batch_size, disease_embed_dim]
        avg_expert_gating_weights = self.expert_gating(avg_disease_embedding)  # [batch_size, num_experts]

        # 4. 聚合专家输出
        final_representation = torch.zeros(batch_size, self.expert_dim, device=global_disease_context.device)
        for expert_idx, expert_output in enumerate(final_expert_outputs):
            weight = avg_expert_gating_weights[:, expert_idx:expert_idx + 1]
            final_representation += weight * expert_output

        # 5. 准备专家信息
        expert_info = {
            'expert_outputs': final_expert_outputs,
            'gating_weights': avg_expert_gating_weights,
            'raw_expert_outputs': [expert_tokens[:, i, :] for i in range(self.num_experts)]
        }
        final_representation = self.fc(final_representation)

        return final_representation

    def get_expert_importance(self, disease_embedding: torch.Tensor) -> torch.Tensor:
        """
        获取专家重要性权重

        Args:
            disease_embedding: [class_num, disease_embed_dim]
        Returns:
            [class_num, num_experts]
        """
        return self.expert_gating(disease_embedding)


# ============================================================================
# 默认配置
# ============================================================================

if __name__ == '__main__':
    # 演示如何调用模型
    print("创建HierarchicalMoE模型示例:")

    # 方法1: 使用默认配置创建模型
    model = HierarchicalMoE(
        disease_embed_dim=1024,
        num_experts=4,
        expert_dim=512,
        modality_feature_dim=1024,
        num_attention_heads=8,
        dropout=0.1
    )

    print(f"模型创建成功，参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 创建示例输入
    batch_size = 4
    num_modalities = 6
    device = torch.device('cpu')  # 实际使用时可以改为'cuda'

    global_disease_context = torch.randn(batch_size, 1024, device=device)
    disease_embedding = torch.randn(25, 1024, device=device)  # class_num=25
    modality_features = [
        torch.randn(batch_size, 1024, device=device) for _ in range(num_modalities)
    ]

    final_representation = model(
        global_disease_context,
        disease_embedding,
        modality_features
    )

    print(f"前向传播完成")
    print(f"输出表示形状: {final_representation.shape}")
    # print(f"专家门控权重形状: {expert_info['gating_weights'].shape}")




# """下面是只有X光时的moe的代码"""
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Dict, List, Optional, Tuple
#
#
# # ============================================================================
# # 配置类
# # ============================================================================
#
# class ModelConfig:
#     """模型配置"""
#
#     def __init__(
#             self,
#             expert_dim: int = 512,
#             num_experts: int = 4,
#             disease_embed_dim: int = 1024,  # 修改为1024维
#             hidden_dim: int = 256,
#             num_attention_heads: int = 8,
#             dropout: float = 0.1,
#             use_residual: bool = True,
#             use_cross_expert_attention: bool = True
#     ):
#         self.expert_dim = expert_dim
#         self.num_experts = num_experts
#         self.disease_embed_dim = disease_embed_dim
#         self.hidden_dim = hidden_dim
#         self.num_attention_heads = num_attention_heads
#         self.dropout = dropout
#         self.use_residual = use_residual
#         self.use_cross_expert_attention = use_cross_expert_attention
#
#
# class ExpertConfig:
#     """专家配置 - 用于解释和分析"""
#
#     def __init__(
#             self,
#             expert_names: Optional[List[str]] = None,
#             expert_specializations: Optional[Dict[int, List[str]]] = None
#     ):
#         # 专家名称
#         if expert_names is None:
#             self.expert_names = [
#                 "Stenosis & Compression",  # 狭窄与压迫
#                 "Fracture & Trauma",  # 骨折与创伤
#                 "Tumor & Mass Effect",  # 肿瘤与占位
#                 "Infection & Inflammation"  # 感染与炎症
#             ]
#         else:
#             self.expert_names = expert_names
#
#         # 专家专门化疾病
#         if expert_specializations is None:
#             self.expert_specializations = {
#                 0: ["Acute Vertebral Compression Fracture", "Chronic Vertebral Compression Fracture", "Burst Fracture",
#                     "Pars Interarticularis Defect", "Transverse Process Fracture",
#                     "Posterior Vertebral Wall High Signal", "Lumbar Disc Herniation"],
#                 1: ["Lumbar Spinal Canal Stenosis", "Lateral Recess Stenosis", "Foraminal Stenosis",
#                     "Lumbar Spondylolisthesis", "Lumbar Scoliosis", "Adjacent Segment Disease"],
#                 2: ["Schmorl's Node", "Kummell Disease", "Modic Endplate Changes", "Spondylitis", "Spinal Infection",
#                     "Subcutaneous Fasciitis"],
#                 3: ["Multiple Myeloma", "Vertebral Body Tumor", "Intradural Spinal Tumor",
#                     "Postoperative Internal Fixation", "Non-instrumented Postoperative Spine",
#                     "Postoperative Percutaneous Kyphoplasty"]
#             }
#         else:
#             self.expert_specializations = expert_specializations
#
#
# # ============================================================================
# # 25种脊柱疾病列表
# # ============================================================================
#
# SPINE_DISEASES = [
#     "Acute Vertebral Compression Fracture",  # 急性椎体压缩骨折
#     "Chronic Vertebral Compression Fracture",  # 慢性椎体压缩骨折
#     "Burst Fracture",  # 爆裂骨折
#     "Pars Interarticularis Defect",  # 峡部裂
#     "Transverse Process Fracture",  # 横突骨折
#     "Posterior Vertebral Wall High Signal",  # 椎体后壁高信号
#     "Lumbar Disc Herniation",  # 腰椎间盘突出
#     "Lumbar Spinal Canal Stenosis",  # 腰椎管狭窄
#     "Lateral Recess Stenosis",  # 侧隐窝狭窄
#     "Foraminal Stenosis",  # 椎间孔狭窄
#     "Lumbar Spondylolisthesis",  # 腰椎滑脱
#     "Lumbar Scoliosis",  # 腰椎侧弯
#     "Adjacent Segment Disease",  # 邻近节段病
#     "Schmorl's Node",  # Schmorl结节
#     "Kummell Disease",  # Kummell病
#     "Modic Endplate Changes",  # Modic终板改变
#     "Spondylitis",  # 脊柱炎
#     "Spinal Infection",  # 脊柱感染
#     "Subcutaneous Fasciitis",  # 皮下筋膜炎
#     "Multiple Myeloma",  # 多发性骨髓瘤
#     "Vertebral Body Tumor",  # 椎体肿瘤
#     "Intradural Spinal Tumor",  # 椎管内肿瘤
#     "Postoperative Internal Fixation",  # 术后内固定
#     "Non-instrumented Postoperative Spine",  # 非器械术后脊柱
#     "Postoperative Percutaneous Kyphoplasty"  # 术后经皮椎体成形
# ]
#
# # ============================================================================
# # 专家先验矩阵 (Expert Prior Matrix)
# # ============================================================================
#
# # 临床放射科专家知识：每种疾病对应的模态重要性
# # 格式: [AP_XR, LAT_XR, SagT1, SagT2, AxT2, T2_FS]
# # 1.0 = 非常重要, 0.5 = 有一定价值, 0.0 = 基本无用
#
# EXPERT_PRIOR_MATRIX = {
#     "Acute Vertebral Compression Fracture": [1.0, 1.0, 0.5, 0.5, 0.5, 1.0],
#     "Chronic Vertebral Compression Fracture": [1.0, 1.0, 0.5, 0.5, 0.0, 0.0],
#     "Burst Fracture": [1.0, 1.0, 0.5, 0.5, 0.5, 1.0],
#     "Pars Interarticularis Defect": [0.5, 0.5, 1.0, 0.5, 0.5, 0.0],
#     "Transverse Process Fracture": [1.0, 1.0, 0.5, 0.5, 0.5, 1.0],
#     "Posterior Vertebral Wall High Signal": [0.0, 0.0, 0.5, 1.0, 0.0, 1.0],
#     "Lumbar Disc Herniation": [0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
#     "Lumbar Spinal Canal Stenosis": [0.5, 0.5, 0.5, 1.0, 1.0, 0.5],
#     "Lateral Recess Stenosis": [0.0, 0.0, 0.5, 1.0, 1.0, 0.5],
#     "Foraminal Stenosis": [0.0, 0.0, 0.5, 1.0, 0.5, 0.5],
#     "Lumbar Spondylolisthesis": [1.0, 1.0, 0.5, 0.5, 0.0, 0.0],
#     "Lumbar Scoliosis": [1.0, 0.5, 0.5, 0.5, 0.0, 0.0],
#     "Adjacent Segment Disease": [0.5, 0.5, 0.5, 1.0, 0.5, 0.5],
#     "Schmorl's Node": [0.5, 0.5, 1.0, 1.0, 0.0, 0.5],
#     "Kummell Disease": [1.0, 1.0, 0.5, 0.5, 0.5, 0.0],
#     "Modic Endplate Changes": [0.5, 0.5, 1.0, 1.0, 0.5, 1.0],
#     "Spondylitis": [0.5, 0.5, 0.5, 1.0, 1.0, 1.0],
#     "Spinal Infection": [0.5, 0.5, 0.5, 1.0, 1.0, 1.0],
#     "Subcutaneous Fasciitis": [0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
#     "Multiple Myeloma": [0.5, 0.5, 1.0, 1.0, 0.5, 1.0],
#     "Vertebral Body Tumor": [0.5, 0.5, 1.0, 1.0, 0.5, 1.0],
#     "Intradural Spinal Tumor": [0.0, 0.0, 0.5, 1.0, 1.0, 0.5],
#     "Postoperative Internal Fixation": [1.0, 1.0, 0.5, 0.5, 0.5, 0.5],
#     "Non-instrumented Postoperative Spine": [0.5, 0.5, 0.5, 1.0, 0.5, 0.5],
#     "Postoperative Percutaneous Kyphoplasty": [1.0, 1.0, 1.0, 0.5, 0.5, 1.0]
# }
#
#
# # ============================================================================
# # 第一部分：ExpertCell 的四个核心组件
# # ============================================================================
#
# class ExpertInternalGating(nn.Module):
#     """
#     专家内部模态门控机制
#
#     计算专家内部的模态门控权重
#     """
#
#     def __init__(self, disease_embed_dim: int):
#         super().__init__()
#         self.disease_embed_dim = disease_embed_dim
#
#         # 为疾病嵌入创建门控网络
#         self.gating_net = nn.Linear(disease_embed_dim, 1)
#
#     def forward(self, disease_embedding: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             disease_embedding: [class_num, disease_embed_dim]
#         Returns:
#             [class_num, 1] - 简化的门控权重
#         """
#         class_num = disease_embedding.size(0)
#
#         # 计算门控logits
#         gating_logits = self.gating_net(disease_embedding)  # [class_num, 1]
#
#         # 对门控权重进行归一化
#         gating_weights = F.softmax(gating_logits, dim=0)  # [class_num, 1]
#
#         return gating_weights
#
#
# class PathologyGlobalTransformation(nn.Module):
#     """
#     病理级全局转换
#
#     处理全局疾病上下文，捕获高级病理模式：
#     h^e_{global} = Ψ^{global}_e(Ĥ_k)
#     """
#
#     def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
#                  use_residual: bool = True, dropout: float = 0.1):
#         super().__init__()
#         self.use_residual = use_residual
#
#         self.transform = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, output_dim)
#         )
#
#         if use_residual and input_dim == output_dim:
#             self.residual_proj = nn.Identity()
#         elif use_residual:
#             self.residual_proj = nn.Linear(input_dim, output_dim)
#         else:
#             self.residual_proj = None
#
#     def forward(self, global_disease_context: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             global_disease_context: [batch_size, input_dim]
#         Returns:
#             [batch_size, output_dim]
#         """
#         transformed = self.transform(global_disease_context)
#
#         if self.residual_proj is not None:
#             residual = self.residual_proj(global_disease_context)
#             return transformed + residual
#         else:
#             return transformed
#
#
# class CrossExpertAttention(nn.Module):
#     """
#     跨专家通信 - 通过多头自注意力实现
#
#     建模专家间的交互：
#     z^e = MHA_e({h^{e'}_{local}}^E_{e'=1})
#     """
#
#     def __init__(self, expert_dim: int, num_heads: int = 8, dropout: float = 0.1):
#         super().__init__()
#         self.multihead_attn = nn.MultiheadAttention(
#             embed_dim=expert_dim,
#             num_heads=num_heads,
#             dropout=dropout,
#             batch_first=True
#         )
#
#     def forward(self, expert_tokens: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             expert_tokens: [batch_size, num_experts, expert_dim]
#         Returns:
#             [batch_size, num_experts, expert_dim]
#         """
#         attended_tokens, _ = self.multihead_attn(
#             query=expert_tokens,
#             key=expert_tokens,
#             value=expert_tokens
#         )
#         return attended_tokens
#
#
# # ============================================================================
# # 第二部分：ExpertCell 完整实现 (修改版)
# # ============================================================================
#
# class ExpertCell(nn.Module):
#     """
#     专家细胞 - 整合三个核心组件（移除模态相关组件）
#
#     每个ExpertCell捕获一种独特的病理风格，产生表示 ĥ^e ∈ ℝ^d
#
#     组件：
#     1. 病理级全局转换
#     2. 专家内部门控
#     3. 跨专家通信（外部应用）
#     """
#
#     def __init__(
#             self,
#             disease_embed_dim: int,
#             expert_dim: int,
#             hidden_dim: int = 256,
#             num_attention_heads: int = 8,
#             dropout: float = 0.1,
#             use_residual: bool = True
#     ):
#         super().__init__()
#
#         self.expert_dim = expert_dim
#
#         # 1. 专家内部门控
#         self.internal_gating = ExpertInternalGating(
#             disease_embed_dim=disease_embed_dim
#         )
#
#         # 2. 病理级全局转换
#         self.global_transform = PathologyGlobalTransformation(
#             input_dim=disease_embed_dim,
#             hidden_dim=hidden_dim,
#             output_dim=expert_dim,
#             use_residual=use_residual,
#             dropout=dropout
#         )
#
#     def forward(
#             self,
#             global_disease_context: torch.Tensor,
#             disease_embedding: torch.Tensor
#     ) -> torch.Tensor:
#         """
#         Args:
#             global_disease_context: [batch_size, disease_embed_dim]
#             disease_embedding: [class_num, disease_embed_dim]
#         Returns:
#             [batch_size, expert_dim]
#         """
#         batch_size = global_disease_context.size(0)
#
#         # 1. 计算专家内部门控权重
#         gating_weights = self.internal_gating(disease_embedding)  # [class_num, 1]
#
#         # 2. 应用病理级全局转换
#         expert_output = self.global_transform(global_disease_context)
#
#         # 3. 使用门控权重对专家输出进行调整
#         # 这里我们简单地将门控权重应用于输出，或者将其作为一个调节因子
#         # 可以根据需要调整这里的逻辑
#         avg_gating_weight = torch.mean(gating_weights)  # 平均门控权重
#         expert_output = expert_output * avg_gating_weight  # 应用门控权重
#
#         return expert_output
#
#
# # ============================================================================
# # 第三部分：HierarchicalMoE 实现
# # ============================================================================
#
# class DiseaseExpertGating(nn.Module):
#     """
#     疾病条件专家门控
#
#     计算专家门控权重：
#     g^{(d)}_{k,e} = softmax_e(w^T_e e_k)
#     """
#
#     def __init__(self, disease_embed_dim: int, num_experts: int):
#         super().__init__()
#         self.num_experts = num_experts
#         self.expert_vectors = nn.Parameter(torch.randn(num_experts, disease_embed_dim))
#         nn.init.normal_(self.expert_vectors, mean=0.0, std=0.02)
#
#     def forward(self, disease_embedding: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             disease_embedding: [class_num, disease_embed_dim]
#         Returns:
#             [class_num, num_experts]
#         """
#         logits = torch.matmul(disease_embedding, self.expert_vectors.T)
#         gating_weights = F.softmax(logits, dim=-1)
#         return gating_weights
#
#
# class HierarchicalMoE(nn.Module):
#     """
#     分层多路径跨模态专家混合模型 (简化版)
#
#     实现疾病级专家聚合：
#     H_k = Σ^E_{e=1} g^{(d)}_{k,e} * ĥ^e
#
#     Args:
#         disease_embed_dim: 疾病嵌入维度 (默认1024)
#         expert_dim: 专家输出维度 (默认512)
#         num_experts: 专家数量 (默认4)
#         hidden_dim: 隐藏层维度 (默认256)
#         num_attention_heads: 注意力头数 (默认8)
#         dropout: Dropout率 (默认0.1)
#         use_residual: 是否使用残差连接 (默认True)
#         use_cross_expert_attention: 是否使用跨专家通信 (默认True)
#     """
#
#     def __init__(
#             self,
#             disease_embed_dim: int = 1024,
#             expert_dim: int = 512,
#             num_experts: int = 4,
#             hidden_dim: int = 256,
#             num_attention_heads: int = 8,
#             dropout: float = 0.1,
#             use_residual: bool = True,
#             use_cross_expert_attention: bool = True
#     ):
#         super().__init__()
#
#         self.num_experts = num_experts
#         self.expert_dim = expert_dim
#         self.use_cross_expert_attention = use_cross_expert_attention
#
#         # 创建多个ExpertCell
#         self.expert_cells = nn.ModuleList([
#             ExpertCell(
#                 disease_embed_dim=disease_embed_dim,
#                 expert_dim=expert_dim,
#                 hidden_dim=hidden_dim,
#                 num_attention_heads=num_attention_heads,
#                 dropout=dropout,
#                 use_residual=use_residual
#             ) for _ in range(num_experts)
#         ])
#
#         # 疾病条件专家门控
#         self.expert_gating = DiseaseExpertGating(
#             disease_embed_dim=disease_embed_dim,
#             num_experts=num_experts
#         )
#
#         # 跨专家通信
#         if use_cross_expert_attention:
#             self.cross_expert_attention = CrossExpertAttention(
#                 expert_dim=expert_dim,
#                 num_heads=num_attention_heads,
#                 dropout=dropout
#             )
#         else:
#             self.cross_expert_attention = None
#
#         self.fc = nn.Linear(512, 25)
#
#     def forward(
#             self,
#             global_disease_context: torch.Tensor,
#             disease_embedding: torch.Tensor
#     ) -> torch.Tensor:
#         """
#         前向传播
#
#         Args:
#             global_disease_context: 全局疾病上下文 [batch_size, disease_embed_dim]
#             disease_embedding: 疾病嵌入 [class_num, disease_embed_dim]
#
#         Returns:
#             final_representation: 最终疾病表示 [batch_size, expert_dim]
#         """
#         batch_size = global_disease_context.size(0)
#
#         # 1. 应用所有专家细胞
#         expert_outputs = []
#         for expert_cell in self.expert_cells:
#             expert_output = expert_cell(
#                 global_disease_context=global_disease_context,
#                 disease_embedding=disease_embedding
#             )
#             expert_outputs.append(expert_output)
#
#         # 堆叠专家输出: [batch_size, num_experts, expert_dim]
#         expert_tokens = torch.stack(expert_outputs, dim=1)
#
#         # 2. 应用跨专家通信
#         if self.cross_expert_attention is not None:
#             attended_tokens = self.cross_expert_attention(expert_tokens)
#             final_expert_outputs = []
#             for expert_idx in range(self.num_experts):
#                 original_output = expert_tokens[:, expert_idx, :]
#                 attention_output = attended_tokens[:, expert_idx, :]
#                 final_output = original_output + attention_output
#                 final_expert_outputs.append(final_output)
#         else:
#             final_expert_outputs = [expert_tokens[:, i, :] for i in range(self.num_experts)]
#
#         # 3. 计算疾病条件专家门控权重
#         expert_gating_weights = self.expert_gating(disease_embedding)  # [class_num, num_experts]
#         # 为了与原逻辑保持一致，我们取疾病嵌入的平均值来获得类似batch的表示
#         avg_disease_embedding = torch.mean(disease_embedding, dim=0, keepdim=True).expand(batch_size,
#                                                                                           -1)  # [batch_size, disease_embed_dim]
#         avg_expert_gating_weights = self.expert_gating(avg_disease_embedding)  # [batch_size, num_experts]
#
#         # 4. 聚合专家输出
#         final_representation = torch.zeros(batch_size, self.expert_dim, device=global_disease_context.device)
#         for expert_idx, expert_output in enumerate(final_expert_outputs):
#             weight = avg_expert_gating_weights[:, expert_idx:expert_idx + 1]
#             final_representation += weight * expert_output
#
#         # 5. 输出最终结果
#         final_representation = self.fc(final_representation)
#
#         return final_representation
#
#     def get_expert_importance(self, disease_embedding: torch.Tensor) -> torch.Tensor:
#         """
#         获取专家重要性权重
#
#         Args:
#             disease_embedding: [class_num, disease_embed_dim]
#         Returns:
#             [class_num, num_experts]
#         """
#         return self.expert_gating(disease_embedding)
#
#
# # ============================================================================
# # 默认配置
# # ============================================================================
#
# if __name__ == '__main__':
#     # 演示如何调用模型
#     print("创建HierarchicalMoE模型示例:")
#
#     # 方法1: 使用默认配置创建模型
#     model = HierarchicalMoE(
#         disease_embed_dim=1024,
#         num_experts=4,
#         expert_dim=512,
#         num_attention_heads=8,
#         dropout=0.1
#     )
#
#     print(f"模型创建成功，参数量: {sum(p.numel() for p in model.parameters()):,}")
#
#     # 创建示例输入
#     batch_size = 4
#     device = torch.device('cpu')  # 实际使用时可以改为'cuda'
#
#     global_disease_context = torch.randn(batch_size, 1024, device=device)
#     disease_embedding = torch.randn(25, 1024, device=device)  # class_num=25
#
#     final_representation = model(
#         global_disease_context,
#         disease_embedding
#     )
#
#     print(f"前向传播完成")
#     print(f"输出表示形状: {final_representation.shape}")
