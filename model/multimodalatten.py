import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiModalHierarchicalAttention(nn.Module):
    def __init__(self,
                 image_feat_dim=1024,
                 text_feat_dim=512,
                 modality_num=4,
                 reduction=4,
                 dropout=0.3):
        super().__init__()

        self.image_feat_dim = image_feat_dim
        self.text_feat_dim = text_feat_dim
        self.modality_num = modality_num
        self.drop = nn.Dropout(dropout)

        # 1. Learnable modality prior weights (bias)
        self.modality_bias = nn.Parameter(torch.tensor((0.15, 0.30, 0.20, 0.35), requires_grad=True))#nn.Parameter(torch.ones(modality_num))
        self.xray_bias = nn.Parameter(torch.tensor((0.5, 0.5), requires_grad=True))
        self.bias = nn.Parameter(torch.tensor((0.7, 0.3), requires_grad=True))

        # 2. Modality attention module (for fusing 4 modalities)
        self.modality_attn = ModalityAttention(
            dim=image_feat_dim,
            reduction=reduction
        )

        # 3. Projection layer: align text and image dimensions
        self.text_proj = nn.Linear(text_feat_dim, image_feat_dim)
        self.image_proj_k = nn.Linear(image_feat_dim, image_feat_dim)
        self.image_proj_v = nn.Linear(image_feat_dim, image_feat_dim)
        self.image_proj_k_ray = nn.Linear(image_feat_dim, image_feat_dim)
        self.image_proj_v_ray = nn.Linear(image_feat_dim, image_feat_dim)

        # 4. Classification head
        # self.classifier = nn.Sequential(
        #     nn.Dropout(dropout),
        #     nn.Linear(image_feat_dim, 512),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(512, num_classes)
        # )

        self._init_weights()

    def _init_weights(self):
        # 初始化投影层
        nn.init.xavier_uniform_(self.text_proj.weight)
        nn.init.xavier_uniform_(self.image_proj_k.weight)
        nn.init.xavier_uniform_(self.image_proj_v.weight)
        nn.init.xavier_uniform_(self.image_proj_k_ray.weight)
        nn.init.xavier_uniform_(self.image_proj_v_ray.weight)
        # Classifier initialization
        # for m in self.classifier:
        #     if isinstance(m, nn.Linear):
        #         nn.init.xavier_uniform_(m.weight)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)

    def forward(self, image_features_list, text_features, xray_feature):
        """
        Args:
            image_features_list: List[Tensor], length = T (time steps)
                                 each tensor: (X, 4, 1024)
            text_features: Tensor, (B, D_text), e.g., (B, 512)

        Returns:
            logits: (B, num_classes)
            time_attn_weights: (B, T) —— 可用于可视化哪个 slice 组合最重要
        """
        T = len(image_features_list)
        # B = image_features_list[0].shape[0]
        # device = text_features.device

        # Step 1: Stack to (B, X, 4, 1024)
        x = torch.stack(image_features_list, dim=0)  # (B, X, 4, 1024)
        # x = x.to(device)
        # Step 2: Apply learnable modality bias
        bias = self.modality_bias.view(1, 1, self.modality_num, 1)  # (1,1,4,1)
        x = x * bias  # broadcasting → (B, X, 4, 1024)

        xray_bias = self.xray_bias.view(1, 2, 1)
        xray_feature = xray_feature * xray_bias

        # Step 3: Modality attention per time step
        fused_per_t = []
        for t in range(T):
            feat_t = x[t]  # (X, 4, 1024)
            fused_t, _ = self.modality_attn(feat_t)  # (B, 1024)
            fused_per_t.append(fused_t)

        # Shape: (B, T, 1024)
        image_seq = torch.stack(fused_per_t, dim=0)  # (B, T, 1024)

        # Step 4: Text projection to same dim
        text_proj = self.text_proj(text_features)  # (B, 1024)
        Q = text_proj.unsqueeze(1)  # (B, 1, 1024)

        # Step 5: Project image sequence for K and V
        K = self.image_proj_k(image_seq)  # (B, T, 1024)
        V = self.image_proj_v(image_seq)  # (B, T, 1024)
        K_xray = self.image_proj_k_ray(xray_feature)
        V_xray = self.image_proj_v_ray(xray_feature)

        # Step 6: Scaled dot-product attention
        scale = self.image_feat_dim ** 0.5
        scores = torch.bmm(Q, K.transpose(1, 2)) / scale  # (B, 1, T)
        time_attn_weights = F.softmax(scores, dim=-1)  # (B, 1, T)
        time_attn_weights = self.drop(time_attn_weights)

        scores_xray = torch.bmm(Q, K_xray.transpose(1, 2)) / scale
        xray_attn_weights = F.softmax(scores_xray, dim=-1)
        xray_attn_weights = self.drop(xray_attn_weights)

        # Step 7: Weighted sum over time steps
        mri_final_feat = torch.bmm(time_attn_weights, V).squeeze(1)  # (B, 1024)
        xray_final_feat = torch.bmm(xray_attn_weights, V_xray).squeeze(1)
        mri_final_feat = mri_final_feat * self.bias[0]
        xray_final_feat = xray_final_feat * self.bias[1]

        # Step 8: Classification
        # logits = self.classifier(final_feat)  # (B, num_classes)

        return mri_final_feat, time_attn_weights.squeeze(1), xray_final_feat


class ModalityAttention(nn.Module):
    def __init__(self, dim=1024, reduction=4):
        super().__init__()
        hidden_dim = dim // reduction
        self.fc = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """
        x: (B, C, D) = (B, 4, 1024)
        Returns:
            fused: (B, D)
            weights: (B, C)
        """
        B, C, D = x.shape
        # Reshape to (B*C, D)
        x_flat = x.view(-1, D)  # (B*4, 1024)
        scores = self.fc(x_flat)  # (B*4, 1)
        scores = scores.view(B, C)  # (B, 4)
        weights = F.softmax(scores, dim=1)  # (B, 4)
        # Weighted sum: (B, 4, 1) * (B, 4, 1024) -> sum over C
        weightss = weights.unsqueeze(-1)
        fea = weightss * x
        fused = torch.sum(fea, dim=1)  # (B, 1024)
        return fused, weights


if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_img = []
    for i in range(8):
        image = torch.randn(10, 4, 1024).to(device)
        input_img.append(image)
    text = torch.randn(8, 512)
    xray = torch.randn(8, 2, 1024)
    model = MultiModalHierarchicalAttention()
    model.to(device)
    a, b, c = model(input_img, text.to(device), xray.to(device))