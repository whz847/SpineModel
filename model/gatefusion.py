import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelGatedFuse(nn.Module):
    """
    y = g * x_ap + (1-g) * x_lat
    g = sigmoid(Linear([x_ap, x_lat]))   # (B, dim)
    """
    def __init__(self, dim=1024, use_ln=True, dropout=0.1, residual=True):
        super().__init__()
        self.gate = nn.Linear(dim * 2, dim)   # 2048 -> 1024
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.use_ln = use_ln
        self.ln = nn.LayerNorm(dim) if use_ln else nn.Identity()
        self.residual = residual

        # 更稳一点：把 gate 初始化为“近似平均融合”
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x_ap, x_lat):
        # x_ap, x_lat: (B, dim)
        if self.use_ln:
            x_ap = self.ln(x_ap)
            x_lat = self.ln(x_lat)

        z = torch.cat([x_ap, x_lat], dim=-1)          # (B, 2*dim)
        g = torch.sigmoid(self.gate(z))               # (B, dim)
        y = g * x_ap + (1.0 - g) * x_lat              # (B, dim)
        y = self.dropout(y)

        if self.residual:
            # 残差让训练更稳（可选）
            y = y + 0.5 * (x_ap + x_lat)

        return y



class ChannelGatedFuseB(nn.Module):
    """
    Token-wise Channel Gated Fusion
    Input:  (B, C=1024, T=50)
    Output: (B, C, T)
    """
    def __init__(self, dim=1024, use_ln=True, dropout=0.1, residual=True):
        super().__init__()
        self.gate = nn.Linear(dim * 2, dim)  # still 2048 -> 1024
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.use_ln = use_ln
        self.ln = nn.LayerNorm(dim) if use_ln else nn.Identity()
        self.residual = residual

        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x_ap, x_lat):
        # x_ap, x_lat: (B, C, T)

        # 1. 把 token 维度放到中间，变成 (B, T, C)
        x_ap = x_ap.transpose(1, 2)
        x_lat = x_lat.transpose(1, 2)

        if self.use_ln:
            x_ap = self.ln(x_ap)
            x_lat = self.ln(x_lat)

        # 2. 在每个 token 上做 gated fusion
        z = torch.cat([x_ap, x_lat], dim=-1)   # (B, T, 2C)
        g = torch.sigmoid(self.gate(z))        # (B, T, C)

        y = g * x_ap + (1.0 - g) * x_lat       # (B, T, C)
        y = self.dropout(y)

        if self.residual:
            y = y + 0.5 * (x_ap + x_lat)

        # 3. 再换回 (B, C, T)
        y = y.transpose(1, 2)
        return y
