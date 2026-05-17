import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from .prompt_learner import ContextOptimization, SpineModel_LP, LinearProbe, FixedPrompt
import h5py
from utils.spine_process.dicom_process import DicomProcess
from .multimodalatten import MultiModalHierarchicalAttention
from clip.simple_tokenizer import SimpleTokenizer
from .selectmodal import LabelAwareModalityAttention
# from .test_hierarchical_moe_fixed import HierarchicalMoE
from .test_hierarchical_moe import HierarchicalMoE

# Spatial Fusion
class ECA_combined(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super().__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_y = self.avgpool(x)
        avg_y = self.conv(avg_y.permute(0, 2, 1))
        max_y = self.maxpool(x)
        max_y = self.conv(max_y.permute(0, 2, 1))
        y = avg_y + max_y
        y = y.permute(0, 2, 1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class MultimodalFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.tensor([0.5, 0.5]))  # Initialize as equal

    def forward(self, feat_mri, feat_ct):
        w = F.softmax(self.weights, dim=0)
        fuse = w[0] * feat_mri + w[1] * feat_ct
        return fuse


class SpineModel(nn.Module):
    def __init__(self, cfg, image_encoder, image_encoder2, text_encoder, prompt_learner):
        super().__init__()
        self.cfg = cfg
        self.image_encoder = image_encoder
        self.image_encoder2 = image_encoder2
        self.text_encoder = text_encoder
        self.prompt_learner = prompt_learner
        self.output_dim = self.cfg.DIM_PROJECTION
        self.image_projection_layer = None
        self.text_projection_layer = None
        self.channel_attention = None
        self.classnum = 25
        self.mmf = MultimodalFusion()
        self.mmha = MultiModalHierarchicalAttention(image_feat_dim=1024,
                                                    text_feat_dim=512,
                                                    modality_num=4,
                                                    reduction=4,
                                                    dropout=0.3)
        self.all_label_aware_modality_attention = LabelAwareModalityAttention(feature_dim=1024, attention_dim=512,
                                                                              num_modalities=6, num_diseases=25)
        self.mri_label_aware_modality_attention = LabelAwareModalityAttention(feature_dim=1024, attention_dim=512,
                                                                              num_modalities=4, num_diseases=25)
        self.xray_label_aware_modality_attention = LabelAwareModalityAttention(feature_dim=1024, attention_dim=512,
                                                                              num_modalities=2, num_diseases=25)
        self.moe = HierarchicalMoE(disease_embed_dim=1024,
                                   num_experts=4,
                                   expert_dim=512,
                                   modality_feature_dim=1024,
                                   num_attention_heads=8,
                                   # class_num=self.classnum,
                                   dropout=0.1
                                   )
        # self.no_moe = nn.Linear(720, 7)   # Only used in ablation experiments without MoE
        # Extract expert prior matrix from table (only using SagT1, SagT2, AxT2, T2FS columns)
        # Order is [AX, FS, T1, T2, AP, LAP]
        self.all_expert_prior_matrix = np.array([
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
            [0.5, 0.5, 0.5, 0.5, 1.0, 1.0],  # Postoperative Internal Fixation
            [0.5, 0.5, 0.5, 1.0, 0.5, 0.5],  # Non-instrumented Postoperative Spine
            [0.5, 1.0, 1.0, 0.5, 1.0, 1.0],  # Postoperative Percutaneous Kyphoplasty
        ])  # [25, 6]
        # Order is [AX, FS, T1, T2]
        self.mri_expert_prior_matrix = np.array([
            [0.5, 1.0, 0.5, 0.5],  # Acute Vertebral Compression Fracture
            [0.0, 0.0, 0.5, 0.5],  # Chronic Vertebral Compression Fracture
            [0.5, 1.0, 0.5, 0.5],  # Burst Fracture
            [0.5, 0.0, 1.0, 0.5],  # Pars Interarticularis Defect
            [0.5, 1.0, 0.5, 0.5],  # Transverse Process Fracture
            [0.0, 1.0, 0.5, 1.0],  # Posterior Vertebral Wall High Signal
            [1.0, 1.0, 0.5, 1.0],  # Lumbar Disc Herniation
            [1.0, 0.5, 0.5, 1.0],  # Lumbar Spine Canal Stenosis
            [1.0, 0.5, 0.5, 1.0],  # Lateral Recess Stenosis
            [0.5, 0.5, 0.5, 1.0],  # Foraminal Stenosis
            [0.0, 0.0, 0.5, 0.5],  # Lumbar Spondylolisthesis
            [0.0, 0.0, 0.5, 0.5],  # Lumbar Scoliosis
            [0.5, 0.5, 0.5, 1.0],  # Adjacent Segment Disease
            [0.0, 0.5, 1.0, 1.0],  # Schmorl's Node
            [0.5, 0.0, 0.5, 0.5],  # Kummell Disease
            [0.5, 1.0, 1.0, 1.0],  # Modic Endplate Changes Modic
            [1.0, 1.0, 0.5, 1.0],  # Spondylitis
            [1.0, 1.0, 0.5, 1.0],  # Spinal Infection
            [1.0, 1.0, 0.5, 1.0],  # Subcutaneous Fasciitis
            [0.5, 1.0, 1.0, 1.0],  # Multiple Myeloma
            [0.5, 1.0, 1.0, 1.0],  # Vertebral Body Tumor
            [1.0, 0.5, 0.5, 1.0],  # Intradural Spinal Tumor
            [0.5, 0.5, 0.5, 0.5],  # Postoperative Internal Fixation
            [0.5, 0.5, 0.5, 1.0],  # Non-instrumented Postoperative Spine
            [0.5, 1.0, 1.0, 0.5],  # Postoperative Percutaneous Kyphoplasty
        ])  # [25, 4]
        # Order is [AP, LAP]
        self.xray_expert_prior_matrix = np.array([
            [1.0, 1.0],  # Acute Vertebral Compression Fracture
            [1.0, 1.0],  # Chronic Vertebral Compression Fracture
            [1.0, 1.0],  # Burst Fracture
            [0.5, 0.5],  # Pars Interarticularis Defect
            [1.0, 1.0],  # Transverse Process Fracture
            [0.0, 0.0],  # Posterior Vertebral Wall High Signal
            [0.0, 0.0],  # Lumbar Disc Herniation
            [0.5, 0.5],  # Lumbar Spine Canal Stenosis
            [0.0, 0.0],  # Lateral Recess Stenosis
            [0.0, 0.0],  # Foraminal Stenosis
            [1.0, 1.0],  # Lumbar Spondylolisthesis
            [1.0, 0.5],  # Lumbar Scoliosis
            [0.5, 0.5],  # Adjacent Segment Disease
            [0.5, 0.5],  # Schmorl's Node
            [1.0, 1.0],  # Kummell Disease
            [0.5, 0.5],  # Modic Endplate Changes Modic
            [0.5, 0.5],  # Spondylitis
            [0.5, 0.5],  # Spinal Infection
            [0.0, 0.0],  # Subcutaneous Fasciitis
            [0.5, 0.5],  # Multiple Myeloma
            [0.5, 0.5],  # Vertebral Body Tumor
            [0.0, 0.0],  # Intradural Spinal Tumor
            [1.0, 1.0],  # Postoperative Internal Fixation
            [0.5, 0.5],  # Non-instrumented Postoperative Spine
            [1.0, 1.0],  # Postoperative Percutaneous Kyphoplasty
        ])  # [25, 2]
        # Initialize attention parameters, create parameters for each modality: W_f^(m), W_e^(m), u_m
        # Linear transformation from feature dimension 1024 to 1024
        self.W_f = nn.ModuleList([nn.Linear(1024, 1024) for _ in range(4)])  # 4 modalities
        self.W_e = nn.Linear(1024, 1024)  # Text feature transformation
        self.u_m = nn.Parameter(torch.randn(4, 1024))  # Attention vector for each modality

        if isinstance(self.prompt_learner, SpineModel_LP):
            self.channel_attention = ECA_combined(50)
            channel_params = sum(p.numel() for p in self.channel_attention.parameters())
            print(f"Attn parameters: {channel_params:.2f}")

        image_encoder_dim = self.get_encoder_dim("image")
        text_encoder_dim = self.get_encoder_dim("text")

        self.logit_scale = nn.Parameter(
            torch.ones([]) * np.log(1 / self.cfg.TEMPERATURE)
        )
        """
        if image_encoder_dim is not None:
            self.image_projection_layer = nn.Parameter(
                torch.empty(image_encoder_dim, self.output_dim)
            )
            trunc_normal_(self.image_projection_layer, std=0.02)
        if text_encoder_dim is not None:
            self.text_projection_layer = nn.Parameter(
                torch.empty(text_encoder_dim, self.output_dim)
            )
            trunc_normal_(self.text_projection_layer, std=0.02)
        """

    def get_encoder_dim(self, type="image"):
        encoder_dim = None
        if type == "image":
            if hasattr(self.image_encoder, "dim_out"):
                encoder_dim = self.image_encoder.dim_out
            elif hasattr(self.image_encoder, "output_dim"):
                encoder_dim = self.image_encoder.output_dim
        elif type == "image":
            if hasattr(self.image_encoder2, "dim_out"):
                encoder_dim = self.image_encoder2.dim_out
            elif hasattr(self.image_encoder2, "output_dim"):
                encoder_dim = self.image_encoder2.output_dim
        elif type == "text":
            if hasattr(self.text_encoder, "output_dim"):
                encoder_dim = self.text_encoder.output_dim

        return encoder_dim

    @property
    def network_name(self):
        name = ""
        name += "SpineModel-{}".format(self.cfg.VISUAL.NAME)
        return name

    @property
    def dtype(self):
        return self.logit_scale.dtype

    @torch.jit.ignore
    def no_weight_decay(self):
        no_weight_decay = {"logit_scale"}
        if hasattr(self.text_encoder, "no_weight_decay"):
            for k in self.text_encoder.no_weight_decay():
                no_weight_decay.add("text_encoder." + k)

        if hasattr(self.image_encoder, "no_weight_decay"):
            for k in self.image_encoder.no_weight_decay():
                no_weight_decay.add("image_encoder." + k)

        if hasattr(self.image_encoder2, "no_weight_decay"):
            for k in self.image_encoder2.no_weight_decay():
                no_weight_decay.add("image_encoder2." + k)

        return no_weight_decay

    @torch.jit.ignore
    def froze_clip_params(self):
        # Freeze parameters based on configuration
        if hasattr(self.cfg, 'FINETUNE_CLIP') and self.cfg.FINETUNE_CLIP:
            print("Keeping image encoders trainable")
        else:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            for param in self.image_encoder2.parameters():
                param.requires_grad = False
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            print("Image encoders are frozen")

    @torch.jit.ignore
    def froze_prompt_params(self):
        for param in self.prompt_learner.parameters():
            param.requires_grad = False

    def encode_image(self, image):
        x, x2 = self.image_encoder(image)
        if self.image_projection_layer is not None:
            x = x @ self.image_projection_layer

        x1 = x / x.norm(dim=-1, keepdim=True)

        return x1, x2, x

    def encode_image2(self, image):
        x, x2 = self.image_encoder2(image)
        if self.image_projection_layer is not None:
            x = x @ self.image_projection_layer

        x1 = x / x.norm(dim=-1, keepdim=True)

        return x1, x2, x

    def encode_text(self, text, embed=None):
        x, _ = self.text_encoder(text, embed)
        if self.text_projection_layer is not None:
            x = x @ self.text_projection_layer

        x = x / x.norm(dim=-1, keepdim=True)

        return x

    def encode_dualcoop(self, image_features, text_features):
        if self.image_projection_layer is not None:
            image_features = image_features @ self.image_projection_layer
        image_features_norm = image_features / image_features.norm(dim=1, keepdim=True)
        output = 20 * F.conv1d(image_features_norm, text_features[:, :, None])
        b, c, _ = output.shape
        output_half = output[:, : c // 2]
        w_half = F.softmax(output_half, dim=-1)
        w = torch.cat([w_half, w_half], dim=1)
        output = 5 * (output * w).sum(-1)

        b, c = output.shape

        # convert the shape of logits to [b, 2, num_class]
        logits = output.view(b, 2, c // 2)
        return logits

    def encode_SpineModellp(self, image_features, image_global, text_features, pair_features):
        pair_logits = None

        if self.image_projection_layer is not None:
            image_features = image_features @ self.image_projection_layer
        image_features = image_features.permute(0, 2, 1)  # 64,50,512
        image_features = torch.cat(
            [image_global.unsqueeze(1), image_features[:, 1:]], dim=1
        )
        if self.channel_attention is not None:
            image_features = self.channel_attention(image_features)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()

        if pair_features is not None:
            pair_logits = logit_scale * image_features @ pair_features.t()
            pair_logits = pair_logits.sum(1)  # 64,91

        output = logit_scale * image_features @ text_features.t()
        output = output.sum(1)
        bs, cls = output.shape
        logits = output.view(bs, 2, cls // 2)
        return logits, pair_logits

    def encode_fixed(self, image_features, image_global, text_features):
        if self.image_projection_layer is not None:
            image_features = image_features @ self.image_projection_layer
        image_features = image_features.permute(0, 2, 1)  # 64,50,512
        image_features = torch.cat(
            [image_global.unsqueeze(1), image_features[:, 1:]], dim=1
        )
        if self.channel_attention is not None:
            image_features = self.channel_attention(image_features)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()

        output = logit_scale * image_features @ text_features.t()
        output = output.sum(1)
        bs, cls = output.shape
        logits = output.view(bs, 2, cls // 2)
        return logits

    def stack_and_transpose(self, f, modality, i):
        slice1 = f[modality]['slices'][i]
        slice2 = f[modality]['slices'][i + 1]
        slice3 = f[modality]['slices'][i + 2]
        data = np.stack([slice1, slice2, slice3], axis=0)  # (3, H, W)
        return np.transpose(data, (1, 2, 0))  # (H, W, 3)

    def forward(self, h5_path=None, ap_img=None, lat_img=None, text=None, labels=None, is_train=True):
        # Use input tensor's device instead of hard-coded device
        if ap_img is not None:
            device = ap_img.device
        elif lat_img is not None:
            device = lat_img.device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if h5_path is not None and ap_img is not None:
            #print("=======================all========================")
            ap_image_features, ap_channel_features, ap_attn_feature = self.encode_image2(ap_img)
            lat_image_features, lat_channel_features, lat_attn_feature = self.encode_image2(lat_img)
            # xray_feature = torch.stack([ap_image_features, lat_image_features], dim=1)
            feature_a = ap_image_features + lat_image_features
            feature_b = ap_channel_features + lat_channel_features
            x_ray_feature = torch.stack([ap_image_features, lat_image_features], dim=1)

            if text is not None:
                text_features = self.encode_text(text)
            clip_logits = None
            lp_logits = None
            pair_logits = None
            pair_features = None
            if isinstance(self.prompt_learner, SpineModel_LP):
                (
                    pos_token,
                    neg_token,
                    pos_embed,
                    neg_embed,
                    pair_token,
                    pair_embed,
                ) = self.prompt_learner(feature_a)
            else:
                (
                    pos_token,
                    neg_token,
                    pos_embed,
                    neg_embed,
                    pair_token,
                    pair_embed,
                ) = self.prompt_learner()
            if pos_embed is None or neg_embed is None:
                pos_features = self.encode_text(pos_token)  # [14, 512]
                neg_features = self.encode_text(neg_token)  # [14, 512]
            else:
                pos_features = self.encode_text(pos_token, pos_embed)
                neg_features = self.encode_text(neg_token, neg_embed)

            if pair_token is not None and pair_embed is not None:
                pair_features = self.encode_text(pair_token, pair_embed)

            # Merge positive and negative features to get final text features (20, 1024)
            if isinstance(self.prompt_learner, SpineModel_LP):
                lp_features = torch.cat([pos_features, neg_features], dim=0)
                lp_logits, pair_logits = self.encode_SpineModellp(
                    feature_b, feature_a, lp_features, pair_features
                )
            else:
                lp_features = torch.cat((pos_features, neg_features), dim=0)
                logit_scale = self.logit_scale.exp()
                lp_logits = logit_scale * feature_a @ lp_features.t()
                bs, cls = lp_logits.shape
                lp_logits = lp_logits.view(bs, 2, cls // 2)

            batch_size = len(h5_path)
            # Step 1: Open all h5 files and get minimum slice count for each patient
            files = []
            min_num_slices = float('inf')
            for path in h5_path:
                f = h5py.File(path, 'r')
                files.append(f)
                # Get slice counts for each modality, take minimum as available slice count. Code requires AX, FS, T1, T2 modalities
                ax_num = f['AX']['slices'].shape[0]
                fs_num = f['FS']['slices'].shape[0]
                t1_num = f['T1']['slices'].shape[0]
                t2_num = f['T2']['slices'].shape[0]
                num_slices = min(ax_num, fs_num, t1_num, t2_num)
                if num_slices < 3:
                    raise ValueError(f"File {path} has less than 3 slices.")
                min_num_slices = min(min_num_slices, num_slices)
            max_steps = min_num_slices - 2
            # Step 2: Initialize result list
            all_features = []
            # Step 3: Iterate
            for t in range(max_steps):
                batch_features = []  # Store features of three DICOM slices for N patients in current loop
                # Iterate through each patient (batch)
                for idx in range(batch_size):
                    f = files[idx]
                    # read
                    ax_data = self.stack_and_transpose(f, 'AX', t)
                    fs_data = self.stack_and_transpose(f, 'FS', t)
                    t1_data = self.stack_and_transpose(f, 'T1', t)
                    t2_data = self.stack_and_transpose(f, 'T2', t)
                    if is_train:
                        ax_data, fs_data, t1_data, t2_data = DicomProcess(ax_data, fs_data, t1_data, t2_data, is_train=True)
                    else:
                        ax_data, fs_data, t1_data, t2_data = DicomProcess(ax_data, fs_data, t1_data, t2_data,
                                                                          is_train=False)
                    ax_image_features, ax_channel_features, ax_attn_feature = self.encode_image(
                        ax_data.unsqueeze(0).to(device))  # First change ax_data shape to (1, 3, H, W).
                    fs_image_features, fs_channel_features, fs_attn_feature = self.encode_image(
                        fs_data.unsqueeze(0).to(device))  # Output at this step is (1, 1024)
                    t1_image_features, t1_channel_features, t1_attn_feature = self.encode_image(
                        t1_data.unsqueeze(0).to(device))
                    t2_image_features, t2_channel_features, t2_attn_feature = self.encode_image(
                        t2_data.unsqueeze(0).to(device))
                    concat_features = torch.cat(
                        [ax_image_features, fs_image_features, t1_image_features, t2_image_features], dim=0)  # Output is (4, 1024)
                    batch_features.append(concat_features)
                # Stack features of four modalities from three DICOM slices for N patients together
                step_feature = torch.stack(batch_features, dim=0)  # (N, 4, 1024)
                all_features.append(step_feature)  # List, each element has shape (8, 4, 1024)

            # Step 4: Calculate attention scores
            # lp_features shape is (20, 1024), we need to convert it to a form suitable for attention calculation
            # Split lp_features into positive and negative parts, each with 10 classes
            pos_lp_features = lp_features[:lp_features.shape[0] // 2]  # (10, 1024)
            neg_lp_features = lp_features[lp_features.shape[0] // 2:]  # (10, 1024)
            # Convert all_features to tensor form: (time steps, batch_size, 4, 1024)
            all_features_tensor = torch.stack(all_features, dim=0)
            # final_features = torch.mean(all_features_tensor, dim=0)   # Only used in slice selection ablation experiments
            # Calculate attention scores (start slice selection task)
            attention_scores = []
            T, N, M, D = all_features_tensor.shape  # T: time steps, N: batch_size, M: modalities (4), D: feature dim (1024)
            # Calculate attention scores for each time step
            for t in range(T):
                # Current time step features: (N, 4, 1024)
                step_features = all_features_tensor[t]  # (N, 4, 1024)
                # Apply attention formula: s_(k,t)^(m) = u_m^T tanh(W_f^(m)f_t^(m) + W_e^(m)e_k)
                # f_t^(m): features of each modality (N, 1024)
                # e_k: text features lp_features (20, 1024)
                modal_attention_scores = []  # Store attention scores for each modality across all texts
                for m in range(M):  # For each modality
                    # Get current modality features (N, 1024)
                    modality_features = step_features[:, m, :]  # (N, 1024)
                    # W_f^(m)f_t^(m): apply linear transformation for each modality
                    transformed_modality_features = self.W_f[m](modality_features)  # (N, 1024)
                    # W_e^(m)e_k: apply linear transformation for text features
                    transformed_text_features = self.W_e(lp_features)  # (20, 1024)
                    # W_f^(m)f_t^(m) + W_e^(m)e_k
                    # Need broadcasting operation: (N, 1, 1024) + (1, 20, 1024) -> (N, 20, 1024)
                    combined_features = transformed_modality_features.unsqueeze(1) + transformed_text_features.unsqueeze(0)
                    # tanh(W_f^(m)f_t^(m) + W_e^(m)e_k)
                    tanh_features = torch.tanh(combined_features)  # (N, 20, 1024)
                    # u_m^T tanh(...)
                    # u_m: (1024,) for each modality
                    modality_attention = torch.matmul(tanh_features, self.u_m[m])  # (N, 20)
                    modal_attention_scores.append(modality_attention)
                # Merge attention scores from 4 modalities: (N, 4, 20)
                modal_attention_scores = torch.stack(modal_attention_scores, dim=1)
                # Merge attention scores from all modalities into one score
                # Method: average across all modalities to get overall attention scores for each time step, each sample, for all text features
                aggregated_attention_scores = torch.mean(modal_attention_scores, dim=1)  # (N, 20)
                attention_scores.append(aggregated_attention_scores)
            # Stack attention scores: (T, N, 20)
            attention_scores = torch.stack(attention_scores, dim=0)
            # Apply softmax to get attention weights α_(k,t)^(m)
            # Apply softmax on the last dimension (text feature dimension)
            attention_weights = F.softmax(attention_scores, dim=-1)  # (T, N, 20)
            # Step 5: Find time step with highest attention score and select corresponding features from all modalities
            # Calculate mean attention weights for each time step: (T, N)
            mean_attention_weights = attention_weights.mean(dim=-1)  # Average on text dimension
            # Find time step index with highest score for each sample: (N,)
            best_time_steps = torch.argmax(mean_attention_weights, dim=0)
            # Select features based on best time step: (N, 4, 1024)
            selected_features = []
            for i in range(N):
                t_idx = best_time_steps[i]
                # Select all modality features for this time step and sample
                features = all_features_tensor[t_idx, i]  # (4, 1024)
                selected_features.append(features)
            # Stack selected features: (N, 4, 1024)
            final_features = torch.stack(selected_features, dim=0)
            # fused_disease_representations = final_features   # Only used in ablation experiments removing modality selection
            # prior_loss = 0.0  # Only used in ablation experiments removing modality selection
            # Start modality selection task!!!!!
            # Ensure batch dimensions of both tensors match before concatenation
            if final_features.size(0) == x_ray_feature.size(0):
                final_features = torch.cat([final_features, x_ray_feature], dim=1)
            else:
                # Adjust to smaller batch size to ensure dimension matching
                min_batch_size = min(final_features.size(0), x_ray_feature.size(0))
                final_features = torch.cat([final_features[:min_batch_size], x_ray_feature[:min_batch_size]], dim=1)
            lambda_prior = np.ones((25, 6)) * 0.1  # Adjustable
            if is_train:
                selected_modality_indices, attention_weights, prior_loss, fused_disease_representations = self.all_label_aware_modality_attention(
                    final_features, lp_features,
                    expert_prior_matrix=self.all_expert_prior_matrix,
                    lambda_prior=lambda_prior,
                    labels=labels,
                    is_train=True
                )
            else:
                selected_modality_indices, attention_weights, prior_loss, fused_disease_representations = self.all_label_aware_modality_attention(
                    final_features, lp_features,
                    expert_prior_matrix=self.all_expert_prior_matrix,
                    lambda_prior=lambda_prior,
                    labels=labels,
                    is_train=False
                )
        elif h5_path is None and ap_img is not None:
            #print("=======================X_Ray========================")
            ap_image_features, ap_channel_features, ap_attn_feature = self.encode_image2(ap_img)
            lat_image_features, lat_channel_features, lat_attn_feature = self.encode_image2(lat_img)
            # xray_feature = torch.stack([ap_image_features, lat_image_features], dim=1)
            feature_a = ap_image_features + lat_image_features
            feature_b = ap_channel_features + lat_channel_features
            x_ray_feature = torch.stack([ap_image_features, lat_image_features], dim=1)

            if text is not None:
                text_features = self.encode_text(text)
            clip_logits = None
            lp_logits = None
            pair_logits = None
            pair_features = None
            if isinstance(self.prompt_learner, SpineModel_LP):
                (
                    pos_token,
                    neg_token,
                    pos_embed,
                    neg_embed,
                    pair_token,
                    pair_embed,
                ) = self.prompt_learner(feature_a)
            else:
                (
                    pos_token,
                    neg_token,
                    pos_embed,
                    neg_embed,
                    pair_token,
                    pair_embed,
                ) = self.prompt_learner()
            if pos_embed is None or neg_embed is None:
                pos_features = self.encode_text(pos_token)  # [14, 512]
                neg_features = self.encode_text(neg_token)  # [14, 512]
            else:
                pos_features = self.encode_text(pos_token, pos_embed)
                neg_features = self.encode_text(neg_token, neg_embed)

            if pair_token is not None and pair_embed is not None:
                pair_features = self.encode_text(pair_token, pair_embed)

            # Merge positive and negative features to get final text features (20, 1024)
            if isinstance(self.prompt_learner, SpineModel_LP):
                lp_features = torch.cat([pos_features, neg_features], dim=0)
                lp_logits, pair_logits = self.encode_SpineModellp(
                    feature_b, feature_a, lp_features, pair_features
                )
            if isinstance(self.prompt_learner, ContextOptimization):
                lp_features = torch.cat([pos_features, neg_features], dim=0)
                lp_logits = self.encode_dualcoop(feature_b, lp_features)
            else:
                lp_features = torch.cat((pos_features, neg_features), dim=0)
                logit_scale = self.logit_scale.exp()
                lp_logits = logit_scale * feature_a @ lp_features.t()
                bs, cls = lp_logits.shape
                lp_logits = lp_logits.view(bs, 2, cls // 2)

            # Start modality selection task!!!!!
            final_features = x_ray_feature
            lambda_prior = np.ones((25, 2)) * 0.1  # Adjustable
            if is_train:
                selected_modality_indices, attention_weights, prior_loss, fused_disease_representations = self.xray_label_aware_modality_attention(
                    final_features, lp_features,
                    expert_prior_matrix=self.xray_expert_prior_matrix,
                    lambda_prior=lambda_prior,
                    labels=labels,
                    is_train=True
                )
            else:
                selected_modality_indices, attention_weights, prior_loss, fused_disease_representations = self.xray_label_aware_modality_attention(
                    final_features, lp_features,
                    expert_prior_matrix=self.xray_expert_prior_matrix,
                    lambda_prior=lambda_prior,
                    labels=labels,
                    is_train=False
                )
        elif h5_path is not None and ap_img is None:
            #print("=======================MRI========================")
            batch_size = len(h5_path)
            # Step 1: Open all h5 files and get minimum slice count for each patient
            files = []
            min_num_slices = float('inf')
            for path in h5_path:
                f = h5py.File(path, 'r')
                files.append(f)
                # Get slice counts for each modality, take minimum as available slice count. Code requires AX, FS, T1, T2 modalities
                ax_num = f['AX']['slices'].shape[0]
                fs_num = f['FS']['slices'].shape[0]
                t1_num = f['T1']['slices'].shape[0]
                t2_num = f['T2']['slices'].shape[0]
                num_slices = min(ax_num, fs_num, t1_num, t2_num)
                if num_slices < 3:
                    raise ValueError(f"File {path} has less than 3 slices.")
                min_num_slices = min(min_num_slices, num_slices)
            max_steps = min_num_slices - 2
            # Step 2: Initialize result list
            all_features = []
            channel_all_features = []
            # Step 3: Iterate
            for t in range(max_steps):
                batch_features = []  # Store features of three DICOM slices for N patients in current loop
                channel_batch_features = []
                # Iterate through each patient (batch)
                for idx in range(batch_size):
                    f = files[idx]
                    # Read three slices from four modalities
                    ax_data = self.stack_and_transpose(f, 'AX', t)
                    fs_data = self.stack_and_transpose(f, 'FS', t)
                    t1_data = self.stack_and_transpose(f, 'T1', t)
                    t2_data = self.stack_and_transpose(f, 'T2', t)
                    if is_train:
                        ax_data, fs_data, t1_data, t2_data = DicomProcess(ax_data, fs_data, t1_data, t2_data,
                                                                          is_train=True)
                    else:
                        ax_data, fs_data, t1_data, t2_data = DicomProcess(ax_data, fs_data, t1_data, t2_data,
                                                                          is_train=False)
                    ax_image_features, ax_channel_features, ax_attn_feature = self.encode_image(ax_data.unsqueeze(0).to(device))  # First change ax_data shape to (1, 3, H, W).
                    fs_image_features, fs_channel_features, fs_attn_feature = self.encode_image(fs_data.unsqueeze(0).to(device))  # Output at this step is (1, 1024)
                    t1_image_features, t1_channel_features, t1_attn_feature = self.encode_image(t1_data.unsqueeze(0).to(device))
                    t2_image_features, t2_channel_features, t2_attn_feature = self.encode_image(t2_data.unsqueeze(0).to(device))
                    concat_features = torch.cat([ax_image_features, fs_image_features, t1_image_features, t2_image_features], dim=0)  # Output is (4, 1024)
                    channel_concat_features = torch.cat([ax_channel_features, fs_channel_features, t1_channel_features, t2_channel_features], dim=0)  # Output is (4, 1024)
                    batch_features.append(concat_features)
                    channel_batch_features.append(channel_concat_features)
                # Stack features of four modalities from three DICOM slices for N patients together
                step_feature = torch.stack(batch_features, dim=0)  # (N, 4, 1024)
                all_features.append(step_feature)  # List, each element has shape (8, 4, 1024)
                channel_step_feature = torch.stack(channel_batch_features, dim=0)  # (N, 4, 1024)
                channel_all_features.append(channel_step_feature)  # List, each element has shape (8, 4, 1024)

            # Convert all_features to tensor form: (time steps, batch_size, 4, 1024)
            all_features_tensor = torch.stack(all_features, dim=0)
            all_features_tensor_result_mean = torch.mean(all_features_tensor[all_features_tensor.shape[0] // 2], dim=1)
            channel_all_features_tensor = torch.stack(channel_all_features, dim=0)
            channel_all_features_tensor_result_mean = torch.mean(channel_all_features_tensor[channel_all_features_tensor.shape[0] // 2], dim=1)
            if text is not None:
                text_features = self.encode_text(text)
            clip_logits = None
            lp_logits = None
            pair_logits = None
            pair_features = None
            if isinstance(self.prompt_learner, SpineModel_LP):
                (
                    pos_token,
                    neg_token,
                    pos_embed,
                    neg_embed,
                    pair_token,
                    pair_embed,
                ) = self.prompt_learner(all_features_tensor_result_mean)
            else:
                (
                    pos_token,
                    neg_token,
                    pos_embed,
                    neg_embed,
                    pair_token,
                    pair_embed,
                ) = self.prompt_learner()
            if pos_embed is None or neg_embed is None:
                pos_features = self.encode_text(pos_token)  # [14, 512]
                neg_features = self.encode_text(neg_token)  # [14, 512]
            else:
                pos_features = self.encode_text(pos_token, pos_embed)
                neg_features = self.encode_text(neg_token, neg_embed)

            if pair_token is not None and pair_embed is not None:
                pair_features = self.encode_text(pair_token, pair_embed)

            # Merge positive and negative features to get final text features (20, 1024)
            if isinstance(self.prompt_learner, SpineModel_LP):
                lp_features = torch.cat([pos_features, neg_features], dim=0)
                lp_logits, pair_logits = self.encode_SpineModellp(
                    channel_all_features_tensor_result_mean, all_features_tensor_result_mean, lp_features, pair_features
                )
            else:
                lp_features = torch.cat((pos_features, neg_features), dim=0)
                logit_scale = self.logit_scale.exp()
                lp_logits = logit_scale * all_features_tensor_result_mean @ lp_features.t()
                bs, cls = lp_logits.shape
                lp_logits = lp_logits.view(bs, 2, cls // 2)

            # Calculate attention scores
            # lp_features shape is (20, 1024), we need to convert it to a form suitable for attention calculation
            # Split lp_features into positive and negative parts, each with 10 classes
            pos_lp_features = lp_features[:lp_features.shape[0] // 2]  # (10, 1024)
            neg_lp_features = lp_features[lp_features.shape[0] // 2:]  # (10, 1024)
            # Calculate attention scores
            attention_scores = []
            T, N, M, D = all_features_tensor.shape  # T: time steps, N: batch_size, M: modalities (4), D: feature dim (1024)
            # Calculate attention scores for each time step
            for t in range(T):
                # Current time step features: (N, 4, 1024)
                step_features = all_features_tensor[t]  # (N, 4, 1024)
                # Apply attention formula: s_(k,t)^(m) = u_m^T tanh(W_f^(m)f_t^(m) + W_e^(m)e_k)
                # f_t^(m): features of each modality (N, 1024)
                # e_k: text features lp_features (20, 1024)
                modal_attention_scores = []  # Store attention scores for each modality across all texts
                for m in range(M):  # For each modality
                    # Get current modality features (N, 1024)
                    modality_features = step_features[:, m, :]  # (N, 1024)
                    # W_f^(m)f_t^(m): apply linear transformation for each modality
                    transformed_modality_features = self.W_f[m](modality_features)  # (N, 1024)
                    # W_e^(m)e_k: apply linear transformation for text features
                    transformed_text_features = self.W_e(lp_features)  # (20, 1024)
                    # W_f^(m)f_t^(m) + W_e^(m)e_k
                    # Need broadcasting operation: (N, 1, 1024) + (1, 20, 1024) -> (N, 20, 1024)
                    combined_features = transformed_modality_features.unsqueeze(
                        1) + transformed_text_features.unsqueeze(0)
                    # tanh(W_f^(m)f_t^(m) + W_e^(m)e_k)
                    tanh_features = torch.tanh(combined_features)  # (N, 20, 1024)
                    # u_m^T tanh(...)
                    # u_m: (1024,) for each modality
                    modality_attention = torch.matmul(tanh_features, self.u_m[m])  # (N, 20)
                    modal_attention_scores.append(modality_attention)
                # Merge attention scores from 4 modalities: (N, 4, 20)
                modal_attention_scores = torch.stack(modal_attention_scores, dim=1)
                # Merge attention scores from all modalities into one score
                # Method: average across all modalities to get overall attention scores for each time step, each sample, for all text features
                aggregated_attention_scores = torch.mean(modal_attention_scores, dim=1)  # (N, 20)
                attention_scores.append(aggregated_attention_scores)
            # Stack attention scores: (T, N, 20)
            attention_scores = torch.stack(attention_scores, dim=0)
            # Apply softmax to get attention weights α_(k,t)^(m)
            # Apply softmax on the last dimension (text feature dimension)
            attention_weights = F.softmax(attention_scores, dim=-1)  # (T, N, 20)
            # Step 5: Find time step with highest attention score and select corresponding features from all modalities
            # Calculate mean attention weights for each time step: (T, N)
            mean_attention_weights = attention_weights.mean(dim=-1)  # Average on text dimension
            # Find time step index with highest score for each sample: (N,)
            best_time_steps = torch.argmax(mean_attention_weights, dim=0)
            # Select features based on best time step: (N, 4, 1024)
            selected_features = []
            for i in range(N):
                t_idx = best_time_steps[i]
                # Select all modality features for this time step and sample
                features = all_features_tensor[t_idx, i]  # (4, 1024)
                selected_features.append(features)
            # Stack selected features: (N, 4, 1024)
            final_features = torch.stack(selected_features, dim=0)

            # Start modality selection task
            lambda_prior = np.ones((25, 4)) * 0.1  # Adjustable
            if is_train:
                selected_modality_indices, attention_weights, prior_loss, fused_disease_representations = self.mri_label_aware_modality_attention(
                    final_features, lp_features,
                    expert_prior_matrix=self.mri_expert_prior_matrix,
                    lambda_prior=lambda_prior,
                    labels=labels,
                    is_train=True
                )
            else:
                selected_modality_indices, attention_weights, prior_loss, fused_disease_representations = self.mri_label_aware_modality_attention(
                    final_features, lp_features,
                    expert_prior_matrix=self.mri_expert_prior_matrix,
                    lambda_prior=lambda_prior,
                    labels=labels,
                    is_train=False
                )
        # Subsequent processing enters MoE to complete classification task
        fused_disease_representations = fused_disease_representations.mean(dim=1)  # Shape becomes (batchsize, 1024)
                
        # Ensure batch dimensions of fused_disease_representations and final_features match
        if fused_disease_representations.size(0) != final_features.size(0):
            min_batch = min(fused_disease_representations.size(0), final_features.size(0))
            fused_disease_representations = fused_disease_representations[:min_batch]
            final_features = final_features[:min_batch]
        
        # Split along dimension 1 into a list of length N
        N = final_features.size(1)  # Get size of dimension 1
        final_features_list = [final_features[:, i, :] for i in range(N)]  # Convert to list, each element has shape (batchsize, 1024)
        final_output = self.moe(fused_disease_representations, pos_features, final_features_list)
        # final_output = self.no_moe(fused_disease_representations)  # Only used in ablation experiments without MoE

        return clip_logits, lp_logits, pair_logits, final_output, prior_loss
