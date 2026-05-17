import time
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from validator import validate
from utils.losses import CLIPLoss, AsymmetricLoss, LossMixture, LossRanking, AsymmetricLossOffice
from model.text_encoder import preprocess_text
from helper import AverageMeter
from utils.metrics import calc_map


def train_fixed(model, cfg, h5_path, ap_img, lap_img, target):
    """
    Train with fixed prompts (e.g., hand-crafted positive/negative prompts).
    Only uses the main classification logits (lp_logits).
    """
    criterion = AsymmetricLossOffice(
        gamma_neg=cfg.LOSS.ASL_GAMMA_NEG,
        gamma_pos=cfg.LOSS.ASL_GAMMA_POS,
        clip=0.05
    )
    with autocast():
        _, lp_logits, _, final_output, prior_loss = model(h5_path, ap_img, lap_img,  labels=target, is_train=True)

    # Merge positive and negative logits
    lp_logits = lp_logits[:, 0, :] - lp_logits[:, 1, :]  # [B, 25]
    loss = criterion(lp_logits, target)
    return loss, lp_logits

def train_clip(model, h5_path, ap_img, lap_img, tokenized_texts, target):
    """
    this function calc the loss of clip, only used when finetuning clip.

    we perform the contrastive learning described in https://arxiv.org/abs/2103.00020

    Args:
        ..., tokenized_texts : tokenized(preprocessed) report texts
    Returns:
        loss
    """

    criterion = CLIPLoss()
    with autocast():
        image_logits, _, _, _, _ = model(h5_path, ap_img, lap_img, tokenized_texts,  labels=target, is_train=True)
    text_logits = image_logits.t()
    loss = criterion(image_logits, text_logits)
    return loss


def train_coop(model, cfg, h5_path, ap_img, lap_img, target):
    """
    this function calc the loss of CoOp.

    used when learning prompts in DualCoOp's way, for reproducing DualCoOp in medical task

    Args:
        model (_type_): _description_
        cfg (_type_): _description_
        images (_type_): _description_
        target (_type_): _description_

    Returns:

    """
    criterion = AsymmetricLoss(
        gamma_neg=cfg.LOSS.ASL_GAMMA_NEG, gamma_pos=cfg.LOSS.ASL_GAMMA_POS
    )
    with autocast():
        _, lp_logits, pair_logits, final_output, prior_loss = model(h5_path, ap_img, lap_img,  labels=target, is_train=True)
    loss = 0.1 * criterion(lp_logits, target)
    loss2 = 0.1 * criterion(final_output, target)
    loss = loss + loss2
    return loss, lp_logits


def train_SpineModel(model, cfg, h5_path, ap_img, lap_img, target):
    if cfg.PROMPT.ENABLE_PAIRLOSS:
        criterion = LossMixture(
            gamma_neg=cfg.LOSS.ASL_GAMMA_NEG,
            gamma_pos=cfg.LOSS.ASL_GAMMA_POS,
        )
    else:
        criterion = LossRanking(
            gamma_neg=cfg.LOSS.ASL_GAMMA_NEG,
            gamma_pos=cfg.LOSS.ASL_GAMMA_POS,
        )
    with autocast():
        _, lp_logits, pair_logits, final_output, prior_loss = model(h5_path, ap_img, lap_img,  labels=target, is_train=True)
    loss1 = criterion(lp_logits, target, pair_logits)
    loss2 = criterion(final_output, target, pair_logits)
    loss = (loss1 + loss2)/2 #+ 0.1*prior_loss
    return loss, final_output


def set_model_train(model, cfg):
    """
    refer to cfg, select a part of the model that not involved in training
    """
    if not isinstance(model, nn.DataParallel):
        if hasattr(cfg, 'TRAIN') and hasattr(cfg.TRAIN, 'FINETUNE_CLIP') and not cfg.TRAIN.FINETUNE_CLIP:
            # If not fine-tuning CLIP, set encoders to eval mode
            model.image_encoder.eval()
            model.text_encoder.eval()
        else:
            # If fine-tuning CLIP, ensure encoders are in train mode
            model.image_encoder.train()
            model.text_encoder.train()

        if not cfg.MODEL.ENABLE_LP:
            model.prompt_learner.eval()
    else:
        if hasattr(cfg, 'TRAIN') and hasattr(cfg.TRAIN, 'FINETUNE_CLIP') and not cfg.TRAIN.FINETUNE_CLIP:
            # If not fine-tuning CLIP, set encoders to eval mode
            model.module.image_encoder.eval()
            model.module.text_encoder.eval()
        else:
            # If fine-tuning CLIP, ensure encoders are in train mode
            model.module.image_encoder.train()
            model.module.text_encoder.train()

        if not cfg.MODEL.ENABLE_LP:
            model.module.prompt_learner.eval()


def train_epoch(
    train_loader,
    val_loader,
    model,
    optimizer,
    cfg,
    model_ema=None,
):
    losses = AverageMeter()
    train_mAP = AverageMeter()
    Softmax = torch.nn.Softmax(dim=1)
    sigmoid = torch.nn.Sigmoid()
    # Use the device already assigned to the model instead of hard-coding
    # If the model hasn't been assigned to a device yet, decide based on CUDA availability
    if next(model.parameters()).is_cuda:
        device = next(model.parameters()).device
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)
    model.train()
    set_model_train(model, cfg)
    timestamp = time.time()
    epoch_time = time.time()
    accumulation_steps = cfg.TRAIN.ACCUMULATION_STEPS

    for i, (h5_path, ap_img, lap_img, texts, target) in enumerate(train_loader):
        # Ensure data is on the correct device
        if next(model.parameters()).is_cuda:
            device = next(model.parameters()).device
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        
        ap_images = ap_img.to(device)
        lap_images = lap_img.to(device)
        tokenized_texts = preprocess_text(texts, cfg.MODEL)
        tokenized_texts = {
            "input_ids": tokenized_texts["input_ids"].to(device),
            "attention_mask": tokenized_texts["attention_mask"].to(device),
        }
        target = target.to(device)
        loss = None
        output = None
        if cfg.TRAIN.FINETUNE_CLIP and not cfg.MODEL.ENABLE_LP:
            loss = train_clip(model, h5_path, ap_img, lap_img, tokenized_texts, target)
        elif cfg.MODEL.PROMPT.NAME == "coop":
            loss, output = train_coop(model, cfg.MODEL, h5_path, ap_images, lap_images, target)
        elif cfg.MODEL.PROMPT.NAME == "SpineModel":
            loss, output = train_SpineModel(model, cfg.MODEL, h5_path, ap_images, lap_images, target)
        elif cfg.MODEL.PROMPT.NAME == "linear":
            loss, output = train_coop(model, cfg.MODEL, h5_path, ap_images, lap_images, target)
        elif cfg.MODEL.PROMPT.NAME == "fixed":
            loss, output = train_fixed(model, cfg.MODEL, h5_path, ap_images, lap_images, target)

        if output is not None:
            if output.dim() == 3:
                pred = Softmax(output.detach())[:, 0]
            else:
                pred = sigmoid(output.detach())
            num_classes = target.shape[1]
            mAP_value, _ = calc_map(
                target.cpu().numpy(), pred.cpu().numpy(), num_classes
            )
            train_mAP.update(mAP_value, ap_images.size(0))
        losses.update(loss.item(), ap_images.size(0))
        if accumulation_steps > 1:
            loss = loss / accumulation_steps
            loss.backward()
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if model_ema is not None:
                model_ema.update(model)
        if i % cfg.TRAIN.PRINT_FREQ == 0:
            info = f"Train: [{i}/{len(train_loader)}]\tTime:{(time.time()-timestamp):.2f}s\tLoss(avg):{losses.avg:.4f}\t"
            if cfg.MODEL.ENABLE_LP:
                info = info + f"mAP(avg):{train_mAP.avg:.4f}"
            print(info)
            timestamp = time.time()
        if (
            cfg.TRAIN.VAL_FREQ_IN_EPOCH != -1
            and (i + 1) % cfg.TRAIN.VAL_FREQ_IN_EPOCH == 0
        ):
            result_dict = validate(val_loader, model, cfg)
            macro_auc = result_dict["macro_auc"]
            micro_auc = result_dict["micro_auc"]
            map_score = result_dict["mAP_score"]
            mcc_score = result_dict["mcc_score"]
            acc_score = result_dict["acc_score"]
            macro_pcs = result_dict["macro_pcs"]
            micro_pcs = result_dict["micro_pcs"]
            macro_rc = result_dict["macro_rc"]
            micro_rc = result_dict["micro_rc"]
            marco_f1 = result_dict["marco_f1"]
            micro_f1 = result_dict["micro_f1"]

            val_info = (
                f"Test: [{i+1}/{len(train_loader)}]\t"
                f"Macro_AUC: {macro_auc:.2f}\t"
                f"Micro_AUC: {micro_auc:.2f}\t"
                f"mAP: {map_score:.2f}\t"
                f"MCC: {mcc_score:.2f}\t"
                f"Accuracy: {acc_score:.2f}\t"
                f"Macro_Precision: {macro_pcs:.2f}\t"
                f"Micro_Precision: {micro_pcs:.2f}\t"
                f"Macro_Recall: {macro_rc:.2f}\t"
                f"Micro_Recall: {micro_rc:.2f}\t"
                f"Macro_F1: {marco_f1:.2f}\t"
                f"Micro_F1: {micro_f1:.2f}\t"
                f"with threshold: {cfg.TEST.THRESHOLD:.2f}"
            )
            print(val_info)
            model.train()
            set_model_train(model, cfg)
    epoch_time = time.time() - epoch_time
    return losses, epoch_time, train_mAP
