import json
import os
import torch
import time
from torch.cuda.amp import autocast
from helper import write_logfile

from utils.metrics import (
    calc_roc,
    calc_map,
    fbetaMacro,
    fbetaMicro,
    Average_MCC,
    accuracyMacro,
    precisionMacro,
    precisionMicro,
    recallMacro,
    recallMicro,
    compute_f1_thresholds,
    compute_dynamic_thresholds,  # New import
)


def threshold_to_binary(y_pred, threshold):
    binary_pred = (y_pred > threshold).astype(int)
    return binary_pred


def threshold_to_binary_per_class(y_pred, thresholds):
    """
    Convert predictions to binary values using per-class thresholds
    Args:
    - y_pred: prediction probabilities with shape (N, C)
    - thresholds: thresholds for each class with shape (C,)
    Returns:
    - binary_pred: binary predictions with shape (N, C)
    """
    binary_pred = (y_pred >= thresholds).astype(int)
    return binary_pred


def generate_json_output(y_pred, y_label, y_true):
    num_samples = y_pred.shape[0]

    json_data = []

    for i in range(num_samples):
        sample_data = {
            "id": i + 1,
            "probabilities": y_pred[i, :].tolist(),
            "labels": y_label[i, :].tolist(),
            "ground_truth": y_true[i, :].tolist(),
        }

        json_data.append(sample_data)

    return json_data


def build_output_file(cfg, json_data, model_name=None):
    folder = os.path.join(cfg.OUTPUT_DIR, model_name)
    if not os.path.exists(folder):
        os.makedirs(folder)

    file_path = os.path.join(folder, model_name + "_output.json")
    with open(file_path, "w") as json_file:
        json.dump(json_data, json_file)


# def save_detailed_predictions(cfg, val_loader, model, y_pred, y_label, y_true, model_name=None):
#     """
#     Save detailed validation set prediction results to Excel file, including prediction probabilities and labels for 25 diseases for each sample
#
#     Args:
#     - cfg: configuration object
#     - val_loader: validation data loader
#     - model: model object
#     - y_pred: prediction probability array with shape (N, 25)
#     - y_label: prediction label array with shape (N, 25)
#     - y_true: ground truth label array with shape (N, 25)
#     - model_name: model name
#     """
#     try:
#         import pandas as pd
#     except ImportError:
#         print("Warning: pandas not installed, cannot save Excel file. Please run 'pip install pandas openpyxl'")
#         return None
#
#     # Get disease names
#     disease_names = val_loader.dataset.classnames
#
#     # Prepare Excel data
#     excel_data = []
#
#     with torch.no_grad():
#         sample_idx = 0
#         for i, (h5_path, ap_img, lap_img, _, target) in enumerate(val_loader):
#             batch_size = target.size(0)
#
#             for j in range(batch_size):
#                 if sample_idx >= len(y_pred):
#                     break
#
#                 # Get sample path
#                 sample_path = h5_path[j] if isinstance(h5_path, list) else h5_path
#
#                 # Create a record row for each disease
#                 for disease_idx, disease_name in enumerate(disease_names):
#                     row_data = {
#                         "Sample ID": sample_idx + 1,
#                         "H5 File Path": sample_path,
#                         "Disease Name": disease_name,
#                         "Disease Index": disease_idx,
#                         "Positive Class Probability (Diseased)": float(y_pred[sample_idx, disease_idx]),
#                         "Negative Class Probability (Not Diseased)": float(1 - y_pred[sample_idx, disease_idx]),
#                         "Predicted Label": int(y_label[sample_idx, disease_idx]),
#                         "Ground Truth Label": int(y_true[sample_idx, disease_idx])
#                     }
#                     excel_data.append(row_data)
#
#                 sample_idx += 1
#
#     # Create DataFrame
#     df = pd.DataFrame(excel_data)
#
#     # Save to Excel file
#     folder = os.path.join(cfg.OUTPUT_DIR, model_name)
#     if not os.path.exists(folder):
#         os.makedirs(folder)
#
#     excel_file_path = os.path.join(folder, model_name + "_detailed_predictions.xlsx")
#     df.to_excel(excel_file_path, index=False, engine='openpyxl')
#
#     print(f"Detailed prediction results saved to Excel file: {excel_file_path}")
#     print(f"Total {len(df)} records saved ({len(y_pred)} samples × {len(disease_names)} diseases)")
#     return excel_file_path
def validate(val_loader, model, cfg, model_name=None):
    preds = []
    targets = []
    collected_feats = []  # New: collect features
    Softmax = torch.nn.Softmax(dim=1)
    sigmoid = torch.nn.Sigmoid()
    # Use the device already assigned to the model instead of hard-coding
    if next(model.parameters()).is_cuda:
        device = next(model.parameters()).device
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)
    model.eval()
    # Get disease names for detailed prediction saving
    disease_names = val_loader.dataset.classnames
    print(f"Detected {len(disease_names)} diseases: {disease_names}")
    timestamp = time.time()
    with torch.no_grad():
        for i, (h5_path, ap_img, lap_img, _, target) in enumerate(val_loader):
            # Ensure data is on the correct device
            if next(model.parameters()).is_cuda:
                device = next(model.parameters()).device
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
            
            ap_images = ap_img.to(device)
            lap_images = lap_img.to(device)
            target = target.to(device)
            # ap_images = None
            # h5_path = None
            with autocast():
                _, outputs, _, final_output, _ = model(h5_path, ap_images, lap_images, labels=target, is_train=False)
            # Save the 25-D features used for classification
            collected_feats.append(final_output.cpu())
            if final_output.dim() == 3:
                final_output = Softmax(final_output).cpu()[:, 0]
            else:
                final_output = sigmoid(final_output).cpu()
            preds.append(final_output.cpu())
            targets.append(target.cpu())
            batch_time = time.time() - timestamp
            timestamp = time.time()

        y_pred = torch.cat(preds).numpy()
        ground_truth = torch.cat(targets)
        y_true = ground_truth.numpy()
        num_classes = ground_truth.shape[1]
        # # ===== New: Save features and labels to disk at once =====
        # import pandas as pd
        # dump_dir = os.path.join("/data4/whz/output/", "tmp")
        # os.makedirs(dump_dir, exist_ok=True)
        #
        # # 1) Save original 25-class features and labels
        # feat_mat = torch.cat(collected_feats).numpy()  # (N, 25)
        # df_feat = pd.DataFrame(feat_mat, columns=val_loader.dataset.classnames)
        # df_feat.to_csv(os.path.join(dump_dir, 'features_25d.csv'), index=False)
        #
        # df_label = pd.DataFrame(y_true, columns=val_loader.dataset.classnames)
        # df_label.to_csv(os.path.join(dump_dir, 'labels_25d.csv'), index=False)
        # print(f'[Info] Features and labels saved to {dump_dir}, t-SNE can be plotted offline later.')

        # Use per-class threshold calculation
        PER_CLASS_THRESHOLD = False
        # Use dynamic threshold strategy
        DYNAMIC_THRESHOLD = False
        if DYNAMIC_THRESHOLD:
            # Use dynamic threshold strategy
            optimal_thresholds = compute_dynamic_thresholds(y_true, y_pred)
            y_label = threshold_to_binary_per_class(y_pred, optimal_thresholds)
        elif PER_CLASS_THRESHOLD:
            # Calculate optimal F1 threshold for each class on validation set
            optimal_thresholds = compute_f1_thresholds(y_true, y_pred)
            y_label = threshold_to_binary_per_class(y_pred, optimal_thresholds)
        else:
            # Use fixed threshold
            y_label = threshold_to_binary(y_pred, cfg.TEST.THRESHOLD)
        
        macro_auc, micro_auc, aucs = calc_roc(y_true, y_pred, num_classes)
        top_10_auc = sorted(aucs, reverse=True)[:10]
        top_10_mean_auc = sum(top_10_auc) / len(top_10_auc)
        map_score,_ = calc_map(y_true, y_pred, num_classes)
        mcc_score = Average_MCC(y_true, y_label, num_classes)
        acc_score, _ = accuracyMacro(y_true, y_label)
        macro_pcs, _ = precisionMacro(y_true, y_label)
        micro_pcs = precisionMicro(y_true, y_label)
        macro_rc, _ = recallMacro(y_true, y_label)
        micro_rc = recallMicro(y_true, y_label)
        marco_f1, _ = fbetaMacro(y_true, y_label)
        micro_f1 = fbetaMicro(y_true, y_label)
        # SAVE_PRED = True
        # if SAVE_PRED:
        #     # Save original simple format
        #     json_data = generate_json_output(y_pred, y_label, y_true)
        #     build_output_file(cfg, json_data, model_name)
        #     print("JSON file generated successfully.")
        #     # Save detailed Excel prediction results
        #     save_detailed_predictions(cfg, val_loader, model, y_pred, y_label, y_true, model_name)
    result_dict = {
        "macro_auc": macro_auc,
        "micro_auc": micro_auc,
        "top10_auc": top_10_mean_auc,
        "mAP_score": map_score,
        "mcc_score": mcc_score,
        "acc_score": acc_score,
        "macro_pcs": macro_pcs,
        "micro_pcs": micro_pcs,
        "macro_rc": macro_rc,
        "micro_rc": micro_rc,
        "marco_f1": marco_f1,
        "micro_f1": micro_f1,
    }
    torch.cuda.empty_cache()
    return result_dict


def run_val(epoch, logger, val_loader, model, cfg, model_name):
    result_dict = validate(val_loader, model, cfg, model_name)

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
        f"Test: [{epoch}/{cfg.OPTIM.MAX_EPOCH}]\t"
        f"Macro_AUC: {macro_auc:.3f}\t"
        f"Micro_AUC: {micro_auc:.3f}\t"
        f"mAP: {map_score:.3f}\t"
        f"MCC: {mcc_score:.3f}\t"
        f"Accuracy: {acc_score:.3f}\t"
        f"Macro_Precision: {macro_pcs:.3f}\t"
        f"Micro_Precision: {micro_pcs:.3f}\t"
        f"Macro_Recall: {macro_rc:.3f}\t"
        f"Micro_Recall: {micro_rc:.3f}\t"
        f"Macro_F1: {marco_f1:.3f}\t"
        f"Micro_F1: {micro_f1:.3f}\t"
    )
    
    PER_CLASS_THRESHOLD = False
    DYNAMIC_THRESHOLD = False
    if DYNAMIC_THRESHOLD:
        print("============Using dynamic thresholds strategy=================")
        val_info += f"\tUsing dynamic thresholds strategy"
    elif PER_CLASS_THRESHOLD:
        print("============Using per-class F1 optimal thresholds=================")
        val_info += f"\tUsing per-class F1 optimal thresholds"
    else:
        print("============Using fixed threshold=================")
        val_info += f"\twith threshold: {cfg.TEST.THRESHOLD:.3f}"
        
    print(val_info)
    write_logfile(val_info, logger)
    return result_dict["macro_auc"]