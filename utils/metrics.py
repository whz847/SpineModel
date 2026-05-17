import numpy as np
from sklearn.metrics import (
    auc,
    roc_curve,
    precision_recall_curve,
    matthews_corrcoef,
)


def multilabelConfussionMatrix(y_test, predictions):
    """
    Returns the TP, FP, TN, FN
    """

    TP = np.zeros(y_test.shape[1])
    FP = np.zeros(y_test.shape[1])
    TN = np.zeros(y_test.shape[1])
    FN = np.zeros(y_test.shape[1])

    for j in range(y_test.shape[1]):
        TPaux = 0
        FPaux = 0
        TNaux = 0
        FNaux = 0
        for i in range(y_test.shape[0]):
            if int(y_test[i, j]) == 1:
                if int(y_test[i, j]) == 1 and int(predictions[i, j]) == 1:
                    TPaux += 1
                else:
                    FPaux += 1
            else:
                if int(y_test[i, j]) == 0 and int(predictions[i, j]) == 0:
                    TNaux += 1
                else:
                    FNaux += 1
        TP[j] = TPaux
        FP[j] = FPaux
        TN[j] = TNaux
        FN[j] = FNaux

    return TP, FP, TN, FN


def multilabelMicroConfussionMatrix(TP, FP, TN, FN):
    """
    Returns Micro TP, FP, TN, FN
    """

    TPMicro = 0.0
    FPMicro = 0.0
    TNMicro = 0.0
    FNMicro = 0.0

    for i in range(len(TP)):
        TPMicro = TPMicro + TP[i]
        FPMicro = FPMicro + FP[i]
        TNMicro = TNMicro + TN[i]
        FNMicro = FNMicro + FN[i]

    return TPMicro, FPMicro, TNMicro, FNMicro


def accuracyMacro(y_test, predictions):

    accuracymacro = 0.0
    accurancy_list = []
    TP, FP, TN, FN = multilabelConfussionMatrix(y_test, predictions)

    for i in range(len(TP)):
        acc_class = (TP[i] + TN[i]) / (TP[i] + FP[i] + TN[i] + FN[i])
        accuracymacro += acc_class
        accurancy_list.append(acc_class)

    accuracymacro = np.mean(accurancy_list)

    return accuracymacro, accurancy_list


def precisionMacro(y_test, predictions):

    precisionmacro = 0.0
    precision_list = []
    TP, FP, TN, FN = multilabelConfussionMatrix(y_test, predictions)

    for i in range(len(TP)):
        if not np.all(y_test[:, i] == 0):
            pcs_class = 0.0
            if TP[i] + FP[i] != 0:
                pcs_class = TP[i] / (TP[i] + FP[i])
                precisionmacro = precisionmacro + pcs_class
            precision_list.append(pcs_class)

    precisionmacro = np.mean(precision_list)

    return precisionmacro, precision_list


def precisionMicro(y_test, predictions):

    precisionmicro = 0.0
    TP, FP, TN, FN = multilabelConfussionMatrix(y_test, predictions)
    TPMicro, FPMicro, TNMicro, FNMicro = multilabelMicroConfussionMatrix(TP, FP, TN, FN)

    if (TPMicro + FPMicro) != 0:
        precisionmicro = float(TPMicro / (TPMicro + FPMicro))

    return precisionmicro


def recallMacro(y_test, predictions):

    recallmacro = 0.0
    recall_list = []
    TP, FP, TN, FN = multilabelConfussionMatrix(y_test, predictions)

    for i in range(len(TP)):
        if not np.all(y_test[:, i] == 0):
            rc_class = 0.0
            if TP[i] + FN[i] != 0:
                rc_class = TP[i] / (TP[i] + FN[i])
                recallmacro = recallmacro + rc_class
            recall_list.append(rc_class)

    recallmacro = np.mean(recall_list)

    return recallmacro, recall_list


def recallMicro(y_test, predictions):
    recallmicro = 0.0
    TP, FP, TN, FN = multilabelConfussionMatrix(y_test, predictions)
    TPMicro, FPMicro, TNMicro, FNMicro = multilabelMicroConfussionMatrix(TP, FP, TN, FN)

    if (TPMicro + FNMicro) != 0:
        recallmicro = float(TPMicro / (TPMicro + FNMicro))

    return recallmicro


def fbetaMacro(y_test, predictions, beta=1):
    fbetamacro = 0.0
    fbeta_list = []
    TP, FP, TN, FN = multilabelConfussionMatrix(y_test, predictions)
    for i in range(len(TP)):
        num = float((1 + pow(beta, 2)) * TP[i])
        den = float((1 + pow(beta, 2)) * TP[i] + pow(beta, 2) * FN[i] + FP[i])
        fbeta_class = 0.0
        if den != 0:
            fbeta_class = num / den
            fbetamacro = fbetamacro + fbeta_class
        fbeta_list.append(fbeta_class)

    fbetamacro = fbetamacro / len(TP)
    return fbetamacro, fbeta_list


def fbetaMicro(y_test, predictions, beta=1):
    fbetamicro = 0.0
    TP, FP, TN, FN = multilabelConfussionMatrix(y_test, predictions)
    TPMicro, FPMicro, TNMicro, FNMicro = multilabelMicroConfussionMatrix(TP, FP, TN, FN)

    num = float((1 + pow(beta, 2)) * TPMicro)
    den = float((1 + pow(beta, 2)) * TPMicro + pow(beta, 2) * FNMicro + FPMicro)
    fbetamicro = float(num / den)

    return fbetamicro


def Average_MCC(ground_truth, predictions, class_num):
    mcc_scores = [
        matthews_corrcoef(ground_truth[:, i], predictions[:, i])
        for i in range(class_num)
    ]
    average_mcc = np.mean(mcc_scores)
    return average_mcc


def calc_roc(ground_truth, predictions, class_num):
    # Micro.
    fpr, tpr, _ = roc_curve(ground_truth.ravel(), predictions.ravel())
    micro_roc_auc = auc(fpr, tpr)
    # Macro.
    roc_per_class = []
    macro_roc_auc = 0

    for i in range(class_num):
        if not np.all(ground_truth[:, i] == 0):
            fpr, tpr, _ = roc_curve(ground_truth[:, i], predictions[:, i])
            new_auc = auc(fpr, tpr)
            roc_per_class.append(new_auc)

    macro_roc_auc = np.mean(roc_per_class)

    return macro_roc_auc, micro_roc_auc, roc_per_class


def calc_ap(ground_truth, predictions):
    precision, recall, thresholds = precision_recall_curve(ground_truth, predictions)
    return auc(recall, precision)


def calc_map(ground_truth, predictions, class_num):
    ap_list = []
    for i in range(class_num):
        gt_class = ground_truth[:, i]
        pred_class = predictions[:, i]
        if not np.all(gt_class == 0):
            ap = calc_ap(gt_class, pred_class)
            ap_list.append(ap)

    map_score = np.mean(ap_list)

    return map_score, ap_list


def compute_f1_thresholds(y_true, y_prob, step=0.001):
    """
    计算每个类别的最优F1阈值
    
    参数:
    - y_true: 真实标签，形状为 (N, C)，其中 N 是样本数量，C 是类别数量
    - y_prob: 预测概率，形状为 (N, C)
    - step: 阈值搜索步长，默认为 0.001
    
    返回:
    - thresholds: 每个类别的最优阈值，形状为 (C,)
    """
    n_classes = y_true.shape[1]
    thresholds = np.zeros(n_classes)
    
    for c in range(n_classes):
        # 检查该类别是否有正样本
        if not np.any(y_true[:, c] == 1):
            # 如果没有正样本，设置默认阈值为0.5
            thresholds[c] = 0.5
            continue
        
        best_f1 = -1
        best_threshold = 0.5
        
        # 遍历可能的阈值
        for threshold in np.arange(0.0, 1.0 + step, step):
            # 根据阈值生成二进制预测
            y_pred_binary = (y_prob[:, c] >= threshold).astype(int)
            
            # 计算TP, FP, FN
            tp = np.sum((y_true[:, c] == 1) & (y_pred_binary == 1))
            fp = np.sum((y_true[:, c] == 0) & (y_pred_binary == 1))
            fn = np.sum((y_true[:, c] == 1) & (y_pred_binary == 0))
            
            # 计算F1分数
            if tp + fp == 0 or tp + fn == 0:
                f1 = 0.0
            else:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                if precision + recall == 0:
                    f1 = 0.0
                else:
                    f1 = 2 * (precision * recall) / (precision + recall)
            
            # 更新最优阈值
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                
        thresholds[c] = best_threshold
    
    return thresholds

def compute_dynamic_thresholds(y_true, y_prob, step=0.01):
    """
    实现动态阈值策略：
    1. 先计算全局阈值：在所有样本和所有类别上展平后，通过网格搜索找到使加权得分=0.7*micro-F1+0.3*MCC最大的阈值
    2. 对每个类别c：
       - 如果正样本数 < 5：直接使用全局阈值
       - 如果正样本数 >= 20：单独为该类别用网格搜索找使F1最大的阈值
       - 如果正样本数在5到19之间：将全局阈值和类别专属阈值按6:4加权融合
    
    参数:
    - y_true: 真实标签，形状为 (N, C)，其中 N 是样本数量，C 是类别数量
    - y_prob: 预测概率，形状为 (N, C)
    - step: 阈值搜索步长，默认为 0.01（由于是网格搜索，较小的步长会增加计算时间）
    
    返回:
    - thresholds: 每个类别的动态阈值，形状为 (C,)
    """
    n_samples, n_classes = y_true.shape
    
    # 计算全局阈值
    # 展平数据用于全局阈值计算
    flat_y_true = y_true.ravel()
    flat_y_prob = y_prob.ravel()
    
    best_global_score = -1
    best_global_threshold = 0.5
    
    for threshold in np.arange(0.05, 0.96, step):  # 从0.05到0.95
        y_pred_binary = (flat_y_prob >= threshold).astype(int)
        
        # 计算micro-F1
        tp_micro = np.sum((flat_y_true == 1) & (y_pred_binary == 1))
        fp_micro = np.sum((flat_y_true == 0) & (y_pred_binary == 1))
        fn_micro = np.sum((flat_y_true == 1) & (y_pred_binary == 0))
        
        if tp_micro + fp_micro == 0 or tp_micro + fn_micro == 0:
            micro_f1 = 0.0
        else:
            precision_micro = tp_micro / (tp_micro + fp_micro)
            recall_micro = tp_micro / (tp_micro + fn_micro)
            if precision_micro + recall_micro == 0:
                micro_f1 = 0.0
            else:
                micro_f1 = 2 * (precision_micro * recall_micro) / (precision_micro + recall_micro)
        
        # 计算MCC
        tn_micro = np.sum((flat_y_true == 0) & (y_pred_binary == 0))
        mcc = matthews_corrcoef(flat_y_true, y_pred_binary)
        
        # 计算加权得分
        score = 0.7 * micro_f1 + 0.3 * mcc
        
        if score > best_global_score:
            best_global_score = score
            best_global_threshold = threshold
    
    # 初始化结果数组
    thresholds = np.zeros(n_classes)
    
    for c in range(n_classes):
        # 计算该类别的正样本数
        pos_count = np.sum(y_true[:, c] == 1)
        
        if pos_count < 5:
            # 小样本，使用全局阈值
            thresholds[c] = best_global_threshold
        elif pos_count >= 20:
            # 大样本，单独计算最优F1阈值
            best_f1 = -1
            best_threshold = 0.5
            
            for threshold in np.arange(0.0, 1.0 + step, step):
                y_pred_binary = (y_prob[:, c] >= threshold).astype(int)
                
                tp = np.sum((y_true[:, c] == 1) & (y_pred_binary == 1))
                fp = np.sum((y_true[:, c] == 0) & (y_pred_binary == 1))
                fn = np.sum((y_true[:, c] == 1) & (y_pred_binary == 0))
                
                if tp + fp == 0 or tp + fn == 0:
                    f1 = 0.0
                else:
                    precision = tp / (tp + fp)
                    recall = tp / (tp + fn)
                    if precision + recall == 0:
                        f1 = 0.0
                    else:
                        f1 = 2 * (precision * recall) / (precision + recall)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
                    
            thresholds[c] = best_threshold
        else:
            # 中等样本，融合全局阈值和类别阈值
            # 先计算类别专属阈值
            best_f1 = -1
            best_threshold = 0.5
            
            for threshold in np.arange(0.0, 1.0 + step, step):
                y_pred_binary = (y_prob[:, c] >= threshold).astype(int)
                
                tp = np.sum((y_true[:, c] == 1) & (y_pred_binary == 1))
                fp = np.sum((y_true[:, c] == 0) & (y_pred_binary == 1))
                fn = np.sum((y_true[:, c] == 1) & (y_pred_binary == 0))
                
                if tp + fp == 0 or tp + fn == 0:
                    f1 = 0.0
                else:
                    precision = tp / (tp + fp)
                    recall = tp / (tp + fn)
                    if precision + recall == 0:
                        f1 = 0.0
                    else:
                        f1 = 2 * (precision * recall) / (precision + recall)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            
            # 融合阈值：0.6 * global_t + 0.4 * class_t
            thresholds[c] = 0.6 * best_global_threshold + 0.4 * best_threshold
    
    return thresholds
