import h5py
import numpy as np
# 打开整合文件
with h5py.File("/mnt/K/WHZ/datasets/Spine/valid/3946091PA172X/LUMBAR_MRI/study.h5", 'r') as f:
    # 列出所有模态
    print("可用模态:", list(f.keys()))  # e.g., ['AX', 'T2']
    ax_slice = f['AX']['slices']
    num_slices = ax_slice.shape[0]
    print(num_slices)
    # 读取 FS 模态的第 5 张切片（即 IM5）
    if 'FS' in f:
        fs_slice_5 = f['FS']['slices'][5]  # shape: (H, W)
        print("FS IM5 shape:", fs_slice_5.shape)
        ax_slice1 = f['AX']['slices'][5]
        ax_slice2 = f['AX']['slices'][6]
        ax_slice3 = f['AX']['slices'][7]
        ax_data = np.stack([ax_slice1, ax_slice2, ax_slice3], axis=0)
        print("ax_data.shape:", ax_data.shape)
        # 验证数据类型
        assert isinstance(ax_data, np.ndarray), "输入数据不是 NumPy 数组"
        assert ax_data.dtype == np.float32, f"数据类型错误，期望 np.float32，实际为 {ax_data.dtype}"

        # 验证形状 (C, H, W)
        assert len(ax_data.shape) == 3, f"数据维度错误，期望三维 (C, H, W)，实际形状为 {data.shape}"
        assert ax_data.shape[0] == 3, f"通道数不匹配，期望 {3}，实际为 {ax_data.shape[0]}"

        # 验证值域 [0, 1]
        min_val = np.min(ax_data)
        max_val = np.max(ax_data)
        assert min_val >= 0.0 and max_val <= 1.0, f"值域超出 [0, 1]，实际范围 [{min_val}, {max_val}]"

        print(f"数据检查通过！形状: {ax_data.shape}, 数据类型: {ax_data.dtype}, 值域: [{min_val}, {max_val}]")

        # 获取 FS 的归一化参数
        mean_fs = f['FS'].attrs['mean']
        std_fs = f['FS'].attrs['std']
        print(f"FS 原始均值: {mean_fs:.2f}, 标准差: {std_fs:.2f}")