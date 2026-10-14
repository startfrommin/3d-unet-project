import nibabel as nib
import numpy as np

def dice_coeff(pred, target):
    pred = pred.flatten()
    target = target.flatten()
    intersection = np.sum(pred * target)
    union = np.sum(pred) + np.sum(target)
    if union == 0:
        return 1.0
    return 2 * intersection / union

# ========== 这里是正确路径 ==========
pred_path = r"D:\python 学习 project\3DUNet\pred_la_003.nii.gz"
label_path = r"D:\python 学习 project\3DUNet\my_datasets\masks\la_003.nii.gz"

pred_nib = nib.load(pred_path)
pred = pred_nib.get_fdata()
label_nib = nib.load(label_path)
label = label_nib.get_fdata()

dice = dice_coeff(pred, label)
print(f"Dice系数: {dice:.4f}")