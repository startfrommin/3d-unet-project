import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import nibabel as nib
import numpy as np
import torch
from unet3d_model import UNet3D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet3D(n_channels=4, n_classes=3).to(device)
model.load_state_dict(torch.load("unet3d_best.pth"))
model.eval()

img_path = r"D:\python 学习 project\3DUNet\my_datasets\images\la_003.nii.gz"
save_path = r"pred_la_003.nii.gz"

img_nib = nib.load(img_path)
img = img_nib.get_fdata()
img = np.transpose(img, axes=(3,0,1,2))
img_tensor = torch.from_numpy(img).float().unsqueeze(0).to(device)

with torch.no_grad():
    pred_logit = model(img_tensor)
    pred = torch.argmax(pred_logit, dim=1)

pred_np = pred.squeeze(0).cpu().numpy()
pred_np = pred_np.astype(np.int16)  # 转int16，放在这里！
out_nii = nib.Nifti1Image(pred_np, img_nib.affine)
nib.save(out_nii, save_path)
print(f"预测完成，保存至：{save_path}")