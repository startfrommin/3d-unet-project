import os
import numpy as np
import nibabel as nib

root = r"D:\python 学习 project\3DUNet\my_datasets"
img_dir = os.path.join(root, "images")
mask_dir = os.path.join(root, "masks")
os.makedirs(img_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)

sample_ids = [3,4,5,7,9,10]
shape = (128,128,128,4)

for idx in sample_ids:
    name = f"la_{idx:03d}"
    img_data = np.random.randint(0,2000, size=shape, dtype=np.int16)
    img_nii = nib.Nifti1Image(img_data, affine=np.eye(4))
    nib.save(img_nii, os.path.join(img_dir,f"{name}.nii.gz"))

    mask_data = np.zeros(shape[:3], dtype=np.uint8)
    mask_data[30:90,30:90,30:90] = np.random.choice([1,2,4])
    mask_nii = nib.Nifti1Image(mask_data, affine=np.eye(4))
    nib.save(mask_nii, os.path.join(mask_dir,f"{name}.nii.gz"))
print("✅模拟数据集生成完毕")