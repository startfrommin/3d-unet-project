import os
import glob
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from unet3d_model import UNet3D, DiceLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据集加载
class BraTSDataset(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        print(f"【调试】数据集根目录：{data_root}")
        img_pattern = os.path.join(data_root,"images","la_*.nii.gz")
        print(f"【调试】匹配路径：{img_pattern}")
        self.img_paths = sorted(glob.glob(img_pattern))
        print(f"【调试】找到图像数量：{len(self.img_paths)}")
        print("【调试】文件列表：", self.img_paths)

        self.mask_paths = []
        for p in self.img_paths:
            name = os.path.basename(p)
            mask_p = os.path.join(data_root,"masks", name)
            self.mask_paths.append(mask_p)
        print(f"【调试】mask列表：{self.mask_paths}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_nib = nib.load(self.img_paths[idx])
        mask_nib = nib.load(self.mask_paths[idx])
        img = img_nib.get_fdata()
        mask = mask_nib.get_fdata()

        img = np.transpose(img, (3,0,1,2))
        mask = np.expand_dims(mask, axis=0)

        return torch.from_numpy(img).float(), torch.from_numpy(mask).long()
# --------------------------
# 配置参数
root = r"D:\python 学习 project\3DUNet\my_datasets"
epoch_num = 30
batch_size = 1
# --------------------------

ds = BraTSDataset(root)
loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

model = UNet3D(n_channels=4, n_classes=3).to(device)
loss_fn = DiceLoss(sigmoid=True)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# 训练循环
for epoch in range(epoch_num):
    model.train()
    total_loss = 0
    for img, label in loader:
        img, label = img.to(device), label.to(device)
        opt.zero_grad()
        pred = model(img)

        # label one-hot编码
        label_onehot = nn.functional.one_hot(label.squeeze(1), num_classes=3).permute(0,4,1,2,3).float()
        loss = loss_fn(pred, label_onehot)

        loss.backward()
        opt.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch [{epoch+1}/{epoch_num}], Loss: {avg_loss:.4f}")

# 保存模型权重
torch.save(model.state_dict(), "unet3d_best.pth")
print("✅训练完成，权重保存为 unet3d_best.pth")