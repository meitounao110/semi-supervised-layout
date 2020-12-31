import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy.io as sio
import torch.nn.functional as F
from pathlib import Path
from torch.nn.functional import interpolate
from architectures.fpn import Fpn_n

data = sio.loadmat(
    "/mnt/layout_data/v0.3/data/one_point/test/0/test/Example10002.mat"
)
u_true = data["u"]  # 真实温度
F = data["f"]  # 布局

fig1 = plt.figure(figsize=(10, 5))
plt.subplot(121)
im = plt.imshow(F)
plt.colorbar(im)

plt.subplot(122)
im = plt.imshow(u_true)
plt.colorbar(im)
print(u_true.max())
print(u_true.min())
plt.show()

PATH = '/mnt/zhangyunyang1/pseudo_label-pytorch-master/experiments/model/model.pth'
model = Fpn_n()
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['weight'])
model.eval()

F_tensor = (torch.from_numpy(F.astype(float)).float().unsqueeze(0).unsqueeze(0)) / 20000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
F_tensor = F_tensor.to(device)
model = model.to(device)
with torch.no_grad():
    u_pred = model(F_tensor)
u_pred = interpolate(u_pred, size=(200, 200),
                     mode='bilinear',
                     align_corners=True) * 50 + 298
u_pred_n = u_pred.cpu().squeeze().numpy()

fig2 = plt.figure(figsize=(10, 5))
plt.subplot(121)
im = plt.imshow(F)
plt.colorbar(im)

plt.subplot(122)
im = plt.imshow(u_pred_n)
plt.colorbar(im)
# fig.savefig('./predict.png', dpi=100)
plt.show()
# %%
print(u_pred_n.max())
print(u_pred_n.min())
a = torch.abs(torch.Tensor(u_pred_n - u_true))
mae = torch.mean(a)
print('mae', mae)
