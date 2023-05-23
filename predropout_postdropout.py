import torch
from torch import nn
import math
import numpy as np

hp_bs = int(1e5)  # reasonable

hp_do = 0.5  # 1-keep

hp_wi = 128
# hp_wi = 256
# hp_wi = 512
# hp_wi = 1024
# hp_wi = 2048

std_X = 1.0
X_0 = torch.normal(mean=torch.zeros((hp_bs, hp_wi)), std=std_X)

B_0 = nn.BatchNorm1d(hp_wi)
A_0 = torch.nn.ReLU()
D_0 = nn.Dropout(p=hp_do)
C_0 = nn.Linear(hp_wi, hp_wi, bias=False)

B_1 = nn.BatchNorm1d(hp_wi)
A_1 = torch.nn.ReLU()
D_1 = nn.Dropout(p=hp_do)
C_1 = nn.Linear(hp_wi, hp_wi, bias=False)

B_0.train()
D_0.train()
B_1.train()
D_1.train()

for w_mean in list(np.arange(-0.02, 0.0202, 0.0002)):
    nn.init.normal_(C_0.weight, mean=w_mean, std=math.sqrt(2 / hp_wi))
    nn.init.normal_(C_1.weight, mean=w_mean, std=math.sqrt(2 / hp_wi))

    # define x as BACBA
    x = A_1(B_1(C_0(A_0(B_0(X_0)))))
    p = 1.0 - hp_do

    Wx = C_1(x)
    DWx = D_1(Wx)
    Dx = D_1(x)
    WDx = C_1(Dx)

    var_Wx = Wx.var(dim=0, unbiased=False).mean().item()
    var_DWx = DWx.var(dim=0, unbiased=False).mean().item()
    var_WDx = WDx.var(dim=0, unbiased=False).mean().item()

    print("Check")
    print(w_mean)
    check_DW = var_Wx / var_DWx
    check_WD = var_Wx / var_WDx
    print(check_DW)
    print(check_WD)

    print("")
