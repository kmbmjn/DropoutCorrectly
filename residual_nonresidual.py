import torch
from torch import nn
import math
import numpy as np

hp_bs = int(1e5)  # reasonable

# hp_do = 0.5  # 1-keep
# hp_do = 0.4  # 1-keep
# hp_do = 0.3  # 1-keep
# hp_do = 0.2  # 1-keep
hp_do = 0.1  # 1-keep

hp_wi = 128

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

w_mean = 0.0
nn.init.normal_(C_0.weight, mean=w_mean, std=math.sqrt(2 / hp_wi))
nn.init.normal_(C_1.weight, mean=w_mean, std=math.sqrt(2 / hp_wi))

for std_X in list(np.arange(0.2, 5.005, 0.005)):
    print("Check")
    print(std_X)

    X_0 = torch.normal(mean=torch.zeros((hp_bs, hp_wi)), std=std_X)

    # define x as BACB
    x = B_1(C_0(A_0(B_0(X_0))))

    f_P6 = C_1(D_1(A_1(x)))
    var_f_P6 = f_P6.var(dim=0, unbiased=False).mean().item()

    f_nodo = C_1(A_1(x))
    var_f_nodo = f_nodo.var(dim=0, unbiased=False).mean().item()

    delta_f = var_f_nodo / var_f_P6
    print(delta_f)

    # residual
    X_next = X_0 + C_1(D_1(A_1(x)))
    var_X_next = X_next.var(dim=0, unbiased=False).mean().item()

    X_nodo = X_0 + C_1(A_1(x))
    var_X_nodo = X_nodo.var(dim=0, unbiased=False).mean().item()

    delta_X = var_X_nodo / var_X_next
    print(delta_X)
    print("")
