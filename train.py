# train.py
import os, time, json
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from model import STGCN, ResidualLSTM
from utils import TrafficWindowDataset, EarlyStopping

# 1) 데이터 로드 -------------------------------------------------------
X_all = np.load('3_tensor/windows/all_X.npy')
Y_all = np.load('3_tensor/windows/all_Y.npy')
N = X_all.shape[0]
train_end = int(N*0.7); val_end = int(N*0.85)

ds_train = TrafficWindowDataset(X_all[:train_end],  Y_all[:train_end])
ds_val   = TrafficWindowDataset(X_all[train_end:val_end], Y_all[train_end:val_end])

train_loader = DataLoader(ds_train, batch_size=4, shuffle=True, drop_last=True)
val_loader   = DataLoader(ds_val,   batch_size=4, shuffle=False)

# 2) 모델 --------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
A = torch.from_numpy(np.load('adjacency/A_lane.npy')).float().to(device)
num_nodes = A.shape[0]

net_st = STGCN(in_channels=9, out_channels=8, num_nodes=num_nodes, A=A).to(device)
net_rs = ResidualLSTM(num_nodes=num_nodes, hidden_size=512).to(device)

crit  = nn.MSELoss()
opt_st = optim.Adam(net_st.parameters(), lr=1e-3, weight_decay=1e-5)
opt_rs = optim.Adam(net_rs.parameters(), lr=1e-3, weight_decay=1e-5)

scheduler_st = optim.lr_scheduler.ReduceLROnPlateau(opt_st, 'min', factor=0.5, patience=3)
scheduler_rs = optim.lr_scheduler.ReduceLROnPlateau(opt_rs, 'min', factor=0.5, patience=3)

early = EarlyStopping(patience=7)

# 3) 학습 --------------------------------------------------------------
EPOCHS = 40
best_val = np.inf
os.makedirs('checkpoints', exist_ok=True)
log = []

for epoch in range(1, EPOCHS+1):
    net_st.train(); net_rs.train()
    tr_loss, n = 0., 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)       # xb: (B,9,12,N)  yb:(B,8,N)

        # --- ST-GCN forward ---
        y_pred_st = net_st(xb)                      # (B,8,N)

        # --- Residual LSTM forward ---
        res = yb - y_pred_st                       # (B,8,N)
        B = xb.size(0)
        res_flat = res.view(B, -1)                 # (B, 8*N)
        flag = xb[:, 8, 0, 0]                      # (B,)
        res_corr = net_rs(res_flat, flag)          # (B,8,N)

        y_final = y_pred_st + res_corr             # (B,8,N)

        loss = crit(y_final, yb)
        opt_st.zero_grad(); opt_rs.zero_grad()
        loss.backward()
        opt_st.step(); opt_rs.step()

        tr_loss += loss.item(); n += 1
    tr_loss /= n

    # --- Validation ---
    net_st.eval(); net_rs.eval()
    vl_loss, n = 0., 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            y_pred_st = net_st(xb)
            res = yb - y_pred_st
            B = xb.size(0); res_flat = res.view(B, -1)
            flag = xb[:, 8, 0, 0]
            res_corr = net_rs(res_flat, flag)
            y_final = y_pred_st + res_corr
            loss = crit(y_final, yb)
            vl_loss += loss.item(); n += 1
    vl_loss /= n

    scheduler_st.step(vl_loss); scheduler_rs.step(vl_loss)
    early.step(vl_loss)

    log.append({'epoch': epoch, 'train': tr_loss, 'val': vl_loss})
    print(f"[{epoch:02d}] Train {tr_loss:.6f}  |  Val {vl_loss:.6f}")

    if vl_loss < best_val:
        best_val = vl_loss
        torch.save({
            'stgcn': net_st.state_dict(),
            'reslstm': net_rs.state_dict(),
            'val_loss': best_val,
            'epoch': epoch
        }, 'checkpoints/best_model.pt')
        print("  ↳ Best model saved!")

    if early.stop:
        print("Early stopping triggered"); break

# 로그 저장
json.dump(log, open('training_log.json', 'w'), indent=2)