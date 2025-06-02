# eval.py
import numpy as np, matplotlib.pyplot as plt, torch, json, os
from torch.utils.data import DataLoader
from model import STGCN, ResidualLSTM
from utils import TrafficWindowDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
A = torch.from_numpy(np.load('adjacency/A_lane.npy')).float().to(device)
num_nodes = A.shape[0]

# --- Dataset ---
X_all = np.load('3_tensor/windows/all_X.npy')
Y_all = np.load('3_tensor/windows/all_Y.npy')
N = X_all.shape[0]; val_end = int(N*0.85)
ds_test = TrafficWindowDataset(X_all[val_end:], Y_all[val_end:])
test_loader = DataLoader(ds_test, batch_size=4, shuffle=False)

# --- 모델 로드 ---
net_st = STGCN(9, 8, num_nodes, A).to(device)
net_rs = ResidualLSTM(num_nodes, 512).to(device)
ckpt = torch.load('checkpoints/best_model.pt')
net_st.load_state_dict(ckpt['stgcn']); net_rs.load_state_dict(ckpt['reslstm'])
net_st.eval(); net_rs.eval()

crit = torch.nn.MSELoss()
loss, n = 0., 0
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        y_st = net_st(xb)
        res = yb - y_st
        B = xb.size(0); res_flat = res.view(B, -1)
        flag = xb[:, 8, 0, 0]
        res_corr = net_rs(res_flat, flag)
        y_final = y_st + res_corr
        loss += crit(y_final, yb).item(); n += 1
test_loss = loss / n
print(f"Test MSE: {test_loss:.6f}")

# --- 예측 vs 실제 시각화 (임의 노드) ---
xb, yb = next(iter(test_loader))
xb = xb.to(device); yb = yb.to(device)
y_pred = net_st(xb) + net_rs((yb - net_st(xb)).view(xb.size(0), -1),
                             xb[:, 8, 0, 0])

node_idx = 10  # 임의 차로
plt.figure(figsize=(6,4))
plt.plot(yb[0,:,node_idx].cpu().numpy(), label='True')
plt.plot(y_pred[0,:,node_idx].cpu().numpy(), '--', label='Pred')
plt.title(f'Node {node_idx} - Queue+Speed (8ch)')
plt.legend(); plt.tight_layout()
os.makedirs('plots', exist_ok=True)
plt.savefig('plots/node10_pred_vs_true.png')
plt.show()