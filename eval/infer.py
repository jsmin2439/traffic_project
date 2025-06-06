import torch, time
import numpy as np

def predict(model, x_batch, model_type, device, weekend_flag=None):
    """
    x_batch : torch.Tensor(B,9,12,1370)
    반환     : np.ndarray(B,1370,8)  (정규화 공간)
    """
    B = x_batch.size(0)
    if model_type == 'lstm':
        x_in = x_batch[:,:8].permute(0,2,3,1).contiguous()  # (B,12,1370,8)
        out  = model(x_in.to(device)).permute(0,2,1)        # (B,8,1370)
    elif model_type == 'stgcn':
        out = model(x_batch.to(device))                     # (B,8,1370)
    else:  # resstgcn
        y_st = model.stgcn(x_batch.to(device))              # (B,8,1370)
        res_seq = torch.zeros((B,12,1370,8), device=device)
        res_seq[:,11] = x_batch[:, :8, -1].permute(0,2,1)   # 잔차 dummy
        corr = model.reslstm(res_seq, weekend_flag.to(device))
        out  = y_st + corr                                  # (B,8,1370)
    return out.detach().cpu().numpy().transpose(0,2,1)              # → (B,1370,8)