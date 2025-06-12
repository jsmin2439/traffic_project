import torch
import numpy as np
from torch import amp

def predict(model, x_batch, model_type, device, weekend_flag=None):
    """
    x_batch : torch.Tensor(B,9,12,1370)
    weekend_flag : torch.Tensor(B,) or None
    반환     : np.ndarray(B,1370,8)  (정규화 공간)
    """
    # 1) 미리 device로 올려두기
    x_batch = x_batch.to(device)
    if weekend_flag is not None:
        weekend_flag = weekend_flag.to(device)

    # 2) AMP 자동 캐스트
    with amp.autocast(device_type="cuda"):
        if model_type == 'lstm':
            # model(x_batch) -> (B,8,N)
            out = model(x_batch)
            # (B,8,N) -> (B,N,8)
            out = out.permute(0, 2, 1)

        elif model_type == 'stgcn':
            # model(x_batch) -> (B,8,N)
            out = model(x_batch)
            # (B,8,N) -> (B,N,8)
            out = out.permute(0, 2, 1)

        elif model_type == 'resstgcn':
            # model.inference -> (B,8,N)
            out = model.inference(x_batch, weekend_flag)
            # (B,8,N) -> (B,N,8)
            out = out.permute(0, 2, 1)

        elif model_type == 'gated':
            # model(x_batch, weekend_flag) -> (B,8,N)
            out = model(x_batch, weekend_flag)
            # (B,8,N) -> (B,N,8)
            out = out.permute(0, 2, 1)

        elif model_type == 'pooled':
            horizon = getattr(model, 'horizon', 1)
            # residual sequence: first `horizon` steps of traffic channels
            res_seq = x_batch[:, :8, :, :].permute(0, 2, 1, 3)[:, :horizon]
            out = model(x_batch, residual_seq=res_seq)
            out = out[:, :, 0, :]  # first step
            out = out.permute(0, 2, 1)

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    # 3) numpy 변환 & shape 통일: always (B,1370,8)
    out_np = out.detach().cpu().numpy()
    # 만약 채널축이 두 번째(8)라면 transpose
    if out_np.shape[1] == 8:
        out_np = out_np.transpose(0, 2, 1)
    return out_np