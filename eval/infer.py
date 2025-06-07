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
            x_in = x_batch[:, :8].permute(0,2,3,1).contiguous()  # (B,12,1370,8)
            out  = model(x_in).permute(0,2,1)                    # (B,8,1370)

        elif model_type == 'stgcn':
            out = model(x_batch)                                 # (B,8,1370)

        elif model_type == 'resstgcn':
            out = model.inference(x_batch, weekend_flag)         # (B,8,1370)

        elif model_type == 'gated':
            # Gated-Fusion 모델은 forward(x, weekend_flag) 호출
            out = model(x_batch, weekend_flag)                   # (B,8,1370)

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    # 3) numpy로 변환 (B,8,1370) -> transpose -> (B,1370,8)
    return out.detach().cpu().numpy().transpose(0,2,1)