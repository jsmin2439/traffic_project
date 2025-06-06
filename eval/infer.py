import torch
import numpy as np

def predict(model, x_batch, model_type, device, weekend_flag=None):
    """
    x_batch : torch.Tensor(B,9,12,1370)
    반환     : np.ndarray(B,1370,8)  (정규화 공간)
    """
    B = x_batch.size(0)
    # 1) 입력 배치 전체를 한 번에 device(GPU/CPU)로 옮겨 두면 .to(device)를 반복 호출하지 않아도 됩니다.
    x_batch = x_batch.to(device)
    if model_type == 'lstm':
        # (B,9,12,1370) 중 채널 0~7(queue+speed)만 골라서 (B,8,12,1370) 만들고
        # permute → (B,12,1370,8) 형태로 바꿔 줍니다.
        x_in = x_batch[:, :8].permute(0,2,3,1).contiguous()  # (B,12,1370,8)
        # 이미 x_batch를 device에 올려두었으므로 model(x_in)만 호출하면 됩니다.
        out  = model(x_in).permute(0,2,1)                    # (B,8,1370)

    elif model_type == 'stgcn':
        # 마찬가지로 x_batch는 이미 device에 올라가 있으므로 .to(device) 호출 불필요
        out = model(x_batch)                                 # (B,8,1370)

    else:  # resstgcn
        # ST-GCN 예측
        y_st = model.stgcn(x_batch)                          # (B,8,1370)
        res_seq = torch.zeros((B,12,1370,8), device=device)
        # 여기서 x_batch[:, :8, -1]은 “마지막 시간 스텝(=index -1)에서 채널 0~7”을 꺼내
        # (B,8,1370) → permute(0,2,1) → (B,1370,8) 형태로 두어 res_seq[:,11]에 할당합니다.
        res_seq[:, 11] = x_batch[:, :8, -1].permute(0,2,1)   # (B,1370,8) 형태

        # 꼭 weekend_flag가 전달되어야 합니다. None이면 에러를 띄우도록 처리
        if weekend_flag is None:
            raise ValueError("model_type='resstgcn'인 경우 반드시 weekend_flag를 함께 넘겨주세요.")
        corr = model.reslstm(res_seq, weekend_flag.to(device))
        out  = y_st + corr                                  # (B,8,1370)
    # gradient 추적 끊고 CPU로 꺼낸 뒤 NumPy 배열로 변환합니다.
    return out.detach().cpu().numpy().transpose(0,2,1)              # → (B,1370,8)