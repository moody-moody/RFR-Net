import torch
from thop import profile
from thop import clever_format
from Image_desnowing.models.RFR import build_net
from models import RFR

# 1. 构建模型
model = build_net('large') # 改成你的模型
model.eval()

# 2. 构造输入（非常重要！）
# ICCV 通常用 256x256 或 512x512
input = torch.randn(1, 3, 256, 256)

# 3. 统计
flops, params = profile(model, inputs=(input,), verbose=False)

# 4. 格式化
flops, params = clever_format([flops, params], "%.3f")

print(f"Params: {params}")
print(f"FLOPs: {flops}")
