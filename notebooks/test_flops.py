# %%
from torchvision import models
from thop import profile
import torch

model = models.resnet50()
input_tensor = torch.randn(1, 3, 224, 224)
flops, params = profile(model, inputs=(input_tensor,))
print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
print(f"Params: {params / 1e6:.2f} M")