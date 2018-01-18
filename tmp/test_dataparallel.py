from __future__ import print_function
import torch
from api import config_fun
import train_helper


cfg = config_fun.config()
model = train_helper.get_model(cfg, pretrained=False)

model_pa = torch.nn.DataParallel(model)

model_cuda = model.cuda()

print(type(model))

print(type(model_cuda))

print(type(model_pa))

print(type(model_pa.module))

print(isinstance(model, torch.nn.DataParallel))

print(isinstance(model, torch.nn.Module))

print(isinstance(model_cuda, torch.nn.DataParallel))

print(isinstance(model_cuda, torch.nn.Module))

print(isinstance(model_pa, torch.nn.DataParallel))

print(isinstance(model_pa.module, torch.nn.DataParallel))

print(isinstance(model_pa.module, torch.nn.Module))
