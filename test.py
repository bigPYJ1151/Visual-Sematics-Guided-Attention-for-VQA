import torch

s = torch.from_numpy(None)
s = "xxx_00001.jpg"
ss = "00001.jpg"
print(s.split('_')[-1])
print(ss.split('_')[-1])