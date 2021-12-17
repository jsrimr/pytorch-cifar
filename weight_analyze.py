import torch

from models import MobileNetV2
from utils import BASE_CFG, get_next_net

PLAN = [
    ("width", 2),
    ("depth", 2),
    ("width", 3),
    ("depth", 3),
    ("width", 4),
    ("depth", 4),
    ("width", 5),
    ("depth", 5),
]
device = 'cuda' if torch.cuda.is_available() else 'cpu'

current_cfg = BASE_CFG
plan_idx = 0
target_stage = 2
net = MobileNetV2(cfg=current_cfg).to(device)

net, current_cfg = get_next_net(net, current_cfg, PLAN, plan_idx)
checkpoint = torch.load('./checkpoint/width_2_ckpt.pth')
net.load_state_dict(checkpoint['net'])

print("loaded")

for i in range(net.layers[target_stage].stage_2_block0.conv3.weight.shape[0]):
    if i == 16:
        print("==============================")
    print(torch.linalg.norm(net.layers[target_stage].stage_2_block0.conv3.weight[i]))
