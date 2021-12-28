import torch
from models.mobilenetv2 import MobileNetV2_1, MobileNetV2_2
from models.mobilenetv2_wider import wider_MobileNetV2
from models.mobilenetv2_deeper import deeper_MobileNetV2_b

#net = MobileNetV2()
#wider_net = wider_MobileNetV2()
#deeper_net = deeper_MobileNetV2()

net = MobileNetV2_1()
net2 = MobileNetV2_2()
deeper_net = deeper_MobileNetV2_b()

# function preservation 구현하기
# net_transform_wider , net_transform_deeper function 을 구현해서 아래 test 를 통과해보세요
# 현재 셀의 function call 은 예시일 뿐, 자유롭게 구현하셔도 됩니다. 아래 셀의 test 만 통과하면 됩니다.

#new_weights = net_transform_wider(net, wider_net)
#wider_net.load_state_dict(new_weights)

#new_weights = net_transform_deeper(net, deeper_net)
#deeper_net.load_state_dict(new_weights)

# test
x = torch.rand(1,3,32,32)
result = net2(net(x))
#result_wider = wider_net(x)
result2 = net2(net(x))
result_deeper = net2(deeper_net(net(x)))

#result_wider = wider_net(x)
#assert torch.allclose(result,result_wider)
assert torch.allclose(result, result2)
print("Assert 1 True")
assert torch.allclose(result,result_deeper)

