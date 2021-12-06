network transformation vs random-init training


network transformation 구현
모델 : MBv2

1. Basenet 준비
2. wider, depper net 을 func_preserve 하면서 10epoch 씩 train
3. 최종 net 의 accuracy get
4. (3) 의 acc 를 random training 때의 acc와 비교


# skill
1. 특정 block 만 learning_rate 높이고 다른 데는 원래 schedule 대로
2. deeper 에서 bn3 뿐 아니라 다른 block 들도 조작


for p_w, c_w in zip(net.state_dict(), child_net.state_dict()):
    if net.state_dict()[p_w].shape != child_net.state_dict()[c_w].shape:
        print(p_w, c_w, net.state_dict()[p_w].shape, child_net.state_dict()[c_w].shape)


# additional work
neptune 에 기록


# todo 
[] noise

plan 에 따라 net 키우기
10epoch 씩 훈련


의문
[] 기존에는 np.newaxis 안썼는데 왜 잘 됐을까..
[] 왜 tf 버전에서는 shortcut 안건드렸는데 됐지?
