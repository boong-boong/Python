import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 다중 선형 회귀 실습
# 앞서 배운 x가 1개인 선형 회귀 -> 단순 선형
# 다수 x로부터 y를 예측하는 다중 선형 회귀

x1_train = torch.FloatTensor([[73], [93], [83], [96], [73]])
x2_train = torch.FloatTensor([[80], [88], [91], [98], [66]])
x3_train = torch.FloatTensor([[75], [93], [90], [100], [70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

# 가중치 w와 편향 b를 선언 필요 w -> 3개 b -> 1개
w1 = torch.zeros(1, requires_grad=True)
w2 = torch.zeros(1, requires_grad=True)
w3 = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# optimizer 1e-04 - 0.0001 0.00001
optimizer = optim.SGD([w1, w2, w3, b], lr=1e-5)

# 학습 몇번 진행?
epoch_num = 1000
for epoch in range(epoch_num + 1):
    # 가설 xw + xw ..... + b
    hypothesis = x1_train * w1 + x2_train * w2 + x3_train * w2 + b

    # loss
    loss = torch.mean((hypothesis - y_train) ** 2)

    # loss H(x) 개선
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} w1 {:.3f} w2 {:.3f} w3 {:.3f} b {:.3f} loss {:.6f}'.format(
            epoch, epoch_num, w1.item(), w2.item(), w3.item(), b.item(), loss.item()))

'''
Epoch    0/1000 w1 0.290 w2 0.591 w3 0.000 b 0.003 loss 29661.800781
Epoch  100/1000 w1 0.407 w2 0.804 w3 0.000 b 0.005 loss 4.473953
Epoch  200/1000 w1 0.417 w2 0.799 w3 0.000 b 0.005 loss 4.353886
Epoch  300/1000 w1 0.426 w2 0.795 w3 0.000 b 0.005 loss 4.241400
Epoch  400/1000 w1 0.435 w2 0.790 w3 0.000 b 0.006 loss 4.136017
Epoch  500/1000 w1 0.444 w2 0.786 w3 0.000 b 0.006 loss 4.037305
Epoch  600/1000 w1 0.453 w2 0.782 w3 0.000 b 0.006 loss 3.944804
Epoch  700/1000 w1 0.461 w2 0.778 w3 0.000 b 0.006 loss 3.858150
Epoch  800/1000 w1 0.469 w2 0.774 w3 0.000 b 0.006 loss 3.776988
Epoch  900/1000 w1 0.477 w2 0.770 w3 0.000 b 0.006 loss 3.700932
Epoch 1000/1000 w1 0.485 w2 0.766 w3 0.000 b 0.007 loss 3.629710

lr = 1e-3, lr이 높아도 문제 낮아도 값이 안나오는 경우가 생김
Epoch    0/1000 w1 28.969 w2 59.098 w3 0.000 b 0.342 loss 29661.800781
Epoch  100/1000 w1 nan w2 nan w3 0.000 b nan loss nan
Epoch  200/1000 w1 nan w2 nan w3 0.000 b nan loss nan
Epoch  300/1000 w1 nan w2 nan w3 0.000 b nan loss nan
Epoch  400/1000 w1 nan w2 nan w3 0.000 b nan loss nan
Epoch  500/1000 w1 nan w2 nan w3 0.000 b nan loss nan
Epoch  600/1000 w1 nan w2 nan w3 0.000 b nan loss nan
Epoch  700/1000 w1 nan w2 nan w3 0.000 b nan loss nan
Epoch  800/1000 w1 nan w2 nan w3 0.000 b nan loss nan
Epoch  900/1000 w1 nan w2 nan w3 0.000 b nan loss nan
Epoch 1000/1000 w1 nan w2 nan w3 0.000 b nan loss nan
'''