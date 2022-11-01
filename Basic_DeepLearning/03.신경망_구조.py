import numpy as np
import matplotlib.pyplot as plt

# AND 게이트
def AND(a, b):
    input = np.array([a, b])

    #가중치 설정
    weights = np.array([0.4,0.4])
    bias = -0.6 # 값 조정 해보기

    #출력값
    value = np.sum(input * weights) + bias

    #반환값
    if value <= 0:
        return 0
    else:
        return 1

print(AND(0,0))
print(AND(0,1))
print(AND(1,0))
print(AND(1,1))

# AND 게이트 동작 시각화
x1 = np.arange(-2,2,0.01)
x2 = np.arange(-2,2,0.01)
bias = -0.6

y = (-0.4 * x1 - bias) / 0.4

plt.plot(x1, y, 'r--') # 조건이 들어왔을때 위쪽이면 1 아래이면 0
plt.scatter(0,0,color='orange', marker='o',s=150)
plt.scatter(0,1,color='orange', marker='o',s=150)
plt.scatter(1,0,color='orange', marker='o',s=150)
plt.scatter(1,1,color='black', marker='^',s=150)

plt.xlim(-0.5,1.5)
plt.ylim(-0.5,1.5)
plt.grid()
#plt.show()

# OR 게이트
def OR(a, b):
    input = np.array([a, b])

    #가중치 설정
    weights = np.array([0.4,0.4])
    bias = -0.3

    #출력값
    value = np.sum(input * weights) + bias

    #반환값
    if value <= 0:
        return 0
    else:
        return 1

print(OR(0,0))
print(OR(0,1))
print(OR(1,0))
print(OR(1,1))

# NAND 게이트
def NAND(a, b):
    input = np.array([a, b])

    #가중치 설정
    weights = np.array([-0.6,-0.6])
    bias = 0.7

    #출력값
    value = np.sum(input * weights) + bias

    #반환값
    if value <= 0:
        return 0
    else:
        return 1
# AND에서 리턴을 안바꾸고 수치를 바꾼 이유는 동작을 똑같이 맞추기 위해서 바꿈
# 하나의 퍼셉트론으로 수치만 바뀌면 다른 결과를 나타낼수 있음, 신경망 구조

print(NAND(0,0))
print(NAND(0,1))
print(NAND(1,0))
print(NAND(1,1))

# XOR 게이트
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1,x2)
    y = AND(s1,s2)

    return y # 구현할 수 없어서 신경망을 조합해서 만들어냄, 신경망 암흑기 해결

print(XOR(0,0))
print(XOR(0,1))
print(XOR(1,0))
print(XOR(1,1))

# Activation Function

# Step Function(계단 함수)
def step_function(x):
    if x > 0:
        return 1
    else:
        return 0

def step_function_for_numpy(x):
    y = x > 0
    value = y.astype(np.int)

    return value

print(step_function(-3))
print(step_function(5))

# 시그모이드 함수

def sigmoid(x):
    value = 1/ (1 + np.exp(-x))
    return value

# np.exp(10) # 지수 함수

print(sigmoid(3))
print(sigmoid(-3))

plt.grid()
x = np.arange(-5,5,0.01)
y1 = sigmoid(x)
y2 = step_function_for_numpy(x)

plt.plot(x, y1,'r-')
plt.plot(x, y2, 'b--')

# ReLu(x)

def ReLu(x):
    if x > 0:
        return x
    else:
        return 0

# Identity Function (항등 함수)
def identify_function(x):
    return x

# Softmax(a)
def Softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

a = np.array([0.3,0.2,3.0,-1.2])
print(Softmax(a))
print(np.sum(Softmax(a)))