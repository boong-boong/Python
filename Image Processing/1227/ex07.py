# 단층 퍼셉트론 이용하면 AND NAND OR 게이트 구현가능

# AND 게이트 구현
# 2개의 입력 값이 모두 1인경우 output 1 아닌 경우 0

# AND 게이트를 만족하는 2개의 가중치와 편향 값에는 무엇이 있는가?
# w1, w2, b 라고 하면 (w 가중치, b 편향)
# 0.5 0.2 0.1 -> 결과 구해진 수치 [0.5, 0.5, -0.7]

def and_gate(x1, x2):
    w1 = 0.5
    w2 = 0.5
    b = -0.7

    # x1*w1 + x2*w2 +b
    result = x1 * w1 + x2 * w2 + b
    print(result)
    # 0.30000000000000004

    if result <= 0:
        return 0
    else:
        return 1


# print(and_gate(1, 1))

# 단층 퍼셉트론 NAND 게이트 구현
# 두개의 입력값이 1인경우에만 출력 값 0 나머지 1

def nand_gate(x1, x2):
    w1 = -0.5
    w2 = -0.5
    b = 0.7

    result = x1 * w1 + x2 * w2 + b

    if result <= 0:
        return 0
    else:
        return 1


# print(nand_gate(1, 1))

def or_gate(x1, x2):
    w1 = 0.5
    w2 = 0.5
    b = -0.4

    result = x1 * w1 + x2 * w2 + b

    if result <= 0:
        return 0
    else:
        return 1


print(or_gate(1, 1))
