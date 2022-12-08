n = int(input())


def hanoi(n, start, mid, end):
    if n == 1:
        print(start, end)
        return
    hanoi(n-1, start, end, mid)
    hanoi(1, start, mid, end)
    hanoi(n-1, mid, start, end)


print(2**n-1)
hanoi(n, 1, 2, 3)

'''
탑을 두 덩이로 가정
위(n-1)개의 원판, 아래 n번째 원판
위 원판들을 B로 이동
아래 원판을 C로 이동 -> 출력
다시 B에있던 원판들을 C로 이동
'''