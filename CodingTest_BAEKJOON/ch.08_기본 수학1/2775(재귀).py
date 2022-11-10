# 시간초과
import sys

t = int(sys.stdin.readline())

def peopleNumber(k,n):
    if k == 0:
        return n
    if n == 1:
        return 1
    else:
        return peopleNumber(k, n-1)+peopleNumber(k-1,n)

for i in range(t):
    k = int(sys.stdin.readline())
    n = int(sys.stdin.readline())
    print(peopleNumber(k,n))
