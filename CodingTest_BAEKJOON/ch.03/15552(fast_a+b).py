import sys

count = int(input())

for i in range(count):
    a,b = map(int, sys.stdin.readline().split()) # 반복으로 입력이 있다면 메모리 절약 및 속도가 더 빠르다.
    print(a+b)