import sys

n = list(map(int, sys.stdin.readline().split()))

a = [[0 for i in range(n[1])]for i in range(n[0])]
b = [[0 for i in range(n[1])]for i in range(n[0])]

for i in range(n[0]):
    a[i] = list(map(int, sys.stdin.readline().split()))

for i in range(n[0]):
    b[i] = list(map(int, sys.stdin.readline().split()))

for i in range(n[0]):
    for j in range(n[1]):
        print(a[i][j]+b[i][j])