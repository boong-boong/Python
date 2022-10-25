import sys
i = int(input())

for n in range(i):
    a, b = map(int, sys.stdin.readline().split())
    print('Case #{}: {} + {} = {}'.format(n+1,a,b,a+b))
