import sys

t = int(sys.stdin.readline())

for i in range(t):
    k = int(sys.stdin.readline())
    n = int(sys.stdin.readline())
    
    f0 = [x for x in range(1, n+1)]

    for a in range(k):
        for s in range(1,n):
            f0[s] += f0[s-1]
    print(f0[-1])
    