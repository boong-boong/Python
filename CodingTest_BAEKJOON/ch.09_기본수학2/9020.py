import sys

seive = [True for x in range(10000)]
seive[0], seive[1] = False, False

for i in range(2, int(10000**0.5)+1):
    if seive[i] == True:
        for j in range(i*i, 10000, i):
            seive[j] = False

n = int(sys.stdin.readline())

for i in range(n):
    m = int(sys.stdin.readline())
    if m%2 == 0:
        a = m//2
        b = m//2
    else:
        a = m//2
        b = m//2+1
    while seive[a] != True or seive[b] != True:
        a -= 1
        b += 1
    print(a,b)