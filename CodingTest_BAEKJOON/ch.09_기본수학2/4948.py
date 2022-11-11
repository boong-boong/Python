import sys

MAX_NUM = 123456

seive = [True for x in range(MAX_NUM*2+1)]
seive[0], seive[1] = False, False

for i in range(2,MAX_NUM+1):
    if seive[i] == True:
        for j in range(i*i, MAX_NUM*2+1, i):
            seive[j] = False

while True:
    n = int(sys.stdin.readline())
    if n == 0:
        break
    print(seive[n+1:(n*2)+1].count(True))

