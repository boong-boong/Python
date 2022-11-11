m,n = map(int, input().split())

seive = [True for x in range(n+1)]
seive[0], seive[1] = False, False

for i in range(2, int(n**0.5)+1):
    if seive[i] == True:
        for j in range(i*i, n+1, i):
            seive[j] = False

for i in range(m,n+1):
    if seive[i] == True:
        print(i)