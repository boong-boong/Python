n = int(input())

a = list(map(int,input().split()))

def prime(n):
    if n <2:
        return False
    
    for i in range(2,int(n ** 0.5)+1):
        if n%i ==0:
            return False
    return True

cnt = 0

for i in a:
    if prime(i):
        cnt += 1

print(cnt)
