n = int(input())
cnt = 0

def hansu(n):
    a =[]
    while n>0:
        a.append(n%10)
        n = int(n/10)
    
    if isHansu(a):
        return True
    
    return False

def isHansu(a):
    if len(a) <= 2:
        return True
    for i in range(len(a)-2):
        if(a[i] - a[i+1] != a[i+1] - a[i+2]):
            return False
    return True

for i in range(1, n+1):
    if hansu(i):
        cnt += 1

print(cnt)