n = int(input())

def decom_add(n):
    if n//10 == 0:
        return n%10
    return n%10 + decom_add(n//10)

for i in range(1, n+1):
    if n == decom_add(i)+i:
        print(i)
        break
    if i == n:
        print(0)