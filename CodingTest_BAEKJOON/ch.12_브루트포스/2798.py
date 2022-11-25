n, m = map(int, input().split())

result = 0
num = []

num = list(map(int, input().split()))

for i in range(n-2):
    for j in range(i+1,n-1):
        for k in range(j+1, n):
            if m - (num[i] + num[j] + num[k]) < 0:
                continue
            else:
                result = max(result, num[i] + num[j] + num[k])
                
print(result)