m = int(input())
n = int(input())

arr = [True for i in range(n+1)]
arr[0], arr[1] =False, False
for i in range(2,n+1):
    if arr[i] == True:
        for j in range(i*i, n+1, i):
            arr[j] = False

sum = sum([x for x in range(m,n+1) if arr[x] == True])
if sum == 0:
    print(-1)
else:
    print(sum)
    print([x for x in range(m,n+1) if arr[x] == True][0])