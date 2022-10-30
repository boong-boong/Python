n = int(input())

m = [[0 for col in range(101)]for row in range(101)]


for i in range(n):
    a, b = map(int, input().split())

    for j in range(a, a+10):
        for k in range(b, b+10):
            m[j][k] = 1

cnt = 0
for i in m:
    cnt += i.count(1)

print(cnt)