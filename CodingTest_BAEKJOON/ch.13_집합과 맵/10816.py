n = int(input())

num = list(map(int, input().split()))
d_num = {}
for i in num:
    if i not in d_num:
        d_num[i] = 1
    else:
        d_num[i] += 1

m = int(input())
check = list(map(int, input().split()))

for i in check:
    try:
        print(d_num[i], end=' ')
    except KeyError as e:
        print(0, end=' ')
