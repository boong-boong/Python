n, x = map(int, input().split())
li = list(map(int, input().split()))

for i in li:
    if i < x:
        print(i, end = " ")