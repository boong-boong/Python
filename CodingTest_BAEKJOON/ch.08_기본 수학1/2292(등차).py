i = int(input())

step = 1
cnt = 1

while step < i:
    step += 6*cnt
    cnt += 1
print(cnt)