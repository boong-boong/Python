n = int(input())

cnt = 0
i = 0

while n > i:
    cnt += 1
    i += cnt

if cnt % 2 == 0:
    s = str(cnt-(i-n)) + '/' + str(1+(i-n))
    print(s)
else:
    s = str(1+(i-n)) + '/' + str(cnt-(i-n))
    print(s)
