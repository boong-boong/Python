n = int(input())

cnt5, cnt3 = 0,0
cnt5 = n//5
n = n%5

while n%3 != 0:
    if cnt5 == -1:
        break
    cnt5 -= 1
    n += 5
if cnt5 == -1:
    print(-1)
else:
    cnt3 = n//3
    print(cnt5+cnt3)