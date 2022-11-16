n = int(input())

li = []

for i in range(n):
    x,y = map(int,input().split())
    li.append((x,y))

li.sort()
for x,y in li:
    print(x, y)