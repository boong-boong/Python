import sys
n = int(sys.stdin.readline())
li =[]

for i in range(n):
    x, y = map(int, sys.stdin.readline().split())
    li.append((y, x))

li.sort()
for y,x in li:
    print(x,y)

'''
import sys
n = int(sys.stdin.readline())
li =[]

for i in range(n):
    x ,y = map(int, sys.stdin.readline().split())
    li.append((x, y))

li.sort(key= lambda x: (x[1], x[0]))
for x,y in li:
    print(x,y)
'''