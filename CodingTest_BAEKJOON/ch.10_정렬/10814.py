import sys
n = int(sys.stdin.readline())

li = []
for i in range(n):
    age, name = sys.stdin.readline().split()
    li.append((int(age), name))

li.sort(key= lambda x: x[0])

for age, name in li:
    print(age, name)