import sys

n =  []

for i in range(28):
    n.append(int(sys.stdin.readline()))

for i in range(1,31):
    if i not in n:
        print(i)
