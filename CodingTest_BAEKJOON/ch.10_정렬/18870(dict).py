import sys
n = int(sys.stdin.readline())

li = list(map(int, sys.stdin.readline().split()))

li2 = list(sorted(set(li)))

dic = {li2[i] : i for i in range(len(li2))}

for i in li:
    print(dic[i], end=' ')