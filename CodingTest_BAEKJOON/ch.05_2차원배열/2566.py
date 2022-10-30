import sys
metrix = []

max_num, max_n, max_m = 0, 0, 0

for i in range(9):
    metrix.append(list(map(int, sys.stdin.readline().split())))

for i in range(9):
    if max_num < max(metrix[i]):
        max_num = max(metrix[i])
        max_n = i
        max_m = metrix[i].index(max(metrix[i]))

print(max_num)
print(max_n+1,max_m+1)