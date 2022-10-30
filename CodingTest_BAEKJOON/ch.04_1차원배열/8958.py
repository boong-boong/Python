from inspect import stack
from os import scandir
import sys

n = int(sys.stdin.readline())


for i in range(n):
    ox = sys.stdin.readline()
    score = 1
    sum = 0
    s = 1
    for j in ox[:]:
        if j =='O':
            sum += (score * s)
            s += 1
        else:
            s = 1
    print(sum)