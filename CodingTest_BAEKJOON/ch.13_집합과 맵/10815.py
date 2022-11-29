import sys
n = int(sys.stdin.readline())
card = list(map(int, sys.stdin.readline().split()))
card = {i for i in card}

m = int(sys.stdin.readline())
li = list(map(int, sys.stdin.readline().split()))

for i in li:
    if i in card:
        print(1, end=' ')
    else:
        print(0, end=' ')