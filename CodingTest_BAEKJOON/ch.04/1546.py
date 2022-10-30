import sys

score = []

n = int(sys.stdin.readline())

score = list(map(int, sys.stdin.readline().split()))

avg = 0
for i in score:
    avg += i/max(score)*100

print(avg/n)