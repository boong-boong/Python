import sys

n = int(sys.stdin.readline())
n_score = []
for i in range(n):
    n_score = list(map(int, sys.stdin.readline().split()))
    avg = sum(n_score[1:])/n_score[0]
    cnt = 0
    for score in n_score[1:]:
        if score > avg:
            cnt += 1
    print('{:.3f}%'.format((cnt/n_score[0])*100))