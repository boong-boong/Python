n, m = map(int, input().split())

n_word =[]
m_word = []

for i in range(n):
    n_word.append(input())

for i in range(m):
    m_word.append(input())

n_word = {i for i in n_word}

cnt = 0

for i in m_word:
    if i in n_word:
        cnt += 1

print(cnt)