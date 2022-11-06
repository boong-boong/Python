s = input()
s = s.upper()
set_s = set(s)

if len(set_s) <= 1:
    print(set_s.pop())
    p = -1

max_cnt = 0
alpha = ''

for a in set_s:
    cnt = s.count(a)
    if max_cnt < cnt:
        #print(cnt, max_cnt)
        max_cnt = cnt
        alpha = a
        p = 1
    elif max_cnt == cnt:
        p = 0

if p == 1:
    print(alpha)
elif p == 0:
    print('?')