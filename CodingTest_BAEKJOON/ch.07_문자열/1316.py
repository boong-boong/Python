import sys
line = int(sys.stdin.readline())

cnt = 0

def group_word_checker(s):
    temp = ''
    for i in s:
        if i in temp:
            if temp[-1] != i:
                return False
            temp += i
        else:
            temp += i
    return True

for i in range(line):
    s = sys.stdin.readline()
    if group_word_checker(s):
        cnt += 1
print(cnt)