n, m = map(int, input().split())

a = set()
for i in range(n):
    a.add(input())

b = set()
for i in range(m):
    b.add(input())

result = sorted(list(a & b))

print(len(result))

for i in result:
    print(i)
# set이 시간은 더 걸림
# n, m = map(int, input().split())
#
# no_hear = []
# no_see = []
#
# for i in range(n):
#     no_hear.append(input())
#
# for i in range(m):
#     no_see.append(input())
#
# no_hear = {i for i in no_hear}
# no_see = {i for i in no_see}
#
# cnt = 0
# no_hear_see = []
# for i in no_hear:
#     if i in no_see:
#         no_hear_see.append(i)
#         cnt += 1
#
# print(cnt)
# no_hear_see.sort()
# for i in no_hear_see:
#     print(i)