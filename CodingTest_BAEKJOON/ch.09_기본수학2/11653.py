n = int(input())

i =2

while n>1:
    if n%i == 0:
        n //= i
        print(i)
    else:
        i += 1

# 수정전
# while n>1:
#     i = 2
#     if n%i == 0:
#         n //= i
#         print(i)
#     else:
#         while n%i != 0:
#             i += 1
#         n //= i
#         print(i)