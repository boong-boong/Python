A = int(input())
B = input()

for i in B[::-1]: # ''.join(reversed(B)) 배열 뒤집기 두가지 방법
    print(A*int(i))
print(A*int(B))