n, m = map(int, input().split())

a = list(map(int, input().split()))
b = list(map(int, input().split()))

a = set(a)
b = set(b)

result = len(a-b)+len(b-a)
print(result)