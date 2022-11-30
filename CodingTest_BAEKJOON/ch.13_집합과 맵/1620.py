import sys

n, m = map(int, sys.stdin.readline().split())

pokemon = {}

for i in range(1,n+1):
    s = sys.stdin.readline().rstrip()
    pokemon[s] = i
    pokemon[i] = s

for i in range(m):
    s = sys.stdin.readline().rstrip()
    if s.isdigit():
        print(pokemon[int(s)])
    else:
        print(pokemon[s])