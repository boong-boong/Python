i = int(input())

for a in range(i):
    h, w, n = map(int, input().split())

    if n%h == 0:
        y = h
        x = n//h
    else:
        y = n%h
        x = n//h +1
    

    print('{}{:02d}'.format(y,x))
