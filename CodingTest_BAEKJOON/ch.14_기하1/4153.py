tri = []

while True:
    tri = list(map(int, input().split()))
    if max(tri) == 0:
        break
    tri.sort()
    if tri[-1]**2 == tri[0]**2 + tri[1]**2:
        print('right')
    else:
        print('wrong')