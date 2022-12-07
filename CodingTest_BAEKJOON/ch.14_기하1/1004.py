t = int(input())
for i in range(t):
    x1, y1, x2, y2 = map(int, input().split())
    n = int(input())
    cnt = 0
    for j in range(n):
        cx, cy, r = map(int, input().split())  #현재 행성
        # 유클리디안 거리
        d1 = ((x1-cx)**2 + (y1-cy)**2)**0.5
        d2 = ((x2-cx)**2 + (y2-cy)**2)**0.5

        if ((d1 < r) and (d2 > r)) or ((d1 > r) and (d2 < r)):
            cnt += 1
    print(cnt)

'''
행성을 피할 수 있으면 무조건 피해서 감
즉, 출발점 혹은 도착점이 행성 내부에 있을 경우에만, 행성경계를 통과
현재 위치가 행성안에 포함되었는지 확인
'''