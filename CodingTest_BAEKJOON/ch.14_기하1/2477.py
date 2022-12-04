n = int(input())

arr = [list(map(int, input().split())) for i in range(6)]

w, h = 0, 0
w_idx, h_idx = 0, 0

for i in range(len(arr)):
    if arr[i][0] == 3 or arr[i][0] == 4:
        if h < arr[i][1]:
            h = arr[i][1]
            h_idx = i
    if arr[i][0] == 1 or arr[i][0] == 2:
        if w < arr[i][1]:
            w = arr[i][1]
            w_idx = i

subW = abs(arr[(w_idx - 1) % 6][1] - arr[(w_idx + 1) % 6][1])
subH = abs(arr[(h_idx - 1) % 6][1] - arr[(h_idx + 1) % 6][1])

result = ((w * h) - (subW * subH)) * n
print(result)
