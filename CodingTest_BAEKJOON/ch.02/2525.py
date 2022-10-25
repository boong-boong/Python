h, m = map(int, input().split())
t = int(input())
if(m+t>=60):
    if(h+int((m+t)/60) >=24):
        h = (h+int((m+t)/60))%24
    else:
        h += int((m+t)/60)
    m = (m+t)%60
else:
    m += t

print(h,m)