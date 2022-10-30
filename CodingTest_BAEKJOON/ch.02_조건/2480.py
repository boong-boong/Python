a,b,c = map(int, input().split())

if(a == b == c):
    print(a*1000 + 10000)
elif(a == b and a != c):
    print(a*100 + 1000)
elif(a == c and a != b):
    print(a*100 + 1000)
elif(b == c and c != a):
    print(b*100 + 1000)
else:
    print(max(a,b,c)*100)