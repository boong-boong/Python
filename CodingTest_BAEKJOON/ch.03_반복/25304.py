bill = int(input())
items = int(input())
cost = 0
for i in range(items):
    item, num = map(int, input().split())
    cost += item * num

if(cost == bill):
    print('Yes')
else:
    print('No')