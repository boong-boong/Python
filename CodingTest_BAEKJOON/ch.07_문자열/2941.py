n = input()

patterns = ['c=', 'c-', 'dz=', 'd-', 'lj', 'nj','s=','z=']

for pattern in patterns:
    n = n.replace(pattern,'a')

print(len(n))