test = int(input())

for a in range(test):

    i, s = input().split()

    for n in s:
        print(n*int(i), end='')
    print('')