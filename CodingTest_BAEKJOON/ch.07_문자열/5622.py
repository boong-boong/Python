n = input()

time = 0
pattern = ['ABC', 'DEF', 'GHI', 'JKL', 'MNO', 'PQRS', 'TUV', 'WXYZ']

for s in n:
    time += 2
    for dial in pattern:
        time += 1
        if s in dial:
            break

print(time)