s = input()

result = []
length = 0
while len(s) >= length:
    for i in range(len(s)+1-length):
        result.append(s[i:i+length])
    length += 1

print(len(set(result))-1)