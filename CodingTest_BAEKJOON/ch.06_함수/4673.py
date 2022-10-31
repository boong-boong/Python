def generated(a):
    sum = a
    while a > 0:
        sum += a%10
        a = int(a/10)
    return sum

generated_num = []

for i in range(1, 10001):
    generated_num.append(generated(i))

for i in range(1, 10001):
    if i not in generated_num:
        print(i)