import math
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

# 일차함수
def linear_function(x):
    a = 0.5
    b = 2

    return a * x + b

# print(linear_function(5))

x = np.arange(-5,5,0.1)
y = linear_function(x)

# plt.plot(x, y)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Linear Function')
# plt.show()

# 이차함수 y = ax^2 + bx + c
def quadratic_function(x):
    a = 1
    b = -1
    c = -2
    return a*x**2 + b*x + c

y = quadratic_function(x)

# plt.plot(x,y)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Quadratic Function')
# plt.show()

# 삼차함수(다항함수) y = ax^3 +bx^2 + cx + d
def cubic_function(x):
    a = 4
    b = 0
    c = -1
    d = -8

    return a*x**3 + b*x**2 +c*x + d

y = cubic_function(x)

# plt.plot(x,y)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Cubic Function')
# plt.show()

def my_func(x):
    a = 1
    b = -3
    c = 10
    return a*x**2 + b*x + c

x = np.arange(-10,10,0.1)
y = my_func(x)

plt.plot(x,y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('My Func')
plt.scatter(1.5, my_func(1.5))
plt.show()

# 지수함수 y = a^x
def exponential_function(x):
    a = 4
    return a**x

print(exponential_function(4))
print(exponential_function(0))

x = np.arange(-3, 2, 0.1)
y = exponential_function(x)

plt.plot(x,y)
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-4,3)
plt.ylim(-1,15)
plt.title('exponential_Function')
plt.show()