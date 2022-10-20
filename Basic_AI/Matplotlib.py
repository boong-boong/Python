import matplotlib.pyplot as plt

# Style 적용하기
plt.style.use('classic')

import numpy as np

# Sample Data 생성
x = np.linspace(0, 10, 100) # 0~10 사이 100개의 데이터를 만듦
print(x)

plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))
#plt.show()

plt.figure()
plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))

fig = plt.figure()
plt.plot(x, np.sin(x),'-')
plt.plot(x, np.cos(x),'--')

#fig.savefig('./Basic_AI/my_figure.png')

#plt.show()

print(fig.canvas.get_supported_filetypes())

plt.subplot(2,1,1)

plt.subplot(2,1,1) #row, column, index
plt.plot(x, np.sin(x))
plt.subplot(2,1,2)
plt.plot(x, np.cos(x))

plt.style.use('seaborn-whitegrid')

fig = plt.figure()
ax = plt.axes()

x = np.linspace(0,10,100)
ax.plot(x, np.sin(x))

plt.plot(x, np.sin(x-0), color='blue') #색상의 이름으로 지정
plt.plot(x, np.sin(x-1), color='g') #짧은 색상코드 RGB, CMYK
plt.plot(x, np.sin(x-2), color='1') #0-1사이 회색조 표현
plt.plot(x, np.sin(x-3), color='#FFDD44') #16진수로 표현 RGB
plt.plot(x, np.sin(x-4), color=(1.0,0.2,0.3)) #RGB 튜플로 표현
plt.plot(x, np.sin(x-5), color='chartreuse')

plt.plot(x, x+0, linestyle='solid')
plt.plot(x, x+1, linestyle='dashed')
plt.plot(x, x+2, linestyle='dashdot')
plt.plot(x, x+3, linestyle='dotted')

plt.plot(x, x+4, linestyle='-')
plt.plot(x, x+5, linestyle='--')
plt.plot(x, x+6, linestyle='-.')
plt.plot(x, x+7, linestyle=':')

plt.plot(x, x+0, '-g')
plt.plot(x, x+1, '--c')
plt.plot(x, x+2, '-.k')
plt.plot(x, x+3, ':r')

#plot 조정하기: 축 경계

plt.plot(x, np.sin(x))
plt.xlim(-1,11)
plt.ylim(-1.5,1.5)

plt.plot(x, np.sin(x))
plt.xlim(10,0)
plt.ylim(-1.5,1.5)

plt.plot(x, np.sin(x))
plt.axis([-1,11,-1.5,1.5]) #xlim, ylim

plt.plot(x, np.sin(x))
plt.axis('tight')

plt.plot(x, np.sin(x))
plt.axis('equal')

# 플롯에 레이블 붙이기
plt.plot(x, np.sin(x))
plt.title('A sin curve')
plt.xlabel('x')
plt.ylabel('sin(x)')

plt.plot(x, np.sin(x), '-g', label='sin(x)') # 선에 라벨
plt.plot(x, np.cos(x), ':b', label='cos(x)')
plt.legend() #범례

# 산점도의 출력

x = np.linspace(0,10,30)
y = np.sin(x)

plt.plot(x,y,'o',color='black')

rng = np.random.RandomState(0)
for marker in ['o',',','x','+','v','<','>','s','d']: #마커
  plt.plot(rng.rand(5),rng.rand(5), marker, label='marker={0}'.format(marker))
  plt.legend()
  plt.xlim(0, 1.8)

plt.plot(x,y,'-ok')

plt.plot(x, y, '-p', color='grey', markersize=15, linewidth=4, markerfacecolor='white', markeredgecolor = 'red', markeredgewidth = '2') #p = 5각형

plt.scatter(x, y, marker='o',s=150)