from itertools import groupby
from operator import index, invert
from telnetlib import SE
from pandas import Series
from pandas import DataFrame

fruit = Series([2500, 3800, 1200, 600], index = ['apple', 'banana', 'peer', 'cherry'])
print(fruit)

# 값과 인덱스를 추출
print(fruit.values)
print(fruit.index)
print(fruit['apple'])

# Dict 표현
fruitData = {'apple': 2500, 'banana': 3800, 'peer': 1200, 'cherry': 600}
fruit = Series(fruitData) # Dict를 통해 Series 객체 생성

print(type(fruitData))
print(type(fruit))

# Series 객체의 이름과 컬러명을 설정
fruit.name = 'fruitPrice'
fruit.index.name = 'fruitName'

print(fruit)

# DataFrame
fruitData = {'fruitName':['apple', 'banana', 'peer', 'cherry'],
            'fruitPrice': [2500, 3800, 1200, 600],
            'num':[10, 5, 3, 8]
            }

fruitFrame = DataFrame(fruitData)
print(fruitFrame)

# 컬럼 순서 지정하기
fruitFrame = DataFrame(fruitData, columns=['fruitPrice','num','fruitName'])
print(fruitFrame)

# 득정 항목 추출
print(fruitFrame['fruitName'])
print(fruitFrame.fruitName) # 두가지 표현법이 존재

# 컬럼 추가하기
fruitFrame['year'] = '2022' # 하나만 적으면 모두 들어감, 없다면 알아서 생성
fruitFrame['year'] = ['2022','2002','2003','2021'] 
print(fruitFrame)

# Series 객체의 추가
variable = Series([4,2,1], index=[0,2,3]) # index에 1이 없음, 없다면 null이 들어감

fruitFrame['stock'] = variable
print(fruitFrame)

# 자료구조 다루기
# 데이터 구조의 항목 삭제
print(fruit)
fruit.drop('banana')
print(fruit)
print(fruit.drop('banana')) # drop은 실제로 삭제해주지 않음, 결과에서 제외하고 보여줌
#fruit = fruit.drop('banana')

fruitName = fruitData['fruitName']
print(fruitName)

fruitFrame = DataFrame(fruitData, index=fruitData['fruitName'], columns=['fruitPrice','num']) #dict객체로 만들겠다
print(fruitFrame)

print(fruitFrame.drop(['apple','cherry']))

# 컬럼 삭제
print(fruitFrame.drop(['num'], axis=1)) # 축을 변경해서 삭제해야함

# pandas Slice를 사용하는 방법
print(fruit['apple':'banana']) # 리스트 슬라이싱과 다름점은 지정한곳까지 잘라줌

# Seires 객체의 기본 연산
fruit1 = Series([5,9,10,3], index = ['apple', 'banana', 'peer', 'cherry'])
fruit2 = Series([3,2,9,5,1], index = ['apple','orange', 'banana', 'cherry', 'mango'])

print(fruit1+fruit2) # 둘 다 있는것은 연상이 되고 없다면 널값이 들어감 (널값과 연산은 무조건 널)

# DataFrame 객체의 기본 연산
fruitData1 = {'Ohio': [4,8,3,5], 'Texas':[0,1,2,3]}
fruitData2 = {'Ohio': [3,0,2,1,7], 'Colorado': [5,4,3,6,0]}

fruitFrame1 = DataFrame(fruitData1, columns=['Ohio','Texas'], index= ['apple','banana','cherry', 'peer'])
fruitFrame2 = DataFrame(fruitData2, columns=['Ohio','Colorado'], index= ['apple','orange','banana','cherry', 'mango'])

print(fruitFrame1)
print(fruitFrame1 + fruitFrame2)

# 데이터의 정렬

#Series의 정렬
print(fruit)
print(fruit.sort_values()) # fruit.sort_values(ascending=True)
print(fruit.sort_values(ascending=False))

fruitName = fruitData['fruitName']

fruitFrame = DataFrame(fruitData, index=fruitName, columns=['num','fruitPrice'])
print(fruitFrame)

print(fruitFrame.sort_index(ascending=False))
print(fruitFrame.sort_index(axis=1)) # columns로 정렬

print(fruitFrame.sort_values(by=['fruitPrice', 'num'])) # 특정 columns로 정렬, 두가지 이상 사용가능 (가격이 같다면 갯수로 정렬)

#pandas 이용한 기초 분석
import pandas as pd

german = pd.read_csv('http://freakonometrics.free.fr/german_credit.csv')

print(type(german))
print(german.columns.values)

german_sample = german[['Creditability','Duration of Credit (month)', 'Purpose', 'Credit Amount']]

print(german_sample.min())
print(german_sample.max())
print(german_sample.mean())

print(german_sample.head()) # 앞부분 데이터 5개 보여줌

print(german_sample.corr()) # correlation 상관관계

# Group By를 이용한 계산 및 요약 통계
german_sample = german[['Credit Amount','Type of apartment']]
print(german_sample)

german_grouped = german_sample['Credit Amount'].groupby(german_sample['Type of apartment'])
print(german_grouped.mean())

print(german_grouped.max())

german_sample = german[['Type of apartment','Sex & Marital Status','Credit Amount']]
#print(german_sample)
for type, group in german_sample.groupby('Type of apartment'): # type of apartment의 type이 3개이기 때문에 3개가 나옴
    print(type)
    print(group.head())

for (type, sex), group in german_sample.groupby(['Type of apartment', 'Sex & Marital Status']):
    print(type, sex)
    print(group.head(n=3))

# 행성 데이터 가져오기
import seaborn as sns

planets = sns.load_dataset('planets')

print(planets.shape)
print(planets.head())
print(planets.head)

print(planets.dropna()) #널값을 드랍

births = pd.read_csv('https://raw.githubusercontent.com/jakevdp/data-CDCbirths/master/births.csv') # 미국 인구통계국 통계

#print(births.head)

#print(births['year'] // 10 * 10)
births['decade'] = births['year'] // 10 * 10
print(births.head())

print(births.pivot_table('births', index='decade', columns='gender', aggfunc='sum')) # agg 집계

import matplotlib.pyplot as plt # 그래프

births.pivot_table('births', index='decade', columns='gender', aggfunc='sum').plot()
births.pivot_table('births', index='year', columns='gender', aggfunc='sum').plot()
plt.show()