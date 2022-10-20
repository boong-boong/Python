import bs4
import requests

# 네이버 쇼핑 데이터
url = 'https://search.shopping.naver.com/search/category/100005307'
result = requests.get(url).text # 이미지면 content, html이면 text

bsObj = bs4.BeautifulSoup(result, 'html.parser') # 파싱

#bsObj.find('a',{'class':'basicList_link__JLQJf'}) # a 태그에서 class명을 찾아라
#bsObj.find('a',{'class':'basicList_link__JLQJf'}).text # 텍스트만 가져옴

items = bsObj.find_all('a',{'class':'basicList_link__JLQJf'}) # find_all 모두 찾아줘

'''for item in items: 
    print(item.text)''' # 4개만 정상적으로 가져옴

print(items[0].text)
print(items[1].text)
print(items[2].text)
print(items[3].text)
#print(items[:3].text) 리스트에는 text속성이 없어 작동되지 않음, 슬라이싱 말고 반복문을 통해 출력하기