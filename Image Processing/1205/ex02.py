# 불용어 삭제
import nltk
from nltk.corpus import stopwords

# 불용어 데이터 다운로드 -> 179개 정도
nltk.download('stopwords')

# 단어 토큰을 생성
tokenize_words = ['i', 'am', 'going', 'to', 'go']

stop_words = stopwords.words('english')  # 불용어 로드

temp = [word for word in tokenize_words if word not in stop_words]

stop_data = stop_words[:5]
print(stop_words)
print(temp)
print(stop_data)