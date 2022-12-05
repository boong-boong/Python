# 단어 중요도 가중치 부여
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

text_data = np.array(['I love Brazil. Brazil!!', 'Sweden is best', 'Germany beats both'])

# tf-idf 특성 행렬 생성
tfidf = TfidfVectorizer()
feature_matrix = tfidf.fit_transform(text_data)
feature_matrix.toarray()  # tf_idf 특성 행렬을 밀집 배열로 확인
# print(feature_matrix)
# 특성 이름 확인
tf = tfidf.vocabulary_
print('...', tf)
