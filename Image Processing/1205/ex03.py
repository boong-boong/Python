from nltk.stem.porter import PorterStemmer

# 단어 토큰 생성
tokenize_words = ['i', 'am', 'played', 'game', 'meeting', 'eaten']

# 어간 추출기 생성
porter = PorterStemmer()

word_list = []
# 어간 추출기 적용
for word in tokenize_words:
    word_list.append(porter.stem(word))

print(word_list)