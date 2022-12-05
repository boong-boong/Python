# 구두점 삭제
# 구두점 글자의 딕셔너리를 만들어 translate() 적용
import sys
import unicodedata

text_data = ['HI!!!!!! I. love. this. song....???',
             '1111333%%&&*?!',
             'oiehgrq$#!',
             '~!@#$%^&*()-_;:,.<>?/`']  # 구두점이 포함된 텍스트

punctuation = dict.fromkeys(i for i in range(sys.maxunicode)
                            if unicodedata.category(chr(i)).startswith('P'))

test = [string.translate(punctuation) for string in text_data]
print(test)