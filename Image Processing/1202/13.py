# 텍스트 데이터 처리 01
import re

text_data = [' Hello. AI ',
             'Parking and going. By Kear Gua']

# 공백 제거
strip_space = [s.strip() for s in text_data]
print(strip_space)

# 마침표 제거
remove_periods = [s.replace('.', '') for s in strip_space]
print(remove_periods)

'''
출력결과
['Hello. AI', 'Parking and going. By Kear Gua']
['Hello AI', 'Parking and going By Kear Gua']
'''


def capitalizer(string: str) -> str: return string.upper()


# ['HELLO AI', 'PARKING AND GOING BY KEAR GUA']

temp = [capitalizer(s) for s in remove_periods]
print(temp)


def replace_letters_with_X(string: str) -> str :
    return re.sub(r'[a-zA-z]', 'X', string)


data = [replace_letters_with_X(s) for s in temp]
print(data)
