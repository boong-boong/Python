import random

random_number = random.randint(1,100)
# print(random_number)

game_count = 1
while True:

    try:

        myNumber = int(input('1-100 사이 숫자를 입력하세요'))
    
        if myNumber > random_number:
            print('down')
        elif myNumber < random_number:
            print('up')
        else:
            print(f'{game_count} 만에 맞추셨습니다.')
            break
    
        game_count += 1

    except:
        print("숫자를 입력하세요")