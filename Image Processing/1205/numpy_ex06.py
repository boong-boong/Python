import numpy as np
# np.filp()
# np.reversed()

# 실행마다 랜덤값
print(np.random.randint(0, 10, (2, 3)))

# 실해마다 값이 변경되지 않도록 seed 생성
np.random.seed(7777)
print(np.random.randint(0, 10, (2, 3)))
