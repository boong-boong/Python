import numpy as np

# np.array 생성
arr = np.array([1,2,3,4])
print(arr)
print(type(arr))

# 0으로 초기화된 배열
arr = np.zeros((3,3))
print(arr)

# 빈 값으로 만들어진 배열, 값이 들어가있음
arr = np.empty((4,4))
print(arr)

# 1로 채우기
arr = np.ones((3,3))
print(arr)

# ndarray 배열의 모양, 차수, 데이터 타입 확인
arr = np.array([[1,2,3,], [4,5,6]])
print(arr)

print(arr.shape) # 모양
print(arr.ndim) # 차원
print(arr.dtype) # 타입

# type 바꾸기
arr_float = arr.astype(np.float64)
print(arr_float)
print(arr_float.dtype)


arr_str = np.array(['1','2','3'])
print(arr_str.dtype)

arr_int = arr_str.astype(np.int64)
print(arr_int.dtype)

# ndarray 배열의 연산
arr1 = np.array([[1,2],[3,4]])
arr2 = np.array([[1,2],[3,4]])

print(arr1+arr2)
print(np.add(arr1, arr2))

print(arr1 * arr2)
print(np.multiply(arr1, arr2))

# ndarray 배열 슬라이싱 하기

arr = np.array([[1,2,3], [4,5,6], [7,8,9]])

arr_1 = arr[:2, 1:3]
print(arr_1)

print(arr[0,2])
print(arr[::2, ::2])

idx = arr > 3
print(idx)
print(arr[idx])

# wine Quality 데이터
redwine = np.loadtxt(fname = './Basic_AI/winequality-red.csv', delimiter=';', skiprows = 1)
print(redwine)

# 합계
print(redwine.sum())

# 평균
print(redwine.mean())

# 축(axis)
print(redwine.sum(axis=0))

print(redwine.mean(axis=0))

print(redwine[:,0]) # 전체 row 1번째 col
print(redwine[:,0].mean())

print(redwine.max(axis=0)) # 각 변수의 최댓값