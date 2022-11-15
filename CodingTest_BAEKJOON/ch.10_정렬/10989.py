import sys
input=sys.stdin.readline

N = int(input()) # 수의 개수

num = [0] * 10001

for _ in range(N) :
    temp = int(input())
    num[temp] += 1

for i in range(10001) :
    if num[i] != 0 :
        for j in range(num[i]) :
            print(i)

# import sys

# n = int(sys.stdin.readline())
# a=[]

# for i in range(n):
#     a.append(int(sys.stdin.readline()))

# def quick_sort(arr, start, end):
#     if start >= end:
#         return
#     pivot = start
#     left = start+1
#     right = end

#     while left <= right:
#         while left <= end and arr[left] <= arr[pivot]:
#             left += 1
#         while right > start and arr[right] >= arr[pivot]:
#             right -= 1
        
#         if left > right:
#             arr[right], arr[pivot] = arr[pivot], arr[right]
#         else:
#             arr[left], arr[right] = arr[right], arr[left]

#     quick_sort(arr,start,right-1)
#     quick_sort(arr,right+1,end)

# quick_sort(a,0,len(a)-1)

# for i in a:
#     print(i)