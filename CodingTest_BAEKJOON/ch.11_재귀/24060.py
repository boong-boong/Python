a, k = map(int, input().split())

arr = list(map(int, input().split()))


def merge(left, right):
    i, j = 0, 0
    sorted_list = []

    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            sorted_list.append(left[i])
            result.append(left[i])
            i += 1
        else:
            sorted_list.append(right[j])
            result.append(right[j])
            j += 1
    while i < len(left):
        sorted_list.append(left[i])
        result.append(left[i])
        i += 1
    while j < len(right):
        sorted_list.append(right[j])
        result.append(right[j])
        j += 1

    return sorted_list


def merge_sort(arr):
    if len(arr) == 1:
        return arr

    # 분할
    mid = (len(arr)+1) // 2
    left_arr = arr[:mid]
    right_arr = arr[mid:]

    # merge
    left = merge_sort(left_arr)
    right = merge_sort(right_arr)
    return merge(left, right)


result = []
merge_sort(arr)
if len(result) >= k:
    print(result[k-1])
else:
    print(-1)

