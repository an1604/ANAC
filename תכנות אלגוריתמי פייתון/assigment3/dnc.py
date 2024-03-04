def dnc(baseFunc, combineFunc):
    def func(arr):
        if len(arr) == 1:
            return baseFunc(arr[0])
        size = len(arr) // 2
        left = func(arr[:size])
        right = func(arr[size:])
        return combineFunc(left, right)

    return func


def maxAreaHist(hist):
    def max_area_divide_conquer(arr, start, end):
        if start == end:
            return arr[start]

        mid = (start + end) // 2

        max_left = max_area_divide_conquer(arr, start, mid)
        max_right = max_area_divide_conquer(arr, mid + 1, end)

        min_height = min(arr[mid], arr[mid + 1])
        max_middle = min_height * 2
        left = mid
        right = mid + 1

        while left > start or right < end:
            if right < end and (left == start or arr[left - 1] < arr[right + 1]):
                right += 1
                min_height = min(min_height, arr[right])
            else:
                left -= 1
                min_height = min(min_height, arr[left])

            max_middle = max(max_middle, min_height * (right - left + 1))

        return max(max_left, max_right, max_middle)

    if not hist:
        return 0

    return max_area_divide_conquer(hist, 0, len(hist) - 1)
