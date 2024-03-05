def allSumsDP(arr):
    ret = set()
    stack = {(0, 0)}

    while stack:
        current_sum, index = stack.pop()

        if index == len(arr):
            ret.add(current_sum)
        elif (current_sum, index) in stack:
            continue
        else:
            stack.add((current_sum + arr[index], index + 1))
            stack.add((current_sum, index + 1))

    return ret
