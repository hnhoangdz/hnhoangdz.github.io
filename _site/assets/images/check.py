def change(arr, offset):
    if (offset * -1) % len(arr) == 0:
        return arr
    nega = False
    if offset < 0:
        nega = True
        offset *= -1
    offset %= len(arr)

    for i in range(offset):
        if nega == True:
            pos = arr[0]
            for j in range(1, len(arr)):
                arr[j-1] = arr[j]
            arr[-1] = pos
        else:
            pos = arr[-1]
            for j in range(len(arr)-1, 0,-1):
                arr[j] = arr[j-1]
            arr[0] = pos
    return arr

print(change([1,2,3,4,5],5))