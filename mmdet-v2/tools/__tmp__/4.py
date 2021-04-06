def max_arr(arr, m):
    idx = 0
    max_val = 0
    for i in range(len(arr)):
        a = arr[i:i + m]
        if sum(a) > max_val:
            max_val = sum(a)
            idx = i
    return idx


while True:
    try:
        n, m, C = list(map(int, input().split()))
        Ai = list(map(int, input().split()))
        res = 0
        while len(Ai) != 0:
            idx = max_arr(Ai, m)
            arr = Ai[idx:idx + m]
            Ai = Ai[:idx] + Ai[idx + m:]
            res += max(arr) / C
        print(int(res+0.5))
    except EOFError:
        break
