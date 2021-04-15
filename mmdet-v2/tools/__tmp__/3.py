import collections


def equ(C1, C2):
    for k1, v1 in C1.items():
        if k1 not in C2 or C1[k1] != C2[k1]:
            return False
    return True


while True:
    try:
        n, m = list(map(int, input().split()))
        Ai = list(map(int, input().split()))
        Bi = list(map(int, input().split()))
        res = []
        for a in set(Ai):
            for b in set(Bi):
                x = b + m - a
                Ci = [(_ + x) % m for _ in Ai]
                C1 = dict(collections.Counter(Ci))
                C2 = dict(collections.Counter(Bi))
                if equ(C1, C2):
                    res.append(x)
        print(min(res))

    except EOFError:
        break
