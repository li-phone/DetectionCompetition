while True:
    try:
        s1 = input().strip()
        s2 = input().strip()
        s3 = s1
        n = 0
        for i, c in enumerate(s2):
            if c not in s3:
                s3 += s1
            idx = s3.index(c)
            n += idx
            s3 = s3[idx + 1:]
        print(n)
    except EOFError:
        break
