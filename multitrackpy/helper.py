def make_slices(a, n):
    k, m = divmod(a, n)
    return ((i * k + min(i, m), (i + 1) * k + min(i + 1, m)) for i in range(n))
