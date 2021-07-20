def make_slices(a, n):
    k, m = divmod(a, n)
    return ((i*k+min(i, m),(i+1)*k+min(i+1, m)) for i in range(n))

def dict_copyreplace(d1,d2):
    d = d1.copy()
    for key, value in d2.items():
        d[key] = value
        
    return d
