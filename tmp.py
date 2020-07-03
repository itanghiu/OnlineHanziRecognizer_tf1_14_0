




def test(a, n=2):
    out = a
    for k in range(1, n):
        out += a
    return out 

print(test('a'))
print(test([1, 2], 1))
print(test(n=10, a=3) ==30)
print(test() == None )