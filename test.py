from functools import partial

class A:
    def __init__(self, a, b):
        self.a = a
        self.b = b

p = partial(A, a=1)

x = p(2)
print(x.__class__, x.a, x.b)