import sympy

a,b,c = sympy.symbols("a b c")
M = sympy.Matrix([[-(a+b), a, 0, b, 0], [a, -(a+b), b, 0, 0], [0, b, -(2*b+c), c, b], [b, 0, c, -(2*b+c), b], [0, 0, b, b, -2*b]])

print(sympy.exp(M))