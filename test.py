import sympy as sp
import re
expr = sp.sympify("sin(x1) + x2*x3")
variables = expr.free_symbols
def sort_key(symbol):
    match = re.match(r"x(\d+)", str(symbol))
    return int(match.group(1)) if match else float('inf')

sorted_vars = sorted(variables, key=sort_key)

print([str(var) for var in sorted_vars])

evaluated = expr.subs({"x1":90, "x2":90, "x3":90}).evalf()
print("Evaluated:", evaluated)