from math import exp

x = 1-4/3 * exp(-2)
y = 1-5/4 * exp(-2)
print("exp(-2)", exp(-2))
print("P(X>2)", x)
print("P(Y>2)", y)
print("P(Z>2)", 5/3*exp(-4))