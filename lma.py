import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

def model_func(x, a, b, c):
    return a * np.sin(b * x) + c

def fun(params, x, y):
    return model_func(x, *params) - y

np.random.seed(0)
a = 1.5
b = 1.0
c = 0.5
x = np.linspace(0, 4, 50)
y = model_func(x, a, b, c) + 0.2 * np.random.randn(x.size)

params0 = np.array([2.0, 1.0, 0.0])
res = least_squares(fun, params0, args=(x, y), method='lm')

plt.style.use('dark_background')
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'o', label='data')
plt.plot(x, model_func(x, *res.x), label='fitted curve')
plt.legend()
plt.grid(True)
plt.savefig('lma_plot.png')
