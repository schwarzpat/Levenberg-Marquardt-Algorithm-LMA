import numpy as np
import matplotlib.pyplot as plt

# Define the function form
def model_func(x, a, b, c):
    return a * np.sin(b * x) + c

# Define the residuals
def fun(params, x, y):
    return model_func(x, *params) - y

# Define the Jacobian
def jacobian(params, x, y):
    J = np.empty((x.size, params.size))
    J[:, 0] = np.sin(params[1] * x)
    J[:, 1] = params[0] * x * np.cos(params[1] * x)
    J[:, 2] = 1.0
    return J

# Levenberg-Marquardt optimization function
def lm_optimize(func, jacobian, x0, x, y, max_iter=100, lam=0.001):
    for _ in range(max_iter):
        r = func(x0, x, y)
        J = jacobian(x0, x, y)
        A = J.T @ J + lam * np.eye(J.shape[1])
        g = J.T @ r
        step = np.linalg.solve(A, g)
        x_new = x0 - step
        if np.linalg.norm(x_new - x0) < 1e-8:  # Convergence check
            break
        x0 = x_new
    return x0

# Generate synthetic data
np.random.seed(0)
a = 1.5
b = 1.0
c = 0.5
x = np.linspace(0, 4, 50)
y = model_func(x, a, b, c) + 0.2 * np.random.randn(x.size)

# Initial guess
params0 = np.array([2.0, 1.0, 0.0])

# Run Levenberg-Marquardt optimization
res = lm_optimize(fun, jacobian, params0, x, y)

# Visualization
plt.style.use('dark_background')
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'o', label='data')
plt.plot(x, model_func(x, *res), label='fitted curve')
plt.legend()
plt.grid(True)
plt.show()
