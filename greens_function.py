import scipy
import scipy.integrate as integrate
import mpmath
import numpy as np
import math
import cmath
import matplotlib.pyplot as plt
from scipy.misc import derivative

def k(t, gamma, x):
    k = 2 / (t - gamma * math.cos(x))
    return k

def generate_path(a, b):
    real_path = lambda x: np.real(a) + x * (np.real(b) - np.real(a))
    imag_path = lambda x: np.imag(a) + x * (np.imag(b) - np.imag(a))
    return real_path, imag_path

def complex_quadrature(func, a, b, **kwargs):
    real_path, imag_path = generate_path(a, b)
    def real_func(x):
        return scipy.real(func(x)) * derivative(real_path, x, dx=1e-6) - scipy.imag(func(x)) * derivative(imag_path, x, dx=1e-6)
    def imag_func(x):
        return scipy.imag(func(x)) * derivative(real_path, x, dx=1e-6) + scipy.real(func(x)) * derivative(imag_path, x, dx=1e-6)
    real_integral = integrate.quad(real_func, 0, 1, **kwargs)
    imag_integral = integrate.quad(imag_func, 0, 1, **kwargs)
    return (real_integral[0] + 1j*imag_integral[0])

epsilon = .000001j
gamma = 1
N = 101
t_points = np.linspace(0, 10, N, endpoint=True)
for t in t_points:
    s = t + epsilon
    #print(complex(mpmath.ellipk(cmath.sqrt(k(t, gamma, 1) ** 2 - 1) / k(t, gamma, 1))))
    #print((1 / math.pi) ** 2 * complex_quadrature(lambda x: complex(mpmath.ellipk(cmath.sqrt(k(t, gamma, x) ** 2 - 1) / k(t, gamma, x))), np.absolute(cmath.acos((s + 2)/ gamma)), np.absolute(cmath.acos((s - 2)/ gamma))))
    #print(cmath.acos((s - 2)/ gamma))
    #G_i = (1 / math.pi) ** 2 * complex_quadrature(lambda x: complex(mpmath.ellipk(cmath.sqrt(k(t, gamma, x) ** 2 - 1) / k(t, gamma, x))), cmath.acos((s + 2)/ gamma), cmath.acos((s - 2)/ gamma))
    G_r = (1 / math.pi) ** 2 * (-1 * complex_quadrature(lambda x: abs(k(t, gamma, x)) * complex(mpmath.ellipk(abs(k(t, gamma, x)))), 0, cmath.acos((s + 2)/ gamma)) - complex_quadrature(lambda x: complex(mpmath.ellipk(1 / abs(k(t, gamma, x)))), cmath.acos((s + 2)/ gamma), cmath.acos(s / gamma)) + complex_quadrature(lambda x: complex(mpmath.ellipk(1 / k(t, gamma, x))), cmath.acos(s / gamma), cmath.acos((s - 2)/ gamma)) + complex_quadrature(lambda x: k(t, gamma, x) * complex(mpmath.ellipk(k(t, gamma, x))), cmath.acos((s - 2)/ gamma), math.pi))
    print(t, np.real(G_r))
    plt.scatter(t, np.real(G_r), color="blue")
plt.show()