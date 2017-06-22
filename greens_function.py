import scipy
import scipy.integrate as integrate
import mpmath
import numpy as np
import math
import cmath
import matplotlib.pyplot as plt

def k(t, gamma, x):
    k = 2 / (t - gamma * math.cos(x))
    return k

def complex_quadrature(func, a, b, **kwargs):
    def real_func(x):
        return scipy.real(func(x))
    def imag_func(x):
        return scipy.imag(func(x))
    real_integral = integrate.quad(real_func, a, b, **kwargs)
    imag_integral = integrate.quad(imag_func, a, b, **kwargs)
    return (real_integral[0] + 1j*imag_integral[0])

epsilon = .000001j
gamma = 1
N = 11
t_points = np.linspace(0, 10, N, endpoint=True)
for t in t_points:
    s = t + epsilon
    #print(complex(mpmath.ellipk(cmath.sqrt(k(t, gamma, 1) ** 2 - 1) / k(t, gamma, 1))))
    print(complex_quadrature(lambda x: complex(mpmath.ellipk(cmath.sqrt(k(t, gamma, x) ** 2 - 1) / k(t, gamma, x))), np.absolute(0.0+0.0j), np.absolute(0.0+2.0j)))
    #print(type(np.asscalar(np.absolute(cmath.acos((s - 2)/ gamma)))))
    #G_i = (1 / math.pi) ** 2 * complex_quadrature(lambda x: mpmath.ellipk(cmath.sqrt(k(t, gamma, x) ** 2 - 1) / k(t, gamma, x)), np.absolute(cmath.acos((s + 2)/ gamma)), np.absolute(cmath.acos((s - 2)/ gamma)))
    #print(t, G_i)
    #plt.scatter(t, G_i)
#plt.show