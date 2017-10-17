import scipy
import scipy.integrate as integrate
import mpmath
import numpy as np
import math
import cmath
import matplotlib.pyplot as plt
from scipy.misc import derivative

"""Solves the integral of the Lattice Green's Function for FCC, BCC, 
and simple cubic lattices. Based on the work in the paper "Lattice 
Green's Functions for the Cubic Lattices in Terms of the Complete 
Elliptic Integral" by Tohru Morita and Tsuyoshi and Tsuyoshi 
Horiguchi found in Journal of Mathematical Physics 12, 981 (1971); 
doi: 10.1063/1.1665692"""

def k_fcc(t, x):
    """Calculates the value of the complex modulus k in the 
    paper."""

    k = 2 * complex(t + math.cos(x) ** 2) ** .5 / (t + 1)
    
    return k


def k_bcc(t, x):
    """Calculates the value of the complex modulus k in the 
    paper."""

    k = math.cos(x) / t
    
    return k

def k_simple(t, gamma, x):
    """Calculates the value of the complex modulus k in the 
    paper."""

    k = 2 / (t - gamma * math.cos(x))
    
    return k

def generate_path(a, b):
    """Generates the most direct path in the complex plane between 
    points a and b."""

    real_path = lambda x: np.real(a) + x * (np.real(b) - np.real(a))
    imag_path = lambda x: np.imag(a) + x * (np.imag(b) - np.imag(a))

    return real_path, imag_path

def complex_quadrature(func, a, b, **kwargs):
    """Performs Gaussian quadrature for the input function func 
    along the most direct path in the complex plane between points 
    a and b."""

    real_path, imag_path = generate_path(a, b)

    def real_func(x):
        """Returns the function to be integrated to produce the 
        real portion of the complex integral."""
        
        return scipy.real(func(x)) * \
               derivative(real_path, x, dx=1e-6) - \
               scipy.imag(func(x)) * derivative(imag_path, x,
                                                dx=1e-6)

    def imag_func(x):
        """Returns the function to be integrated to produce the 
        imaginary portion of the complex integral."""

        return scipy.imag(func(x)) * \
               derivative(real_path, x, dx=1e-6) + \
               scipy.real(func(x)) * derivative(imag_path, x,
                                                dx=1e-6)

    real_integral = integrate.quad(real_func, 0, 1, **kwargs)
    imag_integral = integrate.quad(imag_func, 0, 1, **kwargs)

    return (real_integral[0] + 1j*imag_integral[0])

def density_of_states_fcc(s):
    density_of_states = 0
    
    if s <= -1 or s > 3:
        raise ValueError("Error: s must be a value between -1 and 3")
    elif s < 0:
        density_of_states = 4. / (math.pi ** 2. * (s + 1.)) * \
                                (integrate.quad(lambda x:
                                np.real(1. / k_fcc(s, x) * \
                                complex(mpmath.ellipk((
                                k_fcc(s, x) ** 2 - 1.) ** .5 / \
                                k_fcc(s, x)))), 0.,
                                math.acos((1. - s) / 2.))[0] + 2 * \
                                integrate.quad(lambda x:
                                np.real(complex(mpmath.ellipk((1 - \
                                k_fcc(s, x) ** 2) ** .5))),
                                math.acos((1. - s) / 2.),
                                math.acos((-1 * s) ** .5))[0] + 2 * \
                                integrate.quad(lambda x:
                                np.real(1. / (1. - k_fcc(s, x) ** \
                                2) ** .5 * complex(mpmath.ellipk(1. / \
                                (1 - k_fcc(s, x) ** 2) ** .5))),
                                math.acos((-1 * s) ** .5), math.pi / \
                                2)[0])
    elif s < 1:
        density_of_states = 4. / (math.pi ** 2. * (s + 1.)) * (integrate.quad(lambda x: np.real(1. / k_fcc(s, x) * complex(mpmath.ellipk((k_fcc(s, x) ** 2 - 1.) ** .5 / k_fcc(s, x)))), 0., math.acos((1. - s) / 2.))[0] + 2 * integrate.quad(lambda x: np.real(complex(mpmath.ellipk((1 - k_fcc(s, x) ** 2) ** .5))), math.acos((1. - s) / 2.), math.pi / 2.)[0])
    else:
        density_of_states = 4. / (math.pi ** 2. * (s + 1.)) * integrate.quad(lambda x: np.real(1. / k_fcc(s, x) * complex(mpmath.ellipk((k_fcc(s, x) ** 2 - 1) ** .5 / k_fcc(s, x)))), 0, math.acos((s - 1.) / 2.))[0]

    return density_of_states

# def number_of_states_fcc(s):
#     number_of_states = 0

#     if s <= -1 or s > 3:
#         raise ValueError("Error: s must be a value between -1 and 3")
#     else:
#         #number_of_states = integrate.quad(lambda x: density_of_states_fcc(x), -.7, 0) + \
#         number_of_states = integrate.quad(lambda x: density_of_states_fcc(x), .05, s)

#     return number_of_states

def plot_density_of_states_fcc():
    plotting_range = np.linspace(-0.95, 2.9, 100, endpoint=True)

    for s in plotting_range:
        plt.scatter(s, density_of_states_fcc(s), c='b', marker='.')

    plt.show()
    
# def plot_number_of_states_fcc():
#     plotting_range = np.linspace(.1, 2.9, 15, endpoint=True)
    
#     for s in plotting_range:
#         plt.scatter(s, number_of_states_fcc(s)[0])

#     plt.show()


def density_of_states_bcc(s):
    density_of_states = 0
    
    if s <= 0 or s > 1:
        raise ValueError("s must be between 0 and 1.")
    else:
        density_of_states = 4 / (math.pi ** 2 * s) * integrate.quad(lambda x: 1 / k_bcc(s, x) * mpmath.ellipk((k_bcc(s, x) ** 2 - 1) ** 0.5 / k_bcc(s, x)), 0, math.acos(s))[0]

    return density_of_states

def plot_density_of_states_bcc():
    plotting_range = np.linspace(0.1, 1, 100, endpoint=True)

    for s in plotting_range:
        plt.scatter(s, density_of_states_bcc(s), c='b', marker='.')

    plt.show()

def density_of_states_simple(s):
    density_of_states = 0
    gamma = 1

    if s > 1 and s < 3:
        density_of_states = (1 / math.pi) ** 2 * integrate.quad(lambda x: mpmath.ellipk((k_simple(s, gamma, x) ** 2 - 1) ** 0.5 / k_simple(s, gamma, x)), 0, math.acos((s - 2) / gamma))[0]
    elif s > 0 and s < 1:
        density_of_states = (1 / math.pi) ** 2 * integrate.quad(lambda x: mpmath.ellipk(((k_simple(s, gamma, x) ** 2 - 1) ** 0.5) / k_simple(s, gamma, x)), 0, math.pi, limit=10000, points=[math.acos(s / gamma)])[0]

    return density_of_states

def plot_density_of_states_simple():
    plotting_range = np.linspace(0.1, 2.9, 100, endpoint=True)

    for s in plotting_range:
        plt.scatter(s, density_of_states_simple(s), c='b', marker='.')

    plt.show()
    




epsilon = .000001j
gamma = 1
N = 11
# t_points = np.linspace(0, 10, N, endpoint=True)
s_3_to_10 = np.linspace(3, 10, N, endpoint=True)
s_1_to_3 = np.linspace(1, 3, N, endpoint=False)
s_0_to_1 = np.linspace(0, 1, N, endpoint=True)
s_3_to_6 = np.linspace(3, 6, N, endpoint=True)
s_neg_1_to_0 = np.linspace(-.99, 0, N, endpoint=True)

# for s in s_3_to_10:
#     #print(complex(mpmath.ellipk(cmath.sqrt(k(t, gamma, 1) ** 2 - 1) / k(t, gamma, 1))))
#     #print((1 / math.pi) ** 2 * complex_quadrature(lambda x: complex(mpmath.ellipk(cmath.sqrt(k(t, gamma, x) ** 2 - 1) / k(t, gamma, x))), np.absolute(cmath.acos((s + 2)/ gamma)), np.absolute(cmath.acos((s - 2)/ gamma))))
#     #print(cmath.acos((s - 2)/ gamma))
#     #G_i = (1 / math.pi) ** 2 * complex_quadrature(lambda x: complex(mpmath.ellipk(cmath.sqrt(k(t, gamma, x) ** 2 - 1) / k(t, gamma, x))), cmath.acos((s + 2)/ gamma), cmath.acos((s - 2)/ gamma))
#     # G_r = (1 / math.pi) ** 2 * (-1 * complex_quadrature(lambda x: abs(k(t, gamma, x)) * complex(mpmath.ellipk(abs(k(t, gamma, x)))), 0, cmath.acos((s + 2)/ gamma)) - complex_quadrature(lambda x: complex(mpmath.ellipk(1 / abs(k(t, gamma, x)))), cmath.acos((s + 2)/ gamma), cmath.acos(s / gamma)) + complex_quadrature(lambda x: complex(mpmath.ellipk(1 / k(t, gamma, x))), cmath.acos(s / gamma), cmath.acos((s - 2)/ gamma)) + complex_quadrature(lambda x: k(t, gamma, x) * complex(mpmath.ellipk(k(t, gamma, x))), cmath.acos((s - 2)/ gamma), math.pi))
#     #print(integrate.quad(lambda x: (k(s, gamma, x) * mpmath.ellipk(k(s, gamma, x))), 0, math.pi))
#     G_r = (1 / math.pi) ** 2 * integrate.quad(lambda x: (k(s, gamma, x) * mpmath.ellipk(k(s, gamma, x))), 0, math.pi)[0]
#     G_i = 0
#
#     #print(s, np.real(G_r))
#     plt.scatter(s, np.real(G_r), color="blue")
#     plt.scatter(s, np.real(G_i), color="red")

# for s in s_1_to_3:
#     G_r = (1 / math.pi) ** 2 * integrate.quad(lambda x: mpmath.ellipk(1 / k(s, gamma, x)), 0, math.acos((s - 2) / gamma))[0] + (1 / math.pi) ** 2 * integrate.quad(lambda x: (k(s, gamma, x) * mpmath.ellipk(k(s, gamma, x))), math.acos((s - 2) / gamma), math.pi)[0]
#     G_i = (1 / math.pi) ** 2 * integrate.quad(lambda x: mpmath.ellipk((k(s, gamma, x) ** 2 - 1) ** 0.5 / k(s, gamma, x)), 0, math.acos((s - 2) / gamma))[0]
#
#     plt.scatter(s, np.real(G_r), color="blue")
#     plt.scatter(s, np.real(G_i), color="red")

# for s in s_0_to_1:
#     # G_r = -((1 / math.pi) ** 2) * integrate.quad(lambda x: mpmath.ellipk(1 / abs(k(s, gamma, x))), 0, math.acos(s / gamma))[0] + (1 / math.pi) ** 2 * integrate.quad(lambda x: mpmath.ellipk(1 / k(s, gamma, x)), math.acos(s / gamma), math.pi)[0]
#     G_i = (1 / math.pi) ** 2 * integrate.quad(lambda x: mpmath.ellipk(((k(s, gamma, x) ** 2 - 1) ** 0.5) / k(s, gamma, x)), 0, math.pi, limit=10000, points=[math.acos(s / gamma)])[0]

#     # plt.scatter(s, np.real(G_r), color="blue")
#     plt.scatter(s, np.real(G_i), color="red")

# plt.axis([0, 7, 0, 2])
# plt.show()

# for s in s_0_to_1:
#     t = s - epsilon
#     G_r = 4 / (math.pi ** 2 * s) * integrate.quad(lambda x: 1 / k(s, gamma, x) * mpmath.ellipk(1 / k(s, gamma, x)), 0, math.acos(s))[0] + 4 / (math.pi ** 2 * s) * integrate.quad(lambda x: mpmath.ellipk(k(s, gamma, x)), math.acos(s), math.pi / 2)[0]
#     G_i = 4 / (math.pi ** 2 * s) * integrate.quad(lambda x: 1 / k(s, gamma, x) * mpmath.ellipk((k(s, gamma, x) ** 2 - 1) ** 0.5 / k(s, gamma, x)), 0, math.acos(s))[0]
#
#     plt.scatter(s, np.real(G_r), color="blue")
#     plt.scatter(s, np.real(G_i), color="red")
#
# for s in s_1_to_3:
#     #print(4 / (math.pi ** 2 * s) * integrate.quad(lambda x: mpmath.ellipk(k(s, gamma, x)), 0, math.pi / 2)[0])
#     G_r = 4 / (math.pi ** 2 * s) * integrate.quad(lambda x: mpmath.ellipk(k(s, gamma, x)), 0, math.pi / 2)[0]
#     G_i = 0
#
#     plt.scatter(s, np.real(G_r), color="blue")
#     plt.scatter(s, np.real(G_i), color="red")
#
# plt.axis([0, 2, 0, 2.8])
# plt.show()

# for s in s_3_to_6:
#     G_r = 4 / (math.pi ** 2 * (s + 1)) * integrate.quad(lambda x: mpmath.ellipk(k(s, gamma, x)), 0, math.pi / 2)[0]
#     G_i = 0
#
#     plt.scatter(s, np.real(G_r), color="blue")
#     plt.scatter(s, np.real(G_i), color="red")

# for s in s_1_to_3:
#     #G_r = 4 / (math.pi ** 2 * (s + 1)) * complex_quadrature(lambda x: 1 / k(s, gamma, x) * mpmath.ellipk(k(s, gamma, x)), 0, cmath.acos((s - 1) / 2))[0] #+ 4 / (math.pi ** 2 * (s + 1)) * integrate.quad(lambda x: mpmath.ellipk(float(k(s, gamma, x))), math.acos((s - 1) / 2), math.pi / 2)[0]
#     G_i = 4. / (math.pi ** 2. * (s + 1.)) * integrate.quad(lambda x: np.real(1. / k(s, gamma, x) * complex(mpmath.ellipk((k(s, gamma, x) ** 2 - 1) ** .5 / k(s, gamma, x)))), 0, math.acos((s - 1.) / 2.))[0]
#     #print(k(s, gamma, 0))

#     #plt.scatter(s, np.real(G_r), color="blue")
#     plt.scatter(s, np.real(G_i), color="red")

# for s in s_0_to_1:
#     #G_r = 4. / (math.pi ** 2. * (s + 1.)) * (integrate.quad(lambda x: np.real(1. / k(s, gamma, x) * complex(mpmath.ellipk(k(s, gamma, x)))), 0., math.acos((s - 1.) / 2.))[0] + integrate.quad(lambda x: np.real(complex(mpmath.ellipk(k(s, gamma, x)))), math.acos((1. - s) / 2.), math.pi / 2.)[0])
#     G_i = 4. / (math.pi ** 2. * (s + 1.)) * (integrate.quad(lambda x: np.real(1. / k(s, gamma, x) * complex(mpmath.ellipk((k(s, gamma, x) ** 2 - 1.) ** .5 / k(s, gamma, x)))), 0., math.acos((1. - s) / 2.))[0] + 2 * integrate.quad(lambda x: np.real(complex(mpmath.ellipk((1 - k(s, gamma, x) ** 2) ** .5))), math.acos((1. - s) / 2.), math.pi / 2.)[0])

#     #print(1 / k(s, gamma, 0.1) * mpmath.ellipk(k(s, gamma, 0.1)))

#     #plt.scatter(s, np.real(G_r), color="blue")
#     plt.scatter(s, np.real(G_i), color="red")

# for s in s_neg_1_to_0:
#     G_i = 4. / (math.pi ** 2. * (s + 1.)) * (integrate.quad(lambda x: np.real(1. / k(s, gamma, x) * complex(mpmath.ellipk((k(s, gamma, x) ** 2 - 1.) ** .5 / k(s, gamma, x)))), 0., math.acos((1. - s) / 2.))[0] + 2 * integrate.quad(lambda x: np.real(complex(mpmath.ellipk((1 - k(s, gamma, x) ** 2) ** .5))), math.acos((1. - s) / 2.), math.acos((-1 * s) ** .5))[0] + 2 * integrate.quad(lambda x: np.real(1. / (1. - k(s, gamma, x) ** 2) ** .5 * complex(mpmath.ellipk(1. / (1 - k(s, gamma, x) ** 2) ** .5))), math.acos((-1 * s) ** .5), math.pi / 2)[0])

#     #print(k(s, gamma, 1.5)) #* complex(mpmath.ellipk(1. / (1 - k(s, gamma, 1.5) ** 2) ** .5))))
#     #print(complex(s + math.cos(1.5) ** 2))
#     plt.scatter(s, np.real(G_i), color="red")

# plt.axis([-4, 6, -.8, 2.8])
# plt.show()
