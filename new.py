import scipy.special as special
import scipy.integrate as integrate
import numpy as np
import math
import matplotlib.pyplot as plt

def I(n):
    I = 0.
    
    if (n % 2 == 0):
        I = math.sqrt(math.pi) * special.gamma((n + 1.) / 2.) / special.gamma(n / 2. + 1.)
    else:
        I = 0.

    return I

def E(n):
    E = n / 2. - 1. / 4. * (1. - (-1.) ** n)
    #print("E", E)
    return E

def F(m, n):
    F = 0.

    if m < 0:
        F = 0
    elif np.isclose([round(n)], [n]):
        top = 1.
        top_multiplied_terms_count = 0

        while top_multiplied_terms_count < m:
            top *= (n - top_multiplied_terms_count)
            top_multiplied_terms_count += 1

        F = top / math.factorial(m)
    else:
        F = (-1.) ** m * special.gamma(m - n) / (math.factorial(m) * special.gamma(-n))

    #print("F", F, "m", m, "n", n)
    return F

def L(n, k):
    L = 2. ** (k - 1.) * I(k + n) + k * sum((-1.) ** i * 2. ** (k - 2. * i - 1.) * F(i - 1., k - i - 1.) * I(k + n - 2. * i) for i in range(1, int(round(E(k) + 1))))
    #print("L", L)
    return L

def J(n, k):
    J = 0.
    
    if k > n or ((k + n) % 2 == 1):
        J = 0
    elif k > 1:
        J = L(n, k)
    elif k == 1:
        J = I(n + 1)
    elif k == 0:
        J = I(n)

    #print("J", J, n, k)
    return J

def sum_0_to_i(i, j, l, m, n):
    sum_0_to_i = sum(F(k, i) * J(j + k, l) * J(j + i - k, m) * J(i, n) for k in range(i + 1))
    #print("sum_0_to_i", sum_0_to_i, isinstance(sum_0_to_i, np.float64), i, j)
    
    return sum_0_to_i
    
def sum_0_to_L(L, i, t, l, m, n):
    sum_0_to_L = sum((-1.) ** j * F(j, -1. - i) * t ** (-j) * sum_0_to_i(i, j, l, m, n) for j in range(L + 1))
    #print("sum_0_to_L", sum_0_to_L)
    return sum_0_to_L

def sum_0_to_N(N, L, t, l, m, n):
    sum_0_to_N = sum((-1.) ** i * F(i, -1.) * t ** (-1. - i) * sum_0_to_L(L, i, t, l, m, n) for i in range(N + 1))
    #print("sum_0_to_N", sum_0_to_N)
    return sum_0_to_N

def G(t, l, m, n):
    if t < 3.:
        print("Error: t < 3")
        print("t = ", t)
    else:
        L = 50
        N = 50
        
        G = 1. / math.pi ** 3. * sum_0_to_N(N, L, t, l, m, n)
        #print("G", G)
        return G

print(G(5, 0, 0, 0))
