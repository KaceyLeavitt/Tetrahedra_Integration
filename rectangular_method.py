import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.integrate as integrate

def toy_energy1(k_point, number_of_bands):
    energy = np.dot(k_point, k_point)
    return energy

def theoretical_E_Fermi1(V_G, valence_electrons):
    E_Fermi = (3 * V_G * valence_electrons / (8 * np.pi)) ** (2 / 3)
    return E_Fermi

def theoretical_integral_result1(E_Fermi):
    rho = np.sqrt(E_Fermi)
    integral_result = np.pi / 10 * rho ** 5
    return integral_result

def toy_energy2(k_point, number_of_bands):
    energy = np.sqrt(np.dot(k_point, k_point))
    return energy

def theoretical_E_Fermi2(V_G, valence_electrons):
    E_Fermi = (3 * V_G * valence_electrons / (8 * np.pi)) ** (1 / 3)
    return E_Fermi

def theoretical_integral_result2(E_Fermi):
    rho = E_Fermi
    integral_result = np.pi / 8 * rho ** 4
    return integral_result

def toy_energy3(k_point, number_of_bands):
    energy = np.dot(k_point, k_point) ** .25
    return energy

def theoretical_E_Fermi3(V_G, valence_electrons):
    E_Fermi = (3 * V_G * valence_electrons / (8 * np.pi)) ** (.5 / 3)
    return E_Fermi

def theoretical_integral_result3(E_Fermi):
    rho = E_Fermi ** 2
    integral_result = np.pi / 7 * rho ** 3.5
    return integral_result

def toy_energy4(k_point, number_of_bands):
    if np.dot(k_point, k_point) < 1:
        energy = .5 * math.exp(math.cos(2 * math.pi * np.dot(k_point, k_point) ** .5))
    else:
        energy = .5 * math.exp(math.cos(2 * math.pi * np.dot(k_point, k_point) ** .5))
    return energy

def theoretical_E_Fermi4(V_G, valence_electrons):
    E_Fermi = math.exp(math.cos((3 * math.pi ** 2 * V_G * valence_electrons
                                 + math.pi ** 3) ** (1 / 3)))
    return E_Fermi

def theoretical_integral_result4(E_Fermi):
    left_bound = math.acos(math.log(E_Fermi)) / (2 * math.pi)
    right_bound = 1 - left_bound
    integral_result = integrate.quad(lambda x:
        math.exp(math.cos(2 * math.pi * x)), left_bound, right_bound)[0]
    return integral_result

def toy_energy5(k_point, number_of_bands):
    energy = 1 - k_point[0]
    return energy

def theoretical_E_Fermi5(V_G, valence_electrons):
    E_Fermi = .5
    return E_Fermi

def theoretical_integral_result5(E_Fermi):
    integral_result = .5 * E_Fermi ** 2
    return integral_result

r_lattice_vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
V_G = 1
valence_electrons = 2
E_Fermi = theoretical_E_Fermi4(V_G, valence_electrons)
print('e', E_Fermi)
theoretical_integral_result = theoretical_integral_result4(E_Fermi)
print('t', theoretical_integral_result)

for k in range(35):
    number_of_intervals = 6 + k
    number_of_grid_points = (number_of_intervals) ** 3
    grid_vecs = np.array([[(1 - .000001) / number_of_intervals, 0, 0],
                          [0, (1 - .000001) / number_of_intervals, 0],
                          [0, 0, (1 - .000001) / number_of_intervals]])
    b1 = grid_vecs[:,0]
    b2 = grid_vecs[:,1]
    b3 = grid_vecs[:,2]
    offset = (b1 + b2 + b3) / 2
    grid = []

    for m in range(number_of_intervals):
        for n in range(number_of_intervals):
            for l in range(number_of_intervals):
                grid.append((b1 * m + b2 * n + b3 * l + offset).tolist())

    integral_result = 0

    for k_point in grid:
        energy = toy_energy4(k_point, number_of_bands=1)
        if energy < E_Fermi:
            integral_result += energy * V_G / number_of_grid_points
    #if number_of_intervals == 8:
    #    print(8, theoretical_integral_result - integral_result)
    #if number_of_intervals == 11:
    #    print(11, theoretical_integral_result - integral_result)
    print(theoretical_integral_result - integral_result)
    integral_error = abs(theoretical_integral_result - integral_result)
    if k == 12:
        print(integral_result)

    plt.yscale('log')
    plt.xscale('log')
    plt.scatter(number_of_grid_points, integral_error, color='green')

plt.show()