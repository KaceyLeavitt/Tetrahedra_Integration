import numpy as np
import math
import tetrahedron_method
import matplotlib.pyplot as plt
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
        energy = math.exp(math.cos(2 * math.pi * np.dot(k_point, k_point) ** .5))
    else:
        energy = math.exp(math.cos(2 * math.pi * np.dot(k_point, k_point) ** .5))
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

r_lattice_vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
V_G = 1
valence_electrons = 2
theoretical_E_Fermi = theoretical_E_Fermi1(V_G, valence_electrons)
print(theoretical_E_Fermi)

for k in range(1):
    number_of_intervals = int(25 + k ** 1.313)
    number_of_grid_points = (number_of_intervals + 1) ** 3
    grid_vecs = np.array([[(1 - .000001) / number_of_intervals, 0, 0],
                          [0, (1 - .000001) / number_of_intervals, 0],
                          [0, 0, (1 - .000001) / number_of_intervals]])
    grid = []

    for m in range(number_of_intervals + 1):
        for n in range(number_of_intervals + 1):
            for l in range(number_of_intervals + 1):
                grid.append((grid_vecs[:,0] * m + grid_vecs[:,1] * n +
                             grid_vecs[:,2] * l).tolist())

    offset = np.array([0, 0, 0])
    apply_weight_correction = True

    (calculated_E_Fermi, calculated_integral_result) = \
        tetrahedron_method.integrate(r_lattice_vectors, grid_vecs, grid,
                                     toy_energy1, valence_electrons, offset,
                                     apply_weight_correction)

    E_Fermi_error = abs(theoretical_E_Fermi - calculated_E_Fermi)

    theoretical_integral_result = \
        theoretical_integral_result1(theoretical_E_Fermi)

    integral_error = abs(theoretical_integral_result -
                         calculated_integral_result)
    print(theoretical_integral_result - calculated_integral_result)
    theoretical_integral_result_for_calculated_E_Fermi = \
        theoretical_integral_result1(calculated_E_Fermi)

    integral_error_for_calculated_E_Fermi = \
        abs(theoretical_integral_result_for_calculated_E_Fermi -
            calculated_integral_result)

    plt.yscale('log')
    plt.xscale('log')
    plt.scatter(number_of_grid_points, E_Fermi_error, color='red')
    plt.scatter(number_of_grid_points, integral_error, color='blue')
    plt.scatter(number_of_grid_points, integral_error_for_calculated_E_Fermi,
                color='green')

    #print(E_Fermi_error, integral_error, integral_error_for_calculated_E_Fermi)

plt.show()