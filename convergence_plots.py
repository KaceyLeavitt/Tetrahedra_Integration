import numpy as np
import tetrahedron_method
import matplotlib.pyplot as plt

def toy_energy1(k_point, number_of_bands):
    energy = np.dot(k_point, k_point)
    return energy

def toy_energy2(k_point, number_of_bands):
    energy = np.dot(k_point, k_point) ** .5
    return energy

def toy_energy3(k_point, number_of_bands):
    energy = np.dot(k_point, k_point) ** .25
    return energy

r_lattice_vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
V_G = 1
valence_electrons = 2
theoretical_E_Fermi = (3 * V_G * valence_electrons / (8 * np.pi)) ** (2 / 3)

for k in range(25):
    number_of_intervals = 5 + k
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

    theoretical_rho = np.sqrt(theoretical_E_Fermi)
    theoretical_integral_result = np.pi / 10 * theoretical_rho ** 5

    integral_error = abs(theoretical_integral_result -
                         calculated_integral_result)

    rho_for_calculated_E_Fermi = np.sqrt(calculated_E_Fermi)
    theoretical_integral_result_for_calculated_E_Fermi = \
        np.pi / 10 * rho_for_calculated_E_Fermi ** 5

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