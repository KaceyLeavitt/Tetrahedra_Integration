import numpy as np
import matplotlib.pyplot as plt

def toy_energy(k_point, number_of_bands):
    energy = np.dot(k_point, k_point)
    return energy

r_lattice_vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
V_G = 1
valence_electrons = 2
E_Fermi = (3 * V_G * valence_electrons / (8 * np.pi)) ** (2 / 3)
rho = np.sqrt(E_Fermi)
theoretical_integral_result = np.pi / 10 * rho ** 5

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
        energy = toy_energy(k_point, number_of_bands=1)
        if energy < E_Fermi:
            integral_result += energy * V_G / number_of_grid_points

    integral_error = abs(theoretical_integral_result - integral_result)

    plt.yscale('log')
    plt.xscale('log')
    plt.scatter(number_of_grid_points, integral_error, color='green')

plt.show()