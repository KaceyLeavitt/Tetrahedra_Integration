import tetrahedron_method
import numpy as np
import math

# Tests for generate_r_lattice_vectors.
test_r_lattice_vectors = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# Reciprocal lattice vectors to be used in the testing.

def test_generate_r_lattice_vectors1():
    # Checking if the first reciprocal lattice vector is correctly determined.
    assert np.array_equal(tetrahedron_method.generate_r_lattice_vectors(
        test_r_lattice_vectors)[0], [1, 4, 7])

def test_generate_r_lattice_vectors2():
    # Checking if the second reciprocal lattice vector is correctly determined.
    assert np.array_equal(tetrahedron_method.generate_r_lattice_vectors(
        test_r_lattice_vectors)[1], [2, 5, 8])

def test_generate_r_lattice_vectors3():
    # Checking if the third reciprocal lattice vector is correctly determined.
    assert np.array_equal(tetrahedron_method.generate_r_lattice_vectors(
        test_r_lattice_vectors)[2], [3, 6, 9])


# Tests for generate_submesh_lattice_vectors.
test_grid_vecs = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# Grid vectors used for testing.

def test_generate_submesh_lattice_vectors1():
    # Checking if the first submesh lattice vector is correctly determined.
    assert np.array_equal(tetrahedron_method.generate_submesh_lattice_vectors(
        test_grid_vecs)[0], [1, 4, 7])

def test_generate_submesh_lattice_vectors2():
    # Checking if the second submesh lattice vector is correctly determined.
    assert np.array_equal(tetrahedron_method.generate_submesh_lattice_vectors(
        test_grid_vecs)[1], [2, 5, 8])

def test_generate_submesh_lattice_vectors3():
    # Checking if the third submesh lattice vector is correctly determined.
    assert np.array_equal(tetrahedron_method.generate_submesh_lattice_vectors(
        test_grid_vecs)[2], [3, 6, 9])


# Tests for generate_diagonal_vectors.
test_B1_1 = np.array([1, 2, 3])
# First submesh lattice vector used for testing.
test_B2_1 = np.array([4, 5, 6])
# Second submesh lattice vector used for testing.
test_B3_1 = np.array([7, 8, 9])
# Third submesh lattice vector used for testing.

def test_generate_diagonal_vectors1():
    # Checking if the first diagonal vector is correctly determined.
    assert np.array_equal(tetrahedron_method.generate_diagonal_vectors(
        test_B1_1, test_B2_1, test_B3_1)[0], [12, 15, 18])

def test_generate_diagonal_vectors2():
    # Checking if the second diagonal vector is correctly determined.
    assert np.array_equal(tetrahedron_method.generate_diagonal_vectors(
        test_B1_1, test_B2_1, test_B3_1)[1], [10, 11, 12])

def test_generate_diagonal_vectors3():
    # Checking if the third diagonal vector is correctly determined.
    assert np.array_equal(tetrahedron_method.generate_diagonal_vectors(
        test_B1_1, test_B2_1, test_B3_1)[2], [4, 5, 6])

def test_generate_diagonal_vectors4():
    # Checking if the fourth diagonal vector is correctly determined.
    assert np.array_equal(tetrahedron_method.generate_diagonal_vectors(
        test_B1_1, test_B2_1, test_B3_1)[3], [-2, -1, 0])


# Tests for calculate_diagonal_length.
test_diagonal1 = np.array([1, 2, 3])
# first diagonal vector used for testing.
test_diagonal2 = np.array([4, 5, 6])
# second diagonal vector used for testing.
test_diagonal3 = np.array([-7, -8, 9])
# third diagonal vector used for testing.
test_diagonal4 = np.array([0, .7, -.3])
# fourth diagonal vector used for testing.

def test_calculate_diagonal_length1():
    """Checking if the magnitude of the vector diagonal1 is correctly 
    determined."""
    assert tetrahedron_method.calculate_diagonal_length(test_diagonal1,
        test_diagonal2, test_diagonal3, test_diagonal4)[0] == math.sqrt(
        1 ** 2 + 2 ** 2 + 3 ** 2)

def test_calculate_diagonal_length2():
    """Checking if the magnitude of the vector diagonal2 is correctly 
    determined."""
    assert tetrahedron_method.calculate_diagonal_length(test_diagonal1,
        test_diagonal2, test_diagonal3, test_diagonal4)[1] == math.sqrt(
        4 ** 2 + 5 ** 2 + 6 ** 2)

def test_calculate_diagonal_length3():
    """Checking if the magnitude of the vector diagonal3 is correctly 
    determined."""
    assert tetrahedron_method.calculate_diagonal_length(test_diagonal1,
        test_diagonal2, test_diagonal3, test_diagonal4)[2] == math.sqrt(
        (-7) ** 2 + (-8) ** 2 + 9 ** 2)

def test_calculate_diagonal_length4():
    """Checking if the magnitude of the vector diagonal4 is correctly 
    determined."""
    assert tetrahedron_method.calculate_diagonal_length(test_diagonal1,
        test_diagonal2, test_diagonal3, test_diagonal4)[3] == math.sqrt(
        0 ** 2 + .7 ** 2 + (-.3) ** 2)


# Tests for determine_shortest_diagonal.
test_diagonal1_length = 2.8
# length of first diagonal used for testing.
test_diagonal2_length = .5
# length of first diagonal used for testing.
test_diagonal3_length = 1
# length of first diagonal used for testing.
test_diagonal4_length = 7
# length of first diagonal used for testing.

def test_determine_shortest_diagonal():
    # Checking if the index of the shortest diagonal is correctly determined.
    assert tetrahedron_method.determine_shortest_diagonal(
        test_diagonal1_length, test_diagonal2_length, test_diagonal3_length,
        test_diagonal4_length) == 2


# Tests for determine_parallelepiped_corners.
test_point1 = np.array([.1, -.2, .3])
# coordinates of first corner of parallelepiped used for testing.
test_B1_2 = np.array([1, 2, 3])
# First submesh lattice vector used for testing.
test_B2_2 = np.array([4, 5, -6])
# Second submesh lattice vector used for testing.
test_B3_2 = np.array([7, 8, 9])
# Third submesh lattice vector used for testing.

def test_determine_parallelepiped_corners2():
    """Checking if the second corner of the parallelepiped is correctly 
    determined."""
    assert np.array_equal(tetrahedron_method.determine_parallelepiped_corners(test_point1,
        test_B1_2, test_B2_2, test_B3_2)[0], [7.1, 7.8, 9.3])

def test_determine_parallelepiped_corners3():
    """Checking if the third corner of the parallelepiped is correctly 
    determined."""
    assert np.array_equal(tetrahedron_method.determine_parallelepiped_corners(
        test_point1, test_B1_2, test_B2_2, test_B3_2)[1], [4.1, 4.8, -5.7])

def test_determine_parallelepiped_corners4():
    """Checking if the fourth corner of the parallelepiped is correctly 
    determined."""
    assert np.array_equal(tetrahedron_method.determine_parallelepiped_corners(
        test_point1, test_B1_2, test_B2_2, test_B3_2)[2], [1.1, 1.8, 3.3])

def test_determine_parallelepiped_corners5():
    """Checking if the fifth corner of the parallelepiped is correctly 
    determined."""
    assert np.allclose(tetrahedron_method.determine_parallelepiped_corners(
        test_point1, test_B1_2, test_B2_2, test_B3_2)[3], [11.1, 12.8, 3.3])

def test_determine_parallelepiped_corners6():
    """Checking if the sixth corner of the parallelepiped is correctly 
    determined."""
    assert np.array_equal(tetrahedron_method.determine_parallelepiped_corners(
        test_point1, test_B1_2, test_B2_2, test_B3_2)[4], [8.1, 9.8, 12.3])

def test_determine_parallelepiped_corners7():
    """Checking if the seventh corner of the parallelepiped is correctly 
    determined."""
    assert np.array_equal(tetrahedron_method.determine_parallelepiped_corners(
        test_point1, test_B1_2, test_B2_2, test_B3_2)[5], [5.1, 6.8, -2.7])

def test_determine_parallelepiped_corners8():
    """Checking if the eighth corner of the parallelepiped is correctly 
    determined."""
    assert np.allclose(tetrahedron_method.determine_parallelepiped_corners(
        test_point1, test_B1_2, test_B2_2, test_B3_2)[6], [12.1, 14.8, 6.3])


# Tests for add_tetrahedra.
test_tetrahedra_quadruples_1 = []
# list of lists of indices of corners of tetrahedra used for testing.
test_point_indices = np.array([1, 2, 3, 4, 5, 6, 7, 8])
# indices of corners of parralelepiped used for testing.

def test_add_tetrahedra1():
    # Checking if the tetrahedra quadruples are correctly generated.
    test_shortest_diagonal1 = 1
    # the index of the shortest diagonal used for testing.
    assert tetrahedron_method.add_tetrahedra(test_tetrahedra_quadruples_1,
        test_shortest_diagonal1, test_point_indices) == [[1, 4, 7, 8],
        [1, 3, 7, 8], [1, 2, 5, 8], [1, 2, 6, 8], [1, 4, 6, 8], [1, 3, 5, 8]]

def test_add_tetrahedra2():
    # Checking if the tetrahedra quadruples are correctly generated.
    test_shortest_diagonal2 = 2
    # the index of the shortest diagonal used for testing.
    assert tetrahedron_method.add_tetrahedra(test_tetrahedra_quadruples_1,
        test_shortest_diagonal2, test_point_indices) == [[4, 6, 2, 5],
        [4, 6, 8, 5], [4, 1, 3, 5], [4, 7, 3, 5], [4, 7, 8, 5], [4, 1, 2, 5]]

def test_add_tetrahedra3():
    # Checking if the tetrahedra quadruples are correctly generated.
    test_shortest_diagonal3 = 3
    # the index of the shortest diagonal used for testing.
    assert tetrahedron_method.add_tetrahedra(test_tetrahedra_quadruples_1,
        test_shortest_diagonal3, test_point_indices) == [[3, 1, 4, 6],
        [3, 7, 4, 6], [3, 1, 2, 6], [3, 7, 8, 6], [3, 5, 2, 6], [3, 5, 8, 6]]

def test_add_tetrahedra4():
    # Checking if the tetrahedra quadruples are correctly generated.
    test_shortest_diagonal4 = 4
    # the index of the shortest diagonal used for testing.
    assert tetrahedron_method.add_tetrahedra(test_tetrahedra_quadruples_1,
        test_shortest_diagonal4, test_point_indices) == [[7, 8, 6, 2],
        [7, 8, 5, 2], [7, 4, 6, 2], [7, 3, 5, 2], [7, 1, 4, 2], [7, 1, 3, 2]]


# Tests for generate_tetrahedra
test_grid = [[0, 0, 0], [.5, 0, 1], [-.5, 1, 0], [1, 0, 0], [0, 1, 1],
             [1.5, 0, 1], [.5, 1, 0],  [1, 1, 1]]
# a grid of parallelepiped corner points used for testing.
test_B1 = np.array([1, 0, 0])
# the first reciprocal lattice vector used for testing.
test_B2 = np.array([-.5, 1, 0])
# the second reciprocal lattice vector used for testing.
test_B3 = np.array([.5, 0, 1])
# the third reciprocal lattice vector used for testing.
test_shortest_diagonal = 4
# the index of the shortest diagonal used for testing.

def test_1_generate_tetrahedra():
    # Checking if the correct tetrahedra quadruples are generated.
    assert tetrahedron_method.generate_tetrahedra(test_grid, test_B1, test_B2,
        test_B3, test_shortest_diagonal) == [[7, 8, 6, 2], [7, 8, 5, 2],
        [7, 4, 6, 2], [7, 3, 5, 2], [7, 1, 4, 2], [7, 1, 3, 2]]

def test_2_generate_tetrahedra():
    """Checking if the indices in the tetrahedra quadruples are linked to the 
    right grid points."""
    assert test_grid[tetrahedron_method.generate_tetrahedra(test_grid, test_B1,
        test_B2, test_B3, test_shortest_diagonal)[0][0] - 1] == [.5, 1, 0]


# Tests for calculate_volume.
test_vector1 = np.array([1, 0, 0])
# the first vector used for testing.
test_vector2 = np.array([-.5, 1, 0])
# the second vector used for testing.
test_vector3 = np.array([.5, 0, 1])
# the third vector used for testing.

def test_calculate_volume():
    """Checking if the volume of the parallelepiped with edges spanned by the 
    test vectors is correctly determined."""
    assert tetrahedron_method.calculate_volume(test_vector1, test_vector2,
                                               test_vector3) == 1


# Tests for bound_fermi_energy.
test_energy_bands_1 = np.array([[1, 2, 3], [.4, -5, 6], [7, .8, -9]])
# the energy level for each band at each point used for testing.

def test_bound_fermi_energy1():
    """Checking if the correct upper and lower bound on the Fermi energy level 
    are determined."""
    test_valence_electrons1 = 3
    # the number of valence electrons used for testing.
    assert np.array_equal(np.asarray(tetrahedron_method.bound_fermi_energy(
        test_valence_electrons1, test_energy_bands_1)), [2, -5])

def test_bound_fermi_energy2():
    """Checking if the correct upper and lower bound on the Fermi energy level 
    are determined."""
    test_valence_electrons2 = 6
    # the number of valence electrons used for testing.
    assert np.array_equal(np.asarray(tetrahedron_method.bound_fermi_energy(
        test_valence_electrons2, test_energy_bands_1)), [6, -9])

def test_bound_fermi_energy3():
    """Checking if the correct upper and lower bound on the Fermi energy level 
    are determined."""
    test_valence_electrons3 = 1
    # the number of valence electrons used for testing.
    assert np.array_equal(np.asarray(tetrahedron_method.bound_fermi_energy(
        test_valence_electrons3, test_energy_bands_1)), [7, .4])


# Tests for determine_energy_at_corners:
test_energy_bands_2 = np.array([[1, 2, 3], [.1, .2, .3], [-1, 0, 1],
                              [1.5, 2.5, 3.5], [-.5, -.2, -.1], [.4, .5, .6],
                              [.7, .8, .9], [.01, .02, .03]])
test_tetrahedra_quadruples_2 = [[7, 8, 6, 2], [7, 8, 5, 2], [7, 4, 6, 2],
                              [7, 3, 5, 2], [7, 1, 4, 2], [7, 1, 3, 2]]
test_E_values_by_tetrahedron = np.empty([18, 10])
test_number_of_bands = 3

def test_determine_energy_at_corners1():
    test_m1 = 0
    test_n1 = 0
    assert np.allclose(tetrahedron_method.determine_energy_at_corners(
        test_energy_bands_2, test_E_values_by_tetrahedron, test_m1, test_n1,
        test_tetrahedra_quadruples_2, test_number_of_bands)[0,:], [.01, .1, .4,
        .7, .09, .39, .3, .69, .6, .3])


# Tests for number_of_states_for_tetrahedron.
test_E_values = np.array([.01, .1, .4, .7, .09, .39, .3, .69, .6, .3])
test_V_G = 1
test_V_T = 1 / 6

def test_number_of_states_for_tetrahedron1():
    test_E_Fermi1 = 4
    assert tetrahedron_method.number_of_states_for_tetrahedron(test_E_Fermi1,
        test_E_values, test_V_G, test_V_T) == 1 / 6




"""
def test_integrate_result_positive():
    assert tetrahedron_method.integrate()"""