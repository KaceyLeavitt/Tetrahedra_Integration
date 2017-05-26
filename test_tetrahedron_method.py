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


# Tests for add_tetrahedron.
test_tetrahedra_quadruples_1 = []
# list of lists of indices of corners of tetrahedra used for testing.
test_point_indices = np.array([1, 2, 3, 4, 5, 6, 7, 8])
# indices of corners of parralelepiped used for testing.

def test_add_tetrahedron1():
    # Checking if the tetrahedra quadruples are correctly generated.
    test_shortest_diagonal1 = 1
    # the index of the shortest diagonal used for testing.
    assert tetrahedron_method.add_tetrahedron(test_tetrahedra_quadruples_1,
        test_shortest_diagonal1, test_point_indices) == [[1, 4, 7, 8],
        [1, 3, 7, 8], [1, 2, 5, 8], [1, 2, 6, 8], [1, 4, 6, 8], [1, 3, 5, 8]]

def test_add_tetrahedron2():
    # Checking if the tetrahedra quadruples are correctly generated.
    test_shortest_diagonal2 = 2
    # the index of the shortest diagonal used for testing.
    assert tetrahedron_method.add_tetrahedron(test_tetrahedra_quadruples_1,
        test_shortest_diagonal2, test_point_indices) == [[4, 6, 2, 5],
        [4, 6, 8, 5], [4, 1, 3, 5], [4, 7, 3, 5], [4, 7, 8, 5], [4, 1, 2, 5]]

def test_add_tetrahedron3():
    # Checking if the tetrahedra quadruples are correctly generated.
    test_shortest_diagonal3 = 3
    # the index of the shortest diagonal used for testing.
    assert tetrahedron_method.add_tetrahedron(test_tetrahedra_quadruples_1,
        test_shortest_diagonal3, test_point_indices) == [[3, 1, 4, 6],
        [3, 7, 4, 6], [3, 1, 2, 6], [3, 7, 8, 6], [3, 5, 2, 6], [3, 5, 8, 6]]

def test_add_tetrahedron4():
    # Checking if the tetrahedra quadruples are correctly generated.
    test_shortest_diagonal4 = 4
    # the index of the shortest diagonal used for testing.
    assert tetrahedron_method.add_tetrahedron(test_tetrahedra_quadruples_1,
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
# the energy level values at each point for each band used for testing.
test_tetrahedra_quadruples_2 = [[7, 8, 6, 2], [7, 8, 5, 2], [7, 4, 6, 2],
                              [7, 3, 5, 2], [7, 1, 4, 2], [7, 1, 3, 2]]
# list of lists of indices of corners of tetrahedra used for testing.
test_E_values_by_tetrahedron_1 = np.empty([18, 10])
"""the energy levels at the corners and other useful energy values for each 
band of each tetrahedron used for testing."""
test_number_of_bands_1 = 3
# the number of energy bands used for testing.

def test_determine_energy_at_corners1():
    # Checking if the correct energy values are determined.
    test_m1 = 0
    """the index in the list test_tetrahedra_quadruples_2 for the given 
    tetrahedron used for testing."""
    test_n1 = 0
    # the index of the given energy band used for testing.
    assert np.allclose(tetrahedron_method.determine_energy_at_corners(
        test_energy_bands_2, test_E_values_by_tetrahedron_1, test_m1, test_n1,
        test_tetrahedra_quadruples_2, test_number_of_bands_1)[0,:], [.01, .1, .4,
        .7, .09, .39, .3, .69, .6, .3])


# Tests for number_of_states_for_tetrahedron.
test_E_values_1 = np.array([.01, .1, .4, .7, .09, .39, .3, .69, .6, .3])
"""the energy levels at the corners and other useful energy values for the 
tetrahedron used for testing."""
test_V_G_1 = 1
# the volume of the reciprocal unit cell used for testing.
test_V_T_1 = 1 / 6
# the volume of a single tetrahedron used for testing.

def test_number_of_states_for_tetrahedron1():
    # Checking if the correct number of states has been calculated.
    test_E_Fermi1_1 = 4
    # the Fermi energy level used for testing.
    assert tetrahedron_method.number_of_states_for_tetrahedron(test_E_Fermi1_1,
        test_E_values_1, test_V_G_1, test_V_T_1) == 1 / 6

def test_number_of_states_for_tetrahedron2():
    # Checking if the correct number of states has been calculated.
    test_E_Fermi2_1 = .5
    # the Fermi energy level used for testing.
    assert math.isclose(tetrahedron_method.number_of_states_for_tetrahedron(
        test_E_Fermi2_1, test_E_values_1, test_V_G_1, test_V_T_1), .15593129361)

def test_number_of_states_for_tetrahedron3():
    # Checking if the correct number of states has been calculated.
    test_E_Fermi3_1 = .3
    # the Fermi energy level used for testing.
    assert math.isclose(tetrahedron_method.number_of_states_for_tetrahedron(
        test_E_Fermi3_1, test_E_values_1, test_V_G_1, test_V_T_1), .08553202031)

def test_number_of_states_for_tetrahedron4():
    # Checking if the correct number of states has been calculated.
    test_E_Fermi4_1 = .05
    # the Fermi energy level used for testing.
    assert math.isclose(tetrahedron_method.number_of_states_for_tetrahedron(
        test_E_Fermi4_1, test_E_values_1, test_V_G_1, test_V_T_1),
        .0004404255611984084114)

def test_number_of_states_for_tetrahedron5():
    # Checking if the correct number of states has been calculated.
    test_E_Fermi5_1 = -.1
    # the Fermi energy level used for testing.
    assert tetrahedron_method.number_of_states_for_tetrahedron(test_E_Fermi5_1,
        test_E_values_1, test_V_G_1, test_V_T_1) == 0


# Tests for adjust_fermi_level.
test_E_Fermi_initial = 5
# an initial value for the Fermi energy level used for testing.
test_upper_bound_initial = 10.1
# an initial upper bound for the Fermi energy level used for testing.
test_lower_bound_initial = -.1
# an initial lower bound for the Fermi energy level used for testing.
test_theoretical_number_of_states = 5
# the theoretical number of states used for testing.

def test_adjust_fermi_level1():
    """Checking if the upper and lower bounds and the Fermi energy level are 
    adjusted correctly."""
    test_total_number_of_states1 = 6
    # the calculated number of states used for testing.
    assert np.array_equal(np.asarray(tetrahedron_method.adjust_fermi_level(
        test_E_Fermi_initial, test_upper_bound_initial,
        test_lower_bound_initial, test_total_number_of_states1,
        test_theoretical_number_of_states)), [2.45, 5, -.1])

def test_adjust_fermi_level2():
    """Checking if the upper and lower bounds and the Fermi energy level are 
    adjusted correctly."""
    test_total_number_of_states2 = 4
    # the calculated number of states used for testing.
    assert np.array_equal(np.asarray(tetrahedron_method.adjust_fermi_level(
        test_E_Fermi_initial, test_upper_bound_initial,
        test_lower_bound_initial, test_total_number_of_states2,
        test_theoretical_number_of_states)), [7.55, 10.1, 5])

def test_adjust_fermi_level3():
    """Checking if the upper and lower bounds and the Fermi energy level are 
    adjusted correctly."""
    test_total_number_of_states3 = 5
    # the calculated number of states used for testing.
    assert np.array_equal(np.asarray(tetrahedron_method.adjust_fermi_level(
        test_E_Fermi_initial, test_upper_bound_initial,
        test_lower_bound_initial, test_total_number_of_states3,
        test_theoretical_number_of_states)), [5, 5, 5])


# Tests for calculate_fermi_energy.


# Tests for add_density_of_states_for_tetrahedron.
test_density_by_tetrahedron_1 = []
# the density of states for the tetrahedra used for testing.
test_number_of_bands_2 = 1
# the number of energy bands used for testing.
test_E_values_by_tetrahedron_2 = np.array([[.01, .1, .4, .7, .09, .39, .3, .69,
                                    .6, .3], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
"""the corner energy values and other useful energy values for each tetrahedron 
and energy band used for testing."""
test_V_G_2 = 1
# the volume of the reciprocal unit cell used for testing.
test_V_T_2 = 1 / 6
# the volume of each tetrahedron used for testing.
test_m = 0
# the index for the given tetrahedron used for testing.
test_n_1 = 0
# the index for the given energy band used for testing.

def test_add_density_of_states_for_tetrahedron1():
    # Checking if the correct density of states has been calculated.
    test_E_Fermi1_2 = 4
    # the Fermi energy level used for testing.
    assert tetrahedron_method.add_density_of_states_for_tetrahedron(
        test_density_by_tetrahedron_1, test_E_Fermi1_2, test_number_of_bands_2,
        test_E_values_by_tetrahedron_2, test_V_G_2, test_V_T_2, test_m,
        test_n_1) == [0]

def test_add_density_of_states_for_tetrahedron2():
    # Checking if the correct density of states has been calculated.
    test_E_Fermi2_2 = .5
    # the Fermi energy level used for testing.
    density_of_states_list = tetrahedron_method.\
        add_density_of_states_for_tetrahedron(test_density_by_tetrahedron_1,
        test_E_Fermi2_2, test_number_of_bands_2,
        test_E_values_by_tetrahedron_2, test_V_G_2, test_V_T_2, test_m, test_n_1)
    assert math.isclose(density_of_states_list[0], .1610305958132045)
    assert len(density_of_states_list) == 1

def test_add_density_of_states_for_tetrahedron3():
    # Checking if the correct density of states has been calculated.
    test_E_Fermi3_2 = .3
    # the Fermi energy level used for testing.
    density_of_states_list = tetrahedron_method.\
        add_density_of_states_for_tetrahedron(test_density_by_tetrahedron_1,
        test_E_Fermi3_2, test_number_of_bands_2,
        test_E_values_by_tetrahedron_2, test_V_G_2, test_V_T_2, test_m, test_n_1)
    assert math.isclose(density_of_states_list[0], 0.5016722408)
    assert len(density_of_states_list) == 1

def test_add_density_of_states_for_tetrahedron4():
    # Checking if the correct density of states has been calculated.
    test_E_Fermi4_2 = .05
    # the Fermi energy level used for testing.
    density_of_states_list = tetrahedron_method.\
        add_density_of_states_for_tetrahedron(test_density_by_tetrahedron_1,
        test_E_Fermi4_2, test_number_of_bands_2,
        test_E_values_by_tetrahedron_2, test_V_G_2, test_V_T_2, test_m, test_n_1)
    assert math.isclose(density_of_states_list[0], 0.03303191708)
    assert len(density_of_states_list) == 1

def test_add_density_of_states_for_tetrahedron5():
    # Checking if the correct density of states has been calculated.
    test_E_Fermi5_2 = -.1
    # the Fermi energy level used for testing.
    assert tetrahedron_method.add_density_of_states_for_tetrahedron(
        test_density_by_tetrahedron_1, test_E_Fermi5_2, test_number_of_bands_2,
        test_E_values_by_tetrahedron_2, test_V_G_2, test_V_T_2, test_m,
        test_n_1) == [0]


# Tests for calculate_density_of_states.
test_tetrahedra_quadruples_3 = [[7, 8, 6, 2], [7, 8, 5, 2]]
# the indices of the corners of each tetrahedron.

def test_add_density_of_states_for_tetrahedron1():
    # Checking if the correct density of states has been calculated.
    test_E_Fermi1_3 = 4
    # the Fermi energy level used for testing.
    assert tetrahedron_method.calculate_density_of_states(test_E_Fermi1_3,
        test_tetrahedra_quadruples_3, test_number_of_bands_2,
        test_E_values_by_tetrahedron_2, test_V_G_2, test_V_T_2) == [0, 0]

def test_add_density_of_states_for_tetrahedron2():
    # Checking if the correct density of states has been calculated.
    test_E_Fermi2_3 = .5
    # the Fermi energy level used for testing.
    density_of_states_list = tetrahedron_method.calculate_density_of_states(
        test_E_Fermi2_3, test_tetrahedra_quadruples_3, test_number_of_bands_2,
        test_E_values_by_tetrahedron_2, test_V_G_2, test_V_T_2)
    assert math.isclose(density_of_states_list[0], .1610305958132045)
    assert density_of_states_list[1] == 0
    assert len(density_of_states_list) == 2

def test_add_density_of_states_for_tetrahedron3():
    # Checking if the correct density of states has been calculated.
    test_E_Fermi3_3 = .3
    # the Fermi energy level used for testing.
    density_of_states_list = tetrahedron_method.calculate_density_of_states(
        test_E_Fermi3_3, test_tetrahedra_quadruples_3, test_number_of_bands_2,
        test_E_values_by_tetrahedron_2, test_V_G_2, test_V_T_2)
    assert math.isclose(density_of_states_list[0], 0.5016722408)
    assert density_of_states_list[1] == 0
    assert len(density_of_states_list) == 2

def test_add_density_of_states_for_tetrahedron4():
    # Checking if the correct density of states has been calculated.
    test_E_Fermi4_3 = .05
    # the Fermi energy level used for testing.
    density_of_states_list = tetrahedron_method.calculate_density_of_states(
        test_E_Fermi4_3, test_tetrahedra_quadruples_3, test_number_of_bands_2,
        test_E_values_by_tetrahedron_2, test_V_G_2, test_V_T_2)
    assert math.isclose(density_of_states_list[0], 0.03303191708)
    assert density_of_states_list[1] == 0
    assert len(density_of_states_list) == 2

def test_add_density_of_states_for_tetrahedron5():
    # Checking if the correct density of states has been calculated.
    test_E_Fermi5_3 = -.1
    # the Fermi energy level used for testing.
    assert tetrahedron_method.calculate_density_of_states(test_E_Fermi5_3,
        test_tetrahedra_quadruples_3, test_number_of_bands_2,
        test_E_values_by_tetrahedron_2, test_V_G_2, test_V_T_2) == [0, 0]


# Tests for cluster_tetrahedra_by_point.
test_k = 8
# the number of points used for testing.
test_tetrahedra_quadruples_4 = [[7, 8, 6, 2], [7, 8, 5, 2], [7, 4, 6, 2],
                                [7, 3, 5, 2], [7, 1, 4, 2], [7, 1, 3, 2]]
# the indices of the corner points of each tetrahedron used for testing.

def test_cluster_tetrahedra_by_point():
    """Checking if the tetrahedra containing each k point are correctly 
    identified."""
    assert tetrahedron_method.cluster_tetrahedra_by_point(
        test_tetrahedra_quadruples_4, test_k) == [[5, 6], [1, 2, 3, 4, 5, 6],
        [4, 6], [3, 5], [2, 4], [1, 3], [1, 2, 3, 4, 5, 6], [1, 2]]


# Tests for sort_corners_by_energy.
test_corners_1 = [1, 2, 3, 4]
# the unsorted list of indices for the tetrahedron's corners used for testing.
test_energy_bands_3 = np.array([[1, 2, 3], [.1, .2, .3], [-1, 0, 1],
                                [1.5, 2.5, 3.5]])
# the energy levels for each band at each k point used for testing.
test_n_2 = 0
# the band index used for testing.

def test_sort_corners_by_energy():
    # Checking if the corners and energy values are correctly sorted
    assert np.array_equal(tetrahedron_method.sort_corners_by_energy(
        test_corners_1, test_energy_bands_3, test_n_2)[0],
                          np.array([-1, .1, 1, 1.5]))
    assert np.array_equal(tetrahedron_method.sort_corners_by_energy(
        test_corners_1, test_energy_bands_3, test_n_2)[1], [3, 2, 1, 4])


# Tests for calculate_integration_weights
test_E_values_2 = np.array([.01, .1, .4, .7, .09, .39, .3, .69, .6, .3])
"""the energy values at the corners and other useful energy values for the 
tetrahedron used for testing."""
test_V_G_3 = 1
# volume of the reciprocal unit cell used for testing.
test_V_T_3 = 1 / 6
# volume of each tetrahedron used for testing.

def test_calculate_integration_weights1():
    """Checking if the integration weights are correctly determined for the 
    given Fermi energy level."""
    test_E_Fermi1_4 = 4
    # the Fermi energy level used for testing.
    assert np.allclose(np.asarray(tetrahedron_method.
                                  calculate_integration_weights(
        test_E_Fermi1_4, test_E_values_2, test_V_G_3, test_V_T_3)),
        np.array([1 / 24, 1 / 24, 1 / 24, 1 / 24]))

def test_calculate_integration_weights2():
    """Checking if the integration weights are correctly determined for the 
    given Fermi energy level."""
    test_E_Fermi2_4 = .5
    # the Fermi energy level used for testing.
    assert np.allclose(np.asarray(tetrahedron_method.
        calculate_integration_weights(test_E_Fermi2_4, test_E_values_2,
        test_V_G_3, test_V_T_3)), np.array([0.040888741083027997,
                                            0.040772052245482197,
                                            0.039877437824297727,
                                            0.03439306245964511]))

def test_calculate_integration_weights3():
    """Checking if the integration weights are correctly determined for the 
    given Fermi energy level."""
    test_E_Fermi3_4 = .3
    # the Fermi energy level used for testing.
    assert np.allclose(np.asarray(tetrahedron_method.
        calculate_integration_weights(test_E_Fermi3_4, test_E_values_2,
        test_V_G_3, test_V_T_3)), np.array([0.029524237974708948,
                                            0.027748530767854439,
                                            0.017482962624174069,
                                            0.010776288947891552]))

def test_calculate_integration_weights4():
    """Checking if the integration weights are correctly determined for the 
    given Fermi energy level."""
    test_E_Fermi4_4 = .05
    # the Fermi energy level used for testing.
    assert np.allclose(np.asarray(tetrahedron_method.
        calculate_integration_weights(test_E_Fermi4_4, test_E_values_2,
        test_V_G_3, test_V_T_3)), np.array([0.000373813445476548,
                                            0.000048936173466500,
                                            0.000011292963107654,
                                            0.000006382979147804]))

def test_calculate_integration_weights5():
    """Checking if the integration weights are correctly determined for the 
    given Fermi energy level."""
    test_E_Fermi5_4 = -.1
    # the Fermi energy level used for testing.
    assert np.allclose(np.asarray(tetrahedron_method.
        calculate_integration_weights(test_E_Fermi5_4, test_E_values_2,
                                      test_V_G_3, test_V_T_3)),
                       np.array([0, 0, 0, 0]))


# Tests for calculate_weight_correction.
test_adjacent_tetrahedra = [1, 2]
# the tetrahedra containing the given corner point used for testing.
test_E_values_by_tetrahedron_3 = np.array([[.01, .1, .4, .7, .09, .39, .3, .69,
                        .6, .3], [.01, .1, .4, .7, .09, .39, .3, .69, .6, .3]])
"""the corner energy values and other useful energy values for each tetrahedron 
and energy band used for testing."""
test_n_3 = 0
# the band index used for testing.
test_number_of_bands_3 = 1
# the number of bands used for testing.
test_E = .4
# the energy value at the given corner used for testing.
test_density_by_tetrahedron_2 = [.2, .1]
# the density of states for the tetrahedra used for testing.

def test_calculate_weight_correction():
    # Checking if the weight correction is calculated correctly.
    assert math.isclose(tetrahedron_method.calculate_weight_correction(
        test_adjacent_tetrahedra, test_E_values_by_tetrahedron_3, test_n_3,
        test_number_of_bands_3, test_E, test_density_by_tetrahedron_2),
        -0.002925)


# Tests for adjust_integration_weightings.
test_tetrahedra_by_point = [[5, 6], [1, 2, 3, 4, 5, 6], [4, 6], [3, 5], [2, 4],
                            [1, 3], [1, 2, 3, 4, 5, 6], [1, 2]]
# the indices of the tetrahedra containing each k point used for testing.
test_corners_2 = [7, 8, 6, 2]
# the indices of the corner points of the given tetrahedron used for testing.
test_E_values_by_tetrahedron_4 = np.array([
    [.01, .1, .4, .7, .09, .39, .3, .69, .6, .3],
    [.01, .1, .4, .7, .09, .39, .3, .69, .6, .3],
    [.01, .1, .4, .7, .09, .39, .3, .69, .6, .3],
    [.01, .1, .4, .7, .09, .39, .3, .69, .6, .3],
    [.01, .1, .4, .7, .09, .39, .3, .69, .6, .3],
    [.01, .1, .4, .7, .09, .39, .3, .69, .6, .3]])
"""the corner energy values and other useful energy values for each tetrahedron 
and energy band used for testing."""
test_n_4 = 0
# the band index used for testing.
test_number_of_bands_4 = 1
# the number of bands used for testing.
test_density_by_tetrahedron_3 = [.1, .2, .3, .4, .5, .6]
# the density of states for each tetrahedron used for testing.
test_weightings = np.array([1, 2, 3, 4])
"""the unadjusted integration weightings for the corners of the given 
tetrahedron used for testing."""
test_E_at_corners = np.array([.01, .1, .4, .7])
# the energy levels at each corner of the tetrahedron used for testing.

def test_adjust_integration_weightings():
    # Checking if each of the integration weights are adjusted correctly.
    assert np.allclose(tetrahedron_method.adjust_integration_weightings(
        test_tetrahedra_by_point, test_corners_2,
        test_E_values_by_tetrahedron_4, test_n_4, test_number_of_bands_4,
        test_density_by_tetrahedron_3, test_weightings, test_E_at_corners),
        np.array([1.061425, 2.006075, 2.9961, 3.916525]))


# Tests for perform_integration.

# Tests for integrate.