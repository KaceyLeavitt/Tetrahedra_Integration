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
test_B1 = np.array([1, 2, 3])
# First submesh lattice vector used for testing.
test_B2 = np.array([4, 5, 6])
# Second submesh lattice vector used for testing.
test_B3 = np.array([7, 8, 9])
# Third submesh lattice vector used for testing.

def test_generate_diagonal_vectors1():
    # Checking if the first diagonal vector is correctly determined.
    assert np.array_equal(tetrahedron_method.generate_diagonal_vectors(
        test_B1, test_B2, test_B3)[0], [12, 15, 18])

def test_generate_diagonal_vectors2():
    # Checking if the second diagonal vector is correctly determined.
    assert np.array_equal(tetrahedron_method.generate_diagonal_vectors(
        test_B1, test_B2, test_B3)[1], [10, 11, 12])

def test_generate_diagonal_vectors3():
    # Checking if the third diagonal vector is correctly determined.
    assert np.array_equal(tetrahedron_method.generate_diagonal_vectors(
        test_B1, test_B2, test_B3)[2], [4, 5, 6])

def test_generate_diagonal_vectors4():
    # Checking if the fourth diagonal vector is correctly determined.
    assert np.array_equal(tetrahedron_method.generate_diagonal_vectors(
        test_B1, test_B2, test_B3)[3], [-2, -1, 0])


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
test_B1 = np.array([1, 2, 3])
# First submesh lattice vector used for testing.
test_B2 = np.array([4, 5, -6])
# Second submesh lattice vector used for testing.
test_B3 = np.array([7, 8, 9])
# Third submesh lattice vector used for testing.

def test_determine_parallelepiped_corners2():
    """Checking if the second corner of the parallelepiped is correctly 
    determined."""
    assert np.array_equal(tetrahedron_method.determine_parallelepiped_corners(test_point1,
        test_B1, test_B2, test_B3)[0], [7.1, 7.8, 9.3])

def test_determine_parallelepiped_corners3():
    """Checking if the third corner of the parallelepiped is correctly 
    determined."""
    assert np.array_equal(tetrahedron_method.determine_parallelepiped_corners(
        test_point1, test_B1, test_B2, test_B3)[1], [4.1, 4.8, -5.7])

def test_determine_parallelepiped_corners4():
    """Checking if the fourth corner of the parallelepiped is correctly 
    determined."""
    assert np.array_equal(tetrahedron_method.determine_parallelepiped_corners(
        test_point1, test_B1, test_B2, test_B3)[2], [1.1, 1.8, 3.3])

def test_determine_parallelepiped_corners5():
    """Checking if the fifth corner of the parallelepiped is correctly 
    determined."""
    assert np.allclose(tetrahedron_method.determine_parallelepiped_corners(
        test_point1, test_B1, test_B2, test_B3)[3], [11.1, 12.8, 3.3])

def test_determine_parallelepiped_corners6():
    """Checking if the sixth corner of the parallelepiped is correctly 
    determined."""
    assert np.array_equal(tetrahedron_method.determine_parallelepiped_corners(
        test_point1, test_B1, test_B2, test_B3)[4], [8.1, 9.8, 12.3])

def test_determine_parallelepiped_corners7():
    """Checking if the seventh corner of the parallelepiped is correctly 
    determined."""
    assert np.array_equal(tetrahedron_method.determine_parallelepiped_corners(
        test_point1, test_B1, test_B2, test_B3)[5], [5.1, 6.8, -2.7])

def test_determine_parallelepiped_corners8():
    """Checking if the eighth corner of the parallelepiped is correctly 
    determined."""
    assert np.allclose(tetrahedron_method.determine_parallelepiped_corners(
        test_point1, test_B1, test_B2, test_B3)[6], [12.1, 14.8, 6.3])






"""test_grid = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]

def test_integrate_result_positive():
    assert tetrahedron_method.integrate()"""