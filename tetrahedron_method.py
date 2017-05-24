import math
import numpy as np


def generate_r_lattice_vectors(r_lattice_vectors):
    """Generates the reciprocal lattice vectors.
    
    Args:
        r_lattice_vectors (NumPy array): the reciprocal lattice vectors. Each 
            column of the array is one of the vectors.
    
    Returns:
        (b1, b2, b3) (tuple of Numpy arrays): b1, b2, and b3 are the first, 
            second, and third reciprocal lattice vectors respectively.
    """

    b1 = r_lattice_vectors[:, 0]
    # NumPy array: the first reciprocal lattice vector.
    b2 = r_lattice_vectors[:, 1]
    # NumPy array: the second reciprocal lattice vector.
    b3 = r_lattice_vectors[:, 2]
    # NumPy array: the third reciprocal lattice vector.

    return (b1, b2, b3)


def generate_submesh_lattice_vectors(grid_vecs):
    """Generates the submesh lattice vectors. These vectors tell you the 
    separation between the points in the grid.
    
    Args:
        grid_vecs (NumPy array): the vectors used to create the grid of k 
            points. All grid points are an integer combination of these 
            vectors. Each column of the array is one of the vectors.
    
    Returns:
        (B1, B2, B3) (tuple of NumPy arrays): B1, B2, and B3 are the first, 
            second, and third submesh lattice vectors respectively.
    """

    B1 = grid_vecs[:, 0]
    """NumPy array: the separation between each point in the grid in the first 
    direction."""
    B2 = grid_vecs[:, 1]
    """NumPy array: the separation between each point in the grid in the second 
    direction."""
    B3 = grid_vecs[:, 2]
    """NumPy array: the separation between each point in the grid in the third 
    direction."""

    return (B1, B2, B3)


def generate_diagonal_vectors(B1, B2, B3):
    """Generates vectors that span the diagonals of the smallest 
    parallelepipeds that can be created with integer combinations of the 
    submesh lattice vectors.
    
    Args:
        B1 (NumPy array): the first submesh lattice vector.
        B2 (NumPy array): the second submesh lattice vector.
        B3 (NumPy array): the third submesh lattice vector.
        
    Returns:
        (diagonal1, diagonal2, diagonal3, diagonal4) (tuple of NumPy arrays): 
            diagonal1, diagonal2, diagonal3, and diagonal4 are the vectors that 
            span the first, second, third, and fourth diagonals of the smallest 
            parallelepipeds that can be created with integer combinations of 
            the submesh lattice vectors respectively.
    """

    diagonal1 = B1 + B2 + B3
    """NumPy array: a vector spanning one of the diagonals of each 
    parallelepiped in the grid."""
    diagonal2 = -B1 + B2 + B3
    """NumPy array: a vector spanning one of the diagonals of each 
    parallelepiped in the grid."""
    diagonal3 = B1 - B2 + B3
    """NumPy array: a vector spanning one of the diagonals of each 
    parallelepiped in the grid."""
    diagonal4 = B1 + B2 - B3
    """NumPy array: a vector spanning one of the diagonals of each 
    parallelepiped in the grid."""

    return (diagonal1, diagonal2, diagonal3, diagonal4)


def calculate_diagonal_length(diagonal1, diagonal2, diagonal3, diagonal4):
    """Calculates the length of each diagonal of the parallelepipeds in the 
    grid.
    
    Args:
        diagonal1 (NumPy array): the vector that spans the first diagonal of 
            the parallelepipeds.
        diagonal2 (NumPy array): the vector that spans the second diagonal of 
            the parallelepipeds.
        diagonal3 (NumPy array): the vector that spans the third diagonal of 
            the parallelepipeds.
        diagonal4 (NumPy array): the vector that spans the fourth diagonal of 
            the parallelepipeds.
    
    Returns:
        (diagonal1_length, diagonal2_length, diagonal3_length, diagonal4_length)
            (tuple of floats): the magnitudes of the vectors diagonal1, 
            diagonal2, diagonal3, and diagonal4 respectively.
    """

    diagonal1_length = math.sqrt(np.dot(diagonal1, diagonal1))
    # float: the magnitude of the vector diagonal1.
    diagonal2_length = math.sqrt(np.dot(diagonal2, diagonal2))
    # float: the magnitude of the vector diagonal2.
    diagonal3_length = math.sqrt(np.dot(diagonal3, diagonal3))
    # float: the magnitude of the vector diagonal3.
    diagonal4_length = math.sqrt(np.dot(diagonal4, diagonal4))
    # float: the magnitude of the vector diagonal4.

    return (diagonal1_length, diagonal2_length, diagonal3_length,
            diagonal4_length)


def determine_shortest_diagonal(diagonal1_length, diagonal2_length,
                                diagonal3_length, diagonal4_length):
    """Determines which of the four diagonals of the parallelepipeds in the 
    grid is the shortest.
    
    Args:
        diagonal1_length (float): the magnitude of the vector spanning the 
            first diagonal of the parallelepipeds.
        diagonal2_length (float): the magnitude of the vector spanning the 
            second diagonal of the parallelepipeds.
        diagonal3_length (float): the magnitude of the vector spanning the 
            third diagonal of the parallelepipeds.
        diagonal4_length (float): the magnitude of the vector spanning the 
            fourth diagonal of the parallelepipeds.
            
    Returns:
        shortest_diagonal (int): either 1, 2, 3, or 4; an index designating 
            whether diagonal1, diagonal2, diagonal3, or diagonal is the 
            shortest.
    """

    shortest_diagonal_length = min(diagonal1_length, diagonal2_length,
                                   diagonal3_length, diagonal4_length)
    """float: the magnitude of the shortest vector that spans a diagonal of 
    each parallelepiped in the grid."""
    shortest_diagonal = 0
    """int: the index of the shortest vector that spans a diagonal of each 
    parallelepiped in the grid. This value will always be 1, 2, 3, or 4."""

    if shortest_diagonal_length == diagonal1_length:
        shortest_diagonal = 1
    elif shortest_diagonal_length == diagonal2_length:
        shortest_diagonal = 2
    elif shortest_diagonal_length == diagonal3_length:
        shortest_diagonal = 3
    elif shortest_diagonal_length == diagonal4_length:
        shortest_diagonal = 4

    return shortest_diagonal


def determine_parallelepiped_corners(point1, B1, B2, B3):
    """Determines the grid points that define the corners of a given 
    parallelepiped. All of the edges of the parallelepipeds can be spanned by 
    the submesh lattice vectors.
    
    Args:
        point1 (Numpy array): the coordinates of the first corner of a 
            parallelepiped. The other corners are determined based on this 
            point.
        B1 (NumPy array): the first submesh lattice vector.
        B2 (NumPy array): the second submesh lattice vector.
        B3 (NumPy array): the third submesh lattice vector.
        
    Returns:
        (point2, point3, point4, point5, point6, point7, point8) (tuple of 
            NumPy arrays): vectors defining the coordinates of the second, 
            third, fourth, fifth, sixth, seventh and eighth coordinates of a 
            parallelepiped respectively.
    """

    point2 = point1 + B3
    """NumPy array: the coordinates of the second corner of a parallelepiped 
    from the grid."""
    point3 = point1 + B2
    """NumPy array: the coordinates of the third corner of a parallelepiped 
    from the grid."""
    point4 = point1 + B1
    """NumPy array: the coordinates of the fourth corner of a parallelepiped 
    from the grid."""
    point5 = point1 + B3 + B2
    """NumPy array: the coordinates of the fifth corner of a parallelepiped 
    from the grid."""
    point6 = point1 + B3 + B1
    """NumPy array: the coordinates of the sixth corner of a parallelepiped 
    from the grid."""
    point7 = point1 + B2 + B1
    """NumPy array: the coordinates of the seventh corner of a parallelepiped 
    from the grid."""
    point8 = point1 + B3 + B2 + B1
    """NumPy array: the coordinates of the eighth corner of a parallelepiped 
    from the grid."""

    return (point2, point3, point4, point5, point6, point7, point8)


def add_tetrahedron(tetrahedra_quadruples, shortest_diagonal, point_indices):
    """Adds a quadruple of point indices of the corners of a tetrahedron to the 
    list tetrahedra_quadruples. The corner points of the tetrahedron are formed 
    by breaking every parallelepiped in the grid up into six tetrahedra that 
    all share the shortest diagonal as an edge and all have the same volume.
    
    Args:
        tetrahedra_quadruples (list of lists of ints): a list of quadruples. 
            There is exactly one quadruple for every tetrahedron. Each 
            quadruple is a list of the grid_points indices for the corners of 
            the tetrahedron.
        shortest_diagonal (int): either 1, 2, 3, or 4; an index designating 
            whether diagonal1, diagonal2, diagonal3, or diagonal is the 
            shortest.
        point_indices (NumPy array of ints): the indices for point one through 
            point eight in the grid arranged in order by point number (i.e. 
            point 1 is first, point 2 is second, etc.).
    
    Returns:
        tetrahedra_quadruples (list of lists of ints): a list of quadruples. 
            There is exactly one quadruple for every tetrahedron. Each 
            quadruple is a list of the grid_points indices for the corners of 
            the tetrahedron.
    """

    point1_index = point_indices[0]
    # int: the index for point1 in the grid.
    point2_index = point_indices[1]
    # int: the index for point2 in the grid.
    point3_index = point_indices[2]
    # int: the index for point3 in the grid.
    point4_index = point_indices[3]
    # int: the index for point4 in the grid.
    point5_index = point_indices[4]
    # int: the index for point5 in the grid.
    point6_index = point_indices[5]
    # int: the index for point6 in the grid.
    point7_index = point_indices[6]
    # int: the index for point7 in the grid.
    point8_index = point_indices[7]
    # int: the index for point8 in the grid.

    if shortest_diagonal == 1:
        tetrahedra_quadruples.append([point1_index, point4_index, point7_index,
                                      point8_index])
        tetrahedra_quadruples.append([point1_index, point3_index, point7_index,
                                      point8_index])
        tetrahedra_quadruples.append([point1_index, point2_index, point5_index,
                                      point8_index])
        tetrahedra_quadruples.append([point1_index, point2_index, point6_index,
                                      point8_index])
        tetrahedra_quadruples.append([point1_index, point4_index, point6_index,
                                      point8_index])
        tetrahedra_quadruples.append([point1_index, point3_index, point5_index,
                                      point8_index])
    elif shortest_diagonal == 2:
        tetrahedra_quadruples.append([point4_index, point6_index, point2_index,
                                      point5_index])
        tetrahedra_quadruples.append([point4_index, point6_index, point8_index,
                                      point5_index])
        tetrahedra_quadruples.append([point4_index, point1_index, point3_index,
                                      point5_index])
        tetrahedra_quadruples.append([point4_index, point7_index, point3_index,
                                      point5_index])
        tetrahedra_quadruples.append([point4_index, point7_index, point8_index,
                                      point5_index])
        tetrahedra_quadruples.append([point4_index, point1_index, point2_index,
                                      point5_index])
    elif shortest_diagonal == 3:
        tetrahedra_quadruples.append([point3_index, point1_index, point4_index,
                                      point6_index])
        tetrahedra_quadruples.append([point3_index, point7_index, point4_index,
                                      point6_index])
        tetrahedra_quadruples.append([point3_index, point1_index, point2_index,
                                      point6_index])
        tetrahedra_quadruples.append([point3_index, point7_index, point8_index,
                                      point6_index])
        tetrahedra_quadruples.append([point3_index, point5_index, point2_index,
                                      point6_index])
        tetrahedra_quadruples.append([point3_index, point5_index, point8_index,
                                      point6_index])
    elif shortest_diagonal == 4:
        tetrahedra_quadruples.append([point7_index, point8_index, point6_index,
                                      point2_index])
        tetrahedra_quadruples.append([point7_index, point8_index, point5_index,
                                      point2_index])
        tetrahedra_quadruples.append([point7_index, point4_index, point6_index,
                                      point2_index])
        tetrahedra_quadruples.append([point7_index, point3_index, point5_index,
                                      point2_index])
        tetrahedra_quadruples.append([point7_index, point1_index, point4_index,
                                      point2_index])
        tetrahedra_quadruples.append([point7_index, point1_index, point3_index,
                                      point2_index])

    return tetrahedra_quadruples


def generate_tetrahedra(grid, B1, B2, B3, shortest_diagonal):
    """Creates a list of corner points of tetrahedra that are formed by 
    breaking every parallelepiped in the grid up into six tetrahedra that all 
    share the shortest diagonal as an edge and all have the same volume.
    
    Args:
        grid (list of lists of floats): the coordinates of each k point in the 
            reciprocal lattice unit cell at which calculations will be 
            performed.
        B1 (NumPy array): the first submesh lattice vector.
        B2 (NumPy array): the second submesh lattice vector.
        B1 (NumPy array): the third submesh lattice vector.
        shortest_diagonal (int): either 1, 2, 3, or 4; an index designating 
            whether diagonal1, diagonal2, diagonal3, or diagonal is the 
            shortest.
            
    Returns:
        tetrahedra_quadruples (list of lists of ints): a list of quadruples. 
            There is exactly one quadruple for every tetrahedron. Each 
            quadruple is a list of the grid_points indices for the corners of 
            the tetrahedron.
    """

    tetrahedra_quadruples = []
    """list of lists of ints: a list of quadruples. There is exactly one 
    quadruple for every tetrahedron. Each quadruple is a list of the 
    grid_points indices for the corners of the tetrahedron."""

    for m in range(len(grid)):
        point1 = np.asarray(grid[m])
        """NumPy array: the coordinates of the first corner of a parallelepiped 
        from the grid."""
        point2, point3, point4, point5, point6, point7, point8 = \
            determine_parallelepiped_corners(point1, B1, B2, B3)
        """tuple of NumPy arrays: the coordinates of the second through eighth 
        corners of a parallelepiped from the grid respectively."""

        pt2_in_grid = False
        # bool: whether or not point2 is a point in the grid.
        pt3_in_grid = False
        # bool: whether or not point3 is a point in the grid.
        pt4_in_grid = False
        # bool: whether or not point4 is a point in the grid.
        pt5_in_grid = False
        # bool: whether or not point5 is a point in the grid.
        pt6_in_grid = False
        # bool: whether or not point6 is a point in the grid.
        pt7_in_grid = False
        # bool: whether or not point7 is a point in the grid.
        pt8_in_grid = False
        # bool: whether or not point8 is a point in the grid.

        point1_index = m + 1
        # int: the index for point1 in the grid.
        point2_index = 0
        # int: the index for point2 in the grid.
        point3_index = 0
        # int: the index for point3 in the grid.
        point4_index = 0
        # int: the index for point4 in the grid.
        point5_index = 0
        # int: the index for point5 in the grid.
        point6_index = 0
        # int: the index for point6 in the grid.
        point7_index = 0
        # int: the index for point7 in the grid.
        point8_index = 0
        # int: the index for point8 in the grid.

        for n in range(len(grid)):
            grid_point = np.asarray(grid[n])
            """NumPy array: the coordinates of a point in the grid to be 
            compared against."""

            if np.array_equal(grid_point, point2):
                pt2_in_grid = True
                point2_index = n + 1
            elif np.array_equal(grid_point, point3):
                pt3_in_grid = True
                point3_index = n + 1
            elif np.array_equal(grid_point, point4):
                pt4_in_grid = True
                point4_index = n + 1
            elif np.array_equal(grid_point, point5):
                pt5_in_grid = True
                point5_index = n + 1
            elif np.array_equal(grid_point, point6):
                pt6_in_grid = True
                point6_index = n + 1
            elif np.array_equal(grid_point, point7):
                pt7_in_grid = True
                point7_index = n + 1
            elif np.array_equal(grid_point, point8):
                pt8_in_grid = True
                point8_index = n + 1

        if pt2_in_grid and pt3_in_grid and pt4_in_grid and pt5_in_grid and \
                pt6_in_grid and pt7_in_grid and pt8_in_grid:
            # create quadruples for the corner points of each tetrahedra
            point_indices = np.array([point1_index, point2_index, point3_index,
                                      point4_index, point5_index, point6_index,
                                      point7_index, point8_index])
            """NumPy array: the indices of the corner points of a 
            parallelepiped in the grid."""
            tetrahedra_quadruples = add_tetrahedron(tetrahedra_quadruples,
                                            shortest_diagonal, point_indices)

    return tetrahedra_quadruples


def calculate_volume(vector1, vector2, vector3):
    """Calculates the volume of a parallelepiped that has its edges spanned by 
    vector1, vector2, and vector3.
    
    Args:
        vector1 (NumPy array): the first vector that spans four of the edges of 
            the parallelepiped.
        vector2 (NumPy array): the second vector that spans four of the edges of 
            the parallelepiped.
        vector3 (NumPy array): the third vector that spans four of the edges of 
            the parallelepiped.
            
    Returns:
        volume (float): the volume of the parallelepiped.
    """

    volume = np.dot(vector1, np.cross(vector2, vector3))

    return volume


def bound_fermi_energy(valence_electrons, energy_bands):
    """Calculates initial upper and lower bounds on the Fermi energy level. The 
    number of valence electrons is used to determine the band containing the 
    Fermi energy. The minimum and maximum energy level of the band are used as 
    the initial bounds.
    
    Args:
        valence_electrons (int): the number of valence electrons possessed by 
            the element in question.
        energy_bands (NumPy array): the energy for each energy band of each of 
            the k points in the grid.
    
    Returns:
        upper_bound (float): an initial upper bound on the Fermi energy level.
        lower_bound (float): an initial lower bound on the Fermi energy level.
    """

    band_index = math.ceil(valence_electrons / 2)
    # int: the index of the energy band containing the Fermi energy level.
    band_minima = np.amin(energy_bands, axis=0)
    # NumPy array: the smallest energy value contained in each energy band.
    lower_bound = band_minima[band_index - 1]
    # float: an initial lower bound on the Fermi energy level.
    band_maxima = np.amax(energy_bands, axis=0)
    # NumPy array: the largest energy value contained in each energy band.
    upper_bound = band_maxima[band_index - 1]
    # float: an initial upper bound on the Fermi energy level.

    return (upper_bound, lower_bound)


def determine_energy_at_corners(energy_bands, E_values_by_tetrahedron, m, n,
                                tetrahedra_quadruples, number_of_bands):
    """Determines the energy values at the corners and other useful energy 
    values for a given energy band and tetrahedron.
    
    Args:
        energy_bands (NumPy array): the energy for each energy band of each of 
            the k points in the grid.
        E_values_by_tetrahedron (NumPy array): the energy levels and other 
            useful energy levels for each band for each tetrahedron.
        m  (int): the index in the list tetrahedra_quadruples for the given 
            tetrahedron.
        n (int): the index of the given energy band.
        tetrahedra_quadruples (list of lists of ints): a list of quadruples. 
            There is exactly one quadruple for every tetrahedron. Each 
            quadruple is a list of the grid_points indices for the corners of 
            the tetrahedron.
        number_of_bands (int): the number of energy bands that the energy level 
            is calculated at.
            
    Returns:
        E_values_by_tetrahedron (NumPy array): the energy levels and other 
            useful energy levels for each band for each tetrahedron.
    """

    M = 1 + n + number_of_bands * m
    # int: a unique index for the current band of the current tetrahedron.

    """The energy at each corner for a given band for a given tetrahedron is 
    determined."""
    E_at_corners = np.array([energy_bands[tetrahedra_quadruples[m][0] - 1, n],
                             energy_bands[tetrahedra_quadruples[m][1] - 1, n],
                             energy_bands[tetrahedra_quadruples[m][2] - 1, n],
                             energy_bands[tetrahedra_quadruples[m][3] - 1, n]])
    """NumPy Array: the energy level at each corner of the tetrahedron for the 
    specified energy band."""
    E_at_corners = np.sort(E_at_corners, axis=0)
    # This reorders the energy at each corner from least to greatest

    E_1 = E_at_corners[0]
    """float: the energy level at the first corner of the tetrahedron for the 
    specified energy band."""
    E_2 = E_at_corners[1]
    """float: the energy level at the second corner of the tetrahedron for the 
    specified energy band."""
    E_3 = E_at_corners[2]
    """float: the energy level at the third corner of the tetrahedron for the 
    specified energy band."""
    E_4 = E_at_corners[3]
    """float: the energy level at the fourth corner of the tetrahedron for the 
    specified energy band."""
    E_21 = E_2 - E_1
    # float: a useful value for the following calculations.
    E_31 = E_3 - E_1
    # float: a useful value for the following calculations.
    E_32 = E_3 - E_2
    # float: a useful value for the following calculations.
    E_41 = E_4 - E_1
    # float: a useful value for the following calculations.
    E_42 = E_4 - E_2
    # float: a useful value for the following calculations.
    E_43 = E_4 - E_3
    # float: a useful value for the following calculations.

    E_values_by_tetrahedron[M - 1, :] = [E_1, E_2, E_3, E_4, E_21, E_31, E_32,
                                         E_41, E_42, E_43]

    return E_values_by_tetrahedron


def number_of_states_for_tetrahedron(E_Fermi, E_values, V_G, V_T):
    """Calculates the contribution to the total number of states that is made 
    by a single tetrahedron and energy band. This calculation is performed with 
    the assumption of a certain Fermi energy level. This assumed value for the 
    Fermi energy level is the current estimate that will continue to be 
    iteratively refined.
    
    Args:
        E_Fermi (float): the current estimate for the Fermi energy level.
        E_values (NumPy array): the energy levels at the corners and other 
            useful energy values for the given tetrahedron and energy band.
        V_G (float): the volume of the reciprocal unit cell.
        V_T (float): the volume of each tetrahedron in reciprocal space.
        
    Returns:
        number_of_states (float): the contribution to the total number of 
            states that is made by a single tetrahedron and energy band.
    """

    [E_1, E_2, E_3, E_4, E_21, E_31, E_32, E_41, E_42, E_43] = E_values

    number_of_states = 0
    """float: the number-of-states for a given tetrahedron and Fermi energy 
    level."""

    if E_Fermi < E_1:
        number_of_states = 0
    elif E_Fermi >= E_1 and E_Fermi < E_2:
        number_of_states = V_T / V_G * (E_Fermi - E_1) ** 3 / (E_21 * E_31 *
                                                               E_41)
    elif E_Fermi >= E_2 and E_Fermi < E_3:
        number_of_states = V_T / (V_G * E_31 * E_41) * (E_21 ** 2 + 3 * E_21 *
            (E_Fermi - E_2) + 3 * (E_Fermi - E_2) ** 2 - (E_31 + E_42) *
            (E_Fermi - E_2) ** 3 / (E_32 * E_42))
    elif E_Fermi >= E_3 and E_Fermi < E_4:
        number_of_states = V_T / V_G * (1 - (E_4 - E_Fermi) ** 3 / (E_41 *
                                                                E_42 * E_43))
    elif E_Fermi >= E_4:
        number_of_states = V_T / V_G

    return number_of_states


def adjust_fermi_level(E_Fermi, upper_bound, lower_bound,
                       total_number_of_states, theoretical_number_of_states,):
    """Adjusts the upper and lower bounds on the Fermi energy level by 
    comparing the calculated number of states with the theoretical number of 
    states. The Fermi energy level is then set in between the upper and lower 
    bounds.
    
    Args:
        E_Fermi (float): the current estimate for the  Fermi energy level.
        upper_bound (float): the current upper bound on the Fermi energy level.
        lower_bound (float): the current lower bound on the Fermi energy level.
        total_number_of_states (float): the total number of states (integrated 
            density of states) for the reciprocal unit cell that is calculated 
            for the given Fermi energy level.
        theoretical_number_of_states (float): the actual total number of states 
            (integrated density of states) for the reciprocal unit cell.
    
    Returns:
        E_Fermi (float): the revised estimate for the Fermi energy level.
        upper_bound (float): the revised upper bound on the Fermi energy level.
        lower_bound (float): the revised lower bound on the Fermi energy level.
    """

    if total_number_of_states > theoretical_number_of_states:
        upper_bound = E_Fermi
    elif total_number_of_states < theoretical_number_of_states:
        lower_bound = E_Fermi
    elif total_number_of_states == theoretical_number_of_states:
        upper_bound = E_Fermi
        lower_bound = E_Fermi

    E_Fermi = (upper_bound + lower_bound) / 2

    return E_Fermi, upper_bound, lower_bound


def calculate_fermi_energy(valence_electrons, energy_bands, V_G, V_T,
                           tetrahedra_quadruples, number_of_bands):
    """Iteratively determines the Fermi energy level from the number of states 
    function.
    
    Args:
        valence_electrons (int): the number of valence electrons possessed by 
            the element in question.
        energy_bands (NumPy array): the energy for each energy band of each of 
            the k points in the grid.
        V_G (float): the volume of a parallelepiped with edges spanned by the 
            reciprocal lattice vectors.
        V_T (float): the volume of each tetrahedron.
        tetrahedra_quadruples (list of lists of ints): a list of quadruples. 
            There is exactly one quadruple for every tetrahedron. Each 
            quadruple is a list of the grid_points indices for the corners of 
            the tetrahedron.
        number_of_bands (int): the number of energy bands that the energy level 
            is calculated at.
    
    Returns:
        E_Fermi (float): the calculated Fermi energy level.
        E_values_by_tetrahedron (NumPy array): the energy levels and other 
            useful energy levels for each band for each tetrahedron.
    """

    upper_bound, lower_bound = bound_fermi_energy(valence_electrons,
                                                  energy_bands)
    """tuple of floats: an initial upper and lower bound on the Fermi energy 
    level respectively."""
    E_Fermi = (lower_bound + upper_bound) / 2
    """float: an initial value for the Fermi energy level that will be 
    iteratively refined by tightening the upper and lower bounds."""

    E_values_by_tetrahedron = np.empty([len(tetrahedra_quadruples) *
                                        number_of_bands, 10])
    """NumPy array: the corner energy values and other useful energy values for 
    each band of each tetrahedron."""

    # number of states function generation
    theoretical_number_of_states = valence_electrons / 2
    """float: the actual total number of states (integrated density of states) 
    for the reciprocal unit cell."""
    total_number_of_states = 0
    """float: the total number of states (integrated density of states) for the 
    reciprocal unit cell that is calculated for the given Fermi energy level. 
    This value is compared to the actual number of states to iteratively refine 
    the Fermi energy level."""
    number_of_states_error_threshold = .00001
    """float: the Fermi energy level is iteratively refined until the 
    calculated number of states varies by less than this amount from the 
    calculated number of states from the previous iteration."""
    new_number_of_states = 2 * number_of_states_error_threshold
    """float: the calculated number of states from the most recent iteration. 
    This value is compared to the calculated number of states from the previous 
    iteration to know whether or not to further refine the Fermi energy 
    level."""
    old_number_of_states = 0
    """float: the calculated number of states from the second most recent 
    iteration. This value is compared to the calculated number of states from 
    the most recent iteration to know whether or not to further refine the 
    Fermi energy level."""

    while abs(new_number_of_states - old_number_of_states) > \
            number_of_states_error_threshold:
        old_number_of_states = total_number_of_states
        total_number_of_states = 0

        for m in range(len(tetrahedra_quadruples)):
            # Each energy band is looped over
            for n in range(number_of_bands):
                M = 1 + n + number_of_bands * m
                """int: a unique index for the current band of the current 
                tetrahedron."""

                """The energy at each corner for a given band for a given 
                tetrahedron is determined."""
                E_values_by_tetrahedron = determine_energy_at_corners(
                    energy_bands, E_values_by_tetrahedron, m, n,
                    tetrahedra_quadruples, number_of_bands)

                E_values = E_values_by_tetrahedron[M - 1, :]
                """NumPy array: the corner energy values and other useful 
                energy values for a given tetrahedron and energy band."""

                """The number of states for a given band for a given 
                tetrahedron is determined."""
                number_of_states = number_of_states_for_tetrahedron(E_Fermi,
                    E_values, V_G, V_T)
                """float: the number-of-states for a given tetrahedron and 
                Fermi energy level."""

                total_number_of_states = total_number_of_states + \
                                         number_of_states

        # Adjust the Fermi level
        (E_Fermi, upper_bound, lower_bound) = adjust_fermi_level(E_Fermi,
            upper_bound, lower_bound, total_number_of_states,
            theoretical_number_of_states)

        new_number_of_states = total_number_of_states

    print("The calculated Fermi Energy Level is:", E_Fermi)

    return (E_Fermi, E_values_by_tetrahedron)


def add_density_of_states_for_tetrahedron(density_by_tetrahedron, E_Fermi,
        number_of_bands, E_values_by_tetrahedron, V_G, V_T, m, n):
    """Calculates the density of states at the Fermi energy level for a given 
    energy band and tetrahedron.

    Args:
        density_by_tetrahedron (list of floats): the density of states at the 
            Fermi energy level for each band of each tetrahedron.
        E_Fermi (float): the Fermi energy level.
        number_of_bands (int): the number of energy bands that the energy level 
            is calculated at.
        E_values_by_tetrahedron (NumPy array): the energy levels and other 
            useful energy levels for each band for each tetrahedron.
        V_G (float): the volume of a parallelepiped with edges spanned by the 
            reciprocal lattice vectors.
        V_T (float): the volume of each tetrahedron.
        m (int): the index in the list tetrahedra_quadruples for the given 
            tetrahedron.
        n (int): the index of the given energy band.

    Returns:
        density_by_tetrahedron (list of floats): the density of states at the 
            Fermi energy level for each band of each tetrahedron.
    """

    M = 1 + n + number_of_bands * m
    # int: a unique index for the current band of the current tetrahedron.

    E_values = E_values_by_tetrahedron[M - 1, :]
    """NumPy Array: the energy level at each corner plus other useful energy 
    level values for the specified tetrahedron and energy band."""
    E_1 = E_values[0]
    """float: the energy level at the first corner of the tetrahedron for the 
    specified energy band."""
    E_2 = E_values[1]
    """float: the energy level at the second corner of the tetrahedron for the 
    specified energy band."""
    E_3 = E_values[2]
    """float: the energy level at the third corner of the tetrahedron for the 
    specified energy band."""
    E_4 = E_values[3]
    """float: the energy level at the fourth corner of the tetrahedron for the 
    specified energy band."""
    E_21 = E_values[4]
    # float: a useful value for the following calculations.
    E_31 = E_values[5]
    # float: a useful value for the following calculations.
    E_32 = E_values[6]
    # float: a useful value for the following calculations.
    E_41 = E_values[7]
    # float: a useful value for the following calculations.
    E_42 = E_values[8]
    # float: a useful value for the following calculations.
    E_43 = E_values[9]
    # float: a useful value for the following calculations.

    """The density of states at the fermi energy level for a given band for a 
    given tetrahedron is determined"""
    density_of_states = 0
    """float: the density of states at the Fermi energy level for a given band 
    of a given tetrahedron."""

    if E_Fermi < E_1:
        density_of_states = 0
    elif E_Fermi >= E_1 and E_Fermi < E_2:
        density_of_states = V_T / V_G * 3 * (E_Fermi - E_1) ** 2 / (E_21 *
                                                                E_31 * E_41)
    elif E_Fermi >= E_2 and E_Fermi < E_3:
        density_of_states = V_T / (V_G * E_31 * E_41) * (3 * E_21 + 6 *
                            (E_Fermi - E_2) - 3 * (E_31 + E_42) * (E_Fermi -
                            E_2) ** 2 / (E_32 * E_42))
    elif E_Fermi >= E_3 and E_Fermi < E_4:
        density_of_states = V_T / V_G * 3 * (E_4 - E_Fermi) ** 2 / (E_41 *
                                                                E_42 * E_43)
    elif E_Fermi >= E_4:
        density_of_states = 0

    density_by_tetrahedron.append(density_of_states)


def calculate_density_of_states(E_Fermi, tetrahedra_quadruples,
                                number_of_bands, E_values_by_tetrahedron, V_G,
                                V_T):
    """Calculates the density of states at the Fermi energy level for each 
    energy band and tetrahedron.
    
    Args:
        E_Fermi (float): the Fermi energy level.
        tetrahedra_quadruples (list of lists of ints): a list of quadruples. 
            There is exactly one quadruple for every tetrahedron. Each 
            quadruple is a list of the grid_points indices for the corners of 
            the tetrahedron.
        number_of_bands (int): the number of energy bands that the energy level 
            is calculated at.
        E_values_by_tetrahedron (NumPy array): the energy levels and other 
            useful energy levels for each band for each tetrahedron.
        V_G (float): the volume of a parallelepiped with edges spanned by the 
            reciprocal lattice vectors.
        V_T (float): the volume of each tetrahedron.
    
    Returns:
        density_by_tetrahedron (list of floats): the density of states at the 
            Fermi energy level for each band of each tetrahedron.
    """

    density_by_tetrahedron = []
    """list of floats: the density of states at the Fermi energy level for each 
    band of each tetrahedron."""
    for m in range(len(tetrahedra_quadruples)):
        # Each energy band is looped over
        for n in range(number_of_bands):
            density_by_tetrahedron = add_density_of_states_for_tetrahedron(
                density_by_tetrahedron, E_Fermi, number_of_bands,
                E_values_by_tetrahedron, V_G, V_T, m, n)

    return density_by_tetrahedron


def integrate(r_lattice_vectors, grid_vecs, grid, PP, valence_electrons,
              apply_weight_correction):
    """A function that performs Brillouin zone integration of the energy bands 
    to determine the total energy. An improved version of the tetrahedron 
    method is used.
    
    This function is an implementation of the algorithm proposed in "Improved 
    tetrahedron method for Brillouin-zone integrations" by Peter E. Blöchl, O. 
    Jepsen, and O. K. Andersen from Physical Review B 49, 16223 – Published 15 
    June 1994. It is modified in that the submesh vectors (grid_vecs) are not 
    necessarily the reciprocal lattice vectors divided by an integer.
    
    Args:
        r_lattice_vectors (NumPy array): the reciprocal lattice vectors. Each 
            column of the array is one of the vectors.
        grid_vecs (NumPy array): the vectors used to create the grid of k 
            points. All grid points are an integer combination of these 
            vectors. Each column of the array is one of the vectors.
        grid (list of lists of floats): the coordinates of each k point in the 
            reciprocal lattice unit cell at which calculations will be 
            performed.
        PP (function): calculates the first n energy levels at a given k point 
            using the pseudopotential method. PP has two arguments. For its 
            first argument, PP takes a list of floats containing the 
            coordinates of the k point to evaluate the energy levels at. For 
            its second argument, PP takes the number of eigenvalues (equivalent 
            to the number of energy levels) to return. It returns a sorted list 
            (from least to greatest) of the first n eigenvalues (energy values) 
            at that point.
        valence_electrons (int): the number of valence electrons possessed by 
            the element in question.
        apply_weight_correction (bool): true if the integration weights should 
            be corrected to take curvature into account, false otherwise.
    
    Returns:
        E_Fermi (float): the calculated Fermi energy level.
        total_energy (float): the total calculated energy in the Brillouin 
            zone, the result of the integration.
    """

    k = len(grid)
    # int: the total number of points in the grid.

    # reciprocal lattice vectors
    b1, b2, b3 = generate_r_lattice_vectors(r_lattice_vectors)
    """tuple of NumPy arrays: the first, second, and third reciprocal lattice 
    vectors respectively."""

    # submesh lattice vectors
    B1, B2, B3 = generate_submesh_lattice_vectors(grid_vecs)
    """tuple of NumPy arrays: the separation between each point in the grid in 
    the first, second, and third directions respectively."""

    # length of diagonals
    diagonal1, diagonal2, diagonal3, diagonal4 = generate_diagonal_vectors(
        B1, B2, B3)
    """tuple of NumPy arrays: vectors spanning all four of the diagonals of 
    each parallelepiped in the grid."""

    diagonal1_length, diagonal2_length, diagonal3_length, diagonal4_length = \
        calculate_diagonal_length(diagonal1, diagonal2, diagonal3, diagonal4)
    """tuple of floats: the magnitudes of the vectors diagonal1, diagonal2, 
    diagonal3, and diagonal4 respectively."""

    shortest_diagonal = determine_shortest_diagonal(diagonal1_length,
        diagonal2_length, diagonal3_length, diagonal4_length)

    # Tetrahedra are generated
    tetrahedra_quadruples = generate_tetrahedra(grid, B1, B2, B3,
                                                shortest_diagonal)
    """list of lists of ints: a list of quadruples. There is exactly one 
    quadruple for every tetrahedron. Each quadruple is a list of the 
    grid_points indices for the corners of the tetrahedron."""


    # determine energy band levels for each of the k points in the submesh
    number_of_bands = 8
    """int: the number of energy band levels to calculate for each point in the 
    grid."""
    energy_bands = np.empty([k, number_of_bands])
    """NumPy array: the energy band levels for each point in the grid. Each row 
    corresponds to the point in the grid with the same index, and each column 
    corresponds to a different energy band."""

    for m in range(k):
        k_vec = grid[m]
        """list of floats: the coordinates of the point that the energy levels 
        will be calculated at."""
        energy_bands_for_point = PP(k_vec, number_of_bands)
        """list of floats: the energy levels at the point k_vec returned by the 
        function PP."""
        energy_bands[m, :] = np.asarray(energy_bands_for_point)

    """The Fermi energy level is iteratively determined from the number of 
    states function"""
    V_G = calculate_volume(b1, b2, b3)
    """float: the volume of the reciprocal unit cell."""
    V_T = calculate_volume(B1, B2, B3) / 6
    """float: the volume of each tetrahedron in reciprocal space. Each 
    tetrahedron has an equal volume. The 6 is present because there are 6 
    tetrahedra per parallelepiped in the grid."""

    E_Fermi, E_values_by_tetrahedron = calculate_fermi_energy(
        valence_electrons, energy_bands, V_G, V_T, tetrahedra_quadruples,
        number_of_bands)
    """tuple with float as first element and NumPy array as second element: 
    E_Fermi is the calculated value for the Fermi energy level. 
    E_values_by_tetrahedron contains the corner energy values and other useful 
    energy values for each band of each tetrahedron."""

    """The density of states function is determined for each band of each 
    tetrahedron."""
    density_by_tetrahedron = calculate_density_of_states(E_Fermi,
        tetrahedra_quadruples, number_of_bands, E_values_by_tetrahedron, V_G,
        V_T)
    """list of floats: the density of states at the Fermi energy level for each 
    band of each tetrahedron."""

    tetrahedra_by_point = []
    """list of list of ints: for each k point in the grid, a list of the indices of each tetrahedron containing that k 
    point is given."""

    for m in range(k):
        adjacent_tetrahedra = []
        """list of ints: the indices of each tetrahedron containing the given k point in the grid."""

        # find all tetrahedra containing the k point
        for n in range(len(tetrahedra_quadruples)):
            for l in range(4):
                if tetrahedra_quadruples[n][l] == m + 1:
                    adjacent_tetrahedra.append(n + 1)

        tetrahedra_by_point.append(adjacent_tetrahedra)

    # Calculating integration weights at the corners of each tetrahedron from the fermi level and energy levels for each
    # band. Integration of the energy levels over the Brillouin zone is performed.
    total_energy = 0
    """float: the total energy in the Brillouin zone. This value is the final result of the integration."""

    for m in range(len(tetrahedra_quadruples)):
        corners = tetrahedra_quadruples[m]  # The corner points for the tetrahedron are called.
        """list of ints: the indices of each corner of the given tetrahedron."""

        # Each energy band is looped over
        for n in range(number_of_bands):
            # The energy at each corner for a given band for a given tetrahedron is determined.
            E_at_corners = np.array([energy_bands[corners[0] - 1, n], energy_bands[corners[1] - 1, n],
                                     energy_bands[corners[2] - 1, n], energy_bands[corners[3] - 1, n]])
            corners.sort(
                key=dict(zip(corners, E_at_corners)).get)  # This reorders the corners list according to the energies
            E_at_corners = np.sort(E_at_corners,
                                   axis=0)  # This reorders the energy at each corner from least to greatest

            E_1 = E_at_corners[0]
            E_2 = E_at_corners[1]
            E_3 = E_at_corners[2]
            E_4 = E_at_corners[3]
            E_21 = E_2 - E_1
            E_31 = E_3 - E_1
            E_32 = E_3 - E_2
            E_41 = E_4 - E_1
            E_42 = E_4 - E_2
            E_43 = E_4 - E_3

            # The weightings for each corner of the tetrahedron are determined
            w_1 = 0
            """float: the weighting assigned to the energy level at the first corner of the tetrahedron when integration 
            is being performed."""
            w_2 = 0
            """float: the weighting assigned to the energy level at the second corner of the tetrahedron when 
            integration is being performed."""
            w_3 = 0
            """float: the weighting assigned to the energy level at the third corner of the tetrahedron when integration 
            is being performed."""
            w_4 = 0
            """float: the weighting assigned to the energy level at the fourth corner of the tetrahedron when 
            integration is being performed."""

            if E_Fermi < E_1:
                w_1 = 0
                w_2 = 0
                w_3 = 0
                w_4 = 0
            elif E_Fermi >= E_1 and E_Fermi < E_2:
                C = V_T / (4 * V_G) * (E_Fermi - E_1) ** 3 / (E_21 * E_31 * E_41)
                """float: a useful value for the following calculations."""

                w_1 = C * (4 - (E_Fermi - E_1) * (1 / E_21 + 1 / E_31 + 1 / E_41))
                w_2 = C * (E_Fermi - E_1) / E_21
                w_3 = C * (E_Fermi - E_1) / E_41
                w_4 = C * (E_Fermi - E_1) / E_41
            elif E_Fermi >= E_2 and E_Fermi < E_3:
                C_1 = V_T / (4 * V_G) * (E_Fermi - E_1) ** 2 / (E_41 * E_31)
                """float: a useful value for the following calculations."""
                C_2 = V_T / (4 * V_G) * (E_Fermi - E_1) * (E_Fermi - E_2) * (E_3 - E_Fermi) / (E_41 * E_32 * E_31)
                """float: a useful value for the following calculations."""
                C_3 = V_T / (4 * V_G) * (E_Fermi - E_2) ** 2 * (E_4 - E_Fermi) / (E_42 * E_32 * E_41)
                """float: a useful value for the following calculations."""

                w_1 = C_1 + (C_1 + C_2) * (E_3 - E_Fermi) / E_31 + (C_1 + C_2 + C_3) * (E_4 - E_Fermi) / E_41
                w_2 = C_1 + C_2 + C_3 + (C_2 + C_3) * (E_3 - E_Fermi) / E_32 + C_3 * (E_4 - E_Fermi) / E_42
                w_3 = (C_1 + C_2) * (E_Fermi - E_1) / E_41 + (C_2 + C_3) * (E_Fermi - E_2) / E_32
                w_4 = (C_1 + C_2 + C_3) * (E_Fermi - E_1) / E_41 + C_3 * (E_Fermi - E_2) / E_42
            elif E_Fermi >= E_3 and E_Fermi < E_4:
                C = V_T / (4 * V_G) * (E_4 - E_Fermi) ** 3 / (E_41 * E_42 * E_43)
                """float: a useful value for the following calculations."""

                w_1 = V_T / (4 * V_G) - C * (E_4 - E_Fermi) / E_41
                w_2 = V_T / (4 * V_G) - C * (E_4 - E_Fermi) / E_42
                w_3 = V_T / (4 * V_G) - C * (E_4 - E_Fermi) / E_43
                w_4 = V_T / (4 * V_G) - C * (4 - (1 / E_41 + 1 / E_42 + 1 / E_43) * (E_4 - E_Fermi))
            elif E_Fermi >= E_4:
                w_1 = V_T / (4 * V_G)
                w_2 = V_T / (4 * V_G)
                w_3 = V_T / (4 * V_G)
                w_4 = V_T / (4 * V_G)

            # The weighting corrections are applied
            if apply_weight_correction == True:
                # corrections for w_1
                adjacent_tetrahedra1 = tetrahedra_by_point[corners[0] - 1]
                """list of ints: the indices of each tetrahedron that contains the first corner of the given 
                tetrahedron."""
                weight_correction1 = 0
                """float: how much the weighting for the first corner should be adjusted by to take curvature into 
                account."""

                for p in range(len(adjacent_tetrahedra1)):
                    E_for_adjacent_tetrahedron = E_values_by_tetrahedron[n + number_of_bands * (adjacent_tetrahedra1[p]
                                                                                                - 1), :]
                    """list of floats: the energy values for the second tetrahedron that contains the first corner of 
                    the first tetrahedron in question."""
                    density_of_states = density_by_tetrahedron[n + number_of_bands * (adjacent_tetrahedra1[p] - 1)]
                    """float the density of states for the second tetrahedron that contains the first corner of the 
                    first tetrahedron in question."""
                    corner_E_sum = 0
                    """float: the sum of the difference between the energy value at the first corner of the first 
                    tetrahedron and the the energy value at each corner of the second tetrahedron."""

                    for q in range(4):
                        corner_E_sum += E_for_adjacent_tetrahedron[q] - E_1

                    weight_correction1 += density_of_states / 40 * corner_E_sum

                w_1 += weight_correction1

                # corrections for w_2
                adjacent_tetrahedra2 = tetrahedra_by_point[corners[1] - 1]
                """list of ints: the indices of each tetrahedron that contains the second corner of the given 
                tetrahedron."""
                weight_correction2 = 0
                """float: how much the weighting for the second corner should be adjusted by to take curvature into 
                account."""

                for p in range(len(adjacent_tetrahedra2)):
                    E_for_adjacent_tetrahedron = E_values_by_tetrahedron[n + number_of_bands * (adjacent_tetrahedra2[p] - 1), :]
                    """list of floats: the energy values for the second tetrahedron that contains the second corner of 
                    the first tetrahedron in question."""
                    density_of_states = density_by_tetrahedron[n + number_of_bands * (adjacent_tetrahedra2[p] - 1)]
                    """float the density of states for the second tetrahedron that contains the second corner of the 
                    first tetrahedron in question."""
                    corner_E_sum = 0
                    """float: the sum of the difference between the energy value at the second corner of the first 
                    tetrahedron and the the energy value at each corner of the second tetrahedron."""

                    for q in range(4):
                        corner_E_sum += E_for_adjacent_tetrahedron[q] - E_2

                    weight_correction2 += density_of_states / 40 * corner_E_sum

                w_2 += weight_correction2

                # corrections for w_3
                adjacent_tetrahedra3 = tetrahedra_by_point[corners[2] - 1]
                """list of ints: the indices of each tetrahedron that contains the third corner of the given 
                tetrahedron."""
                weight_correction3 = 0
                """float: how much the weighting for the third corner should be adjusted by to take curvature into 
                account."""

                for p in range(len(adjacent_tetrahedra3)):
                    E_for_adjacent_tetrahedron = E_values_by_tetrahedron[n + number_of_bands * (adjacent_tetrahedra3[p]
                                                                                                - 1), :]
                    """list of floats: the energy values for the second tetrahedron that contains the third corner of 
                    the first tetrahedron in question."""
                    density_of_states = density_by_tetrahedron[n + number_of_bands * (adjacent_tetrahedra3[p] - 1)]
                    """float the density of states for the second tetrahedron that contains the third corner of the 
                    first tetrahedron in question."""
                    corner_E_sum = 0
                    """float: the sum of the difference between the energy value at the third corner of the first 
                    tetrahedron and the the energy value at each corner of the second tetrahedron."""

                    for q in range(4):
                        corner_E_sum += E_for_adjacent_tetrahedron[q] - E_3

                    weight_correction3 += density_of_states / 40 * corner_E_sum

                w_3 += weight_correction3

                # corrections for w_4
                adjacent_tetrahedra4 = tetrahedra_by_point[corners[3] - 1]
                """list of ints: the indices of each tetrahedron that contains the fourth corner of the given 
                tetrahedron."""
                weight_correction4 = 0
                """float: how much the weighting for the fourth corner should be adjusted by to take curvature into 
                account."""

                for p in range(len(adjacent_tetrahedra4)):
                    E_for_adjacent_tetrahedron = E_values_by_tetrahedron[n + number_of_bands * (adjacent_tetrahedra4[p]
                                                                                                - 1), :]
                    """list of floats: the energy values for the second tetrahedron that contains the fourth corner of 
                    the first tetrahedron in question."""
                    density_of_states = density_by_tetrahedron[n + number_of_bands * (adjacent_tetrahedra4[p] - 1)]
                    """float the density of states for the second tetrahedron that contains the fourth corner of the 
                    first tetrahedron in question."""
                    corner_E_sum = 0
                    """float: the sum of the difference between the energy value at the fourth corner of the first 
                    tetrahedron and the the energy value at each corner of the second tetrahedron."""

                    for q in range(4):
                        corner_E_sum += E_for_adjacent_tetrahedron[q] - E_4

                    weight_correction4 += density_of_states / 40 * corner_E_sum

                w_4 += weight_correction4

            # use weights for integration
            tetrahedra_integral_contribution = w_1 * E_1 + w_2 * E_2 + w_3 * E_3 + w_4 * E_4
            """float: the contribution of the given tetrahedron to the total energy in the Brillouin zone."""
            total_energy += tetrahedra_integral_contribution

    print("The integral result is:", total_energy)

    return E_Fermi, total_energy