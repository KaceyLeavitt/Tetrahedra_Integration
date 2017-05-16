def integrate(r_lattice_vectors, grid, PP, valence_electrons, apply_weight_correction):
    #number of intervals in each direction
    n1 =
    """int: the number of grid intervals in the direction of the first reciprocal lattice vector.
    This value is one less than the number of grid points in the direction of the first reciprocal lattice vector."""
    n2 =
    """int: the number of grid intervals in the direction of the second reciprocal lattice vector.
    This value is one less than the number of grid points in the direction of the second reciprocal lattice vector."""
    n3 = r_lattice_vectors.shape[1] - 1
    """int: the number of grid intervals in the direction of the third reciprocal lattice vector.
    This value is one less than the number of grid points in the direction of the third reciprocal lattice vector."""

    k = (n1 + 1)*(n2 + 1)*(n3 + 1)
    """int: the total number of points in the grid."""

    #reciprocal lattice vectors
    b1 = r_lattice_vectors[0,:]
    """NumPy array: the first reciprocal lattice vector."""
    b2 = r_lattice_vectors[1,:]
    """NumPy array: the second reciprocal lattice vector."""
    b3 = r_lattice_vectors[2,:]
    """NumPy array: the third reciprocal lattice vector."""

    #submesh lattice vectors
    B1 = b1/n1
    """NumPy array: the separation between each point in the grid in the direction of the first reciprocal lattice 
        vector."""
    B2 = b2/n2
    """NumPy array: the separation between each point in the grid in the direction of the second reciprocal lattice 
        vector."""
    B3 = b3/n3
    """NumPy array: the separation between each point in the grid in the direction of the third reciprocal lattice 
        vector."""

    #length of diagonals
    diagonal1 = B1 + B2 + B3
    """NumPy array: a vector spanning one of the diagonals of each parallelepiped in the grid."""
    diagonal2 = -B1 + B2 + B3
    """NumPy array: a vector spanning one of the diagonals of each parallelepiped in the grid."""
    diagonal3 = B1 - B2 + B3
    """NumPy array: a vector spanning one of the diagonals of each parallelepiped in the grid."""
    diagonal4 = B1 + B2 - B3
    """NumPy array: a vector spanning one of the diagonals of each parallelepiped in the grid."""

    diagonal1_length = math.sqrt(np.dot(diagonal1, diagonal1))
    """float: the magnitude of the vector diagonal1."""
    diagonal2_length = math.sqrt(np.dot(diagonal2, diagonal2))
    """float: the magnitude of the vector diagonal2."""
    diagonal3_length = math.sqrt(np.dot(diagonal3, diagonal3))
    """float: the magnitude of the vector diagonal3."""
    diagonal4_length = math.sqrt(np.dot(diagonal4, diagonal4))
    """float: the magnitude of the vector diagonal4."""

    shortest_diagonal_length = min(diagonal1_length, diagonal2_length, diagonal3_length, diagonal4_length)
    """float: the magnitude of the shortest vector that spans a diagonal of each parallelepiped in the grid."""
    shortest_diagonal = 0
    """int: the index of the shortest vector that spans a diagonal of each parallelepiped in the grid.
    This value will always be 1, 2, 3, or 4."""

    if shortest_diagonal_length == diagonal1_length:
        shortest_diagonal = 1
    elif shortest_diagonal_length == diagonal2_length:
        shortest_diagonal = 2
    elif shortest_diagonal_length == diagonal3_length:
        shortest_diagonal = 3
    elif shortest_diagonal_length == diagonal4_length:
        shortest_diagonal = 4


    #grid array generation
    tetrahedra_quadruples = []
    """list of lists of ints: a list of quadruples. There is exactly one quadruple for every tetrahedron. Each quadruple 
    is a list of the grid_points indices for the corners of the tetrahedron."""
    grid_points = np.empty([k, 3])
    grid_points_added = 0

    for m in range(n1 + 1):
        for n in range(n2 + 1):
            for l in range(n3 + 1):
                g = m*B1 + n*B2 + l*B3

                N = 1 + l  + (n1 + 1)*(n + (n2 + 1)*m) #submesh point index number

                grid_points[N - 1, 0] = g[0]
                grid_points[N - 1, 1] = g[1]
                grid_points[N - 1, 2] = g[2]
                grid_points_added += 1

                # generate tetrahedra using submesh cell between (m,n,l) and (m+1,n+1,l+1)
                if m != n1 and n != n2 and l != n3:
                    #index numbers for corner points of submesh cell
                    point1_index = N
                    point2_index = 1 + (l + 1)  + (n1 + 1)*(n + (n2 + 1)*m)
                    point3_index = 1 + l + (n1 + 1)*((n + 1) + (n2 + 1)*m)
                    point4_index = 1 + l + (n1 + 1)*(n + (n2 + 1)*(m + 1))
                    point5_index = 1 + (l + 1) + (n1 + 1)*((n + 1) + (n2 + 1)*m)
                    point6_index = 1 + (l + 1) + (n1 + 1)*(n + (n2 + 1)*(m + 1))
                    point7_index = 1 + l + (n1 + 1)*((n + 1) + (n2 + 1)*(m + 1))
                    point8_index = 1 + (l + 1) + (n1 + 1)*((n + 1) + (n2 + 1)*(m + 1))

                    #create quadruples for the corner points of each tetrahedra
                    if shortest_diagonal == 1:
                        tetrahedra_quadruples.append([point1_index, point4_index, point7_index, point8_index])
                        tetrahedra_quadruples.append([point1_index, point3_index, point7_index, point8_index])
                        tetrahedra_quadruples.append([point1_index, point2_index, point5_index, point8_index])
                        tetrahedra_quadruples.append([point1_index, point2_index, point6_index, point8_index])
                        tetrahedra_quadruples.append([point1_index, point4_index, point6_index, point8_index])
                        tetrahedra_quadruples.append([point1_index, point3_index, point5_index, point8_index])
                    elif shortest_diagonal == 2:
                        tetrahedra_quadruples.append([point4_index, point6_index, point2_index, point5_index])
                        tetrahedra_quadruples.append([point4_index, point6_index, point8_index, point5_index])
                        tetrahedra_quadruples.append([point4_index, point1_index, point3_index, point5_index])
                        tetrahedra_quadruples.append([point4_index, point7_index, point3_index, point5_index])
                        tetrahedra_quadruples.append([point4_index, point7_index, point8_index, point5_index])
                        tetrahedra_quadruples.append([point4_index, point1_index, point2_index, point5_index])
                    elif shortest_diagonal == 3:
                        tetrahedra_quadruples.append([point3_index, point1_index, point4_index, point6_index])
                        tetrahedra_quadruples.append([point3_index, point7_index, point4_index, point6_index])
                        tetrahedra_quadruples.append([point3_index, point1_index, point2_index, point6_index])
                        tetrahedra_quadruples.append([point3_index, point7_index, point8_index, point6_index])
                        tetrahedra_quadruples.append([point3_index, point5_index, point2_index, point6_index])
                        tetrahedra_quadruples.append([point3_index, point5_index, point8_index, point6_index])
                    elif shortest_diagonal == 4:
                        tetrahedra_quadruples.append([point7_index, point8_index, point6_index, point2_index])
                        tetrahedra_quadruples.append([point7_index, point8_index, point5_index, point2_index])
                        tetrahedra_quadruples.append([point7_index, point4_index, point6_index, point2_index])
                        tetrahedra_quadruples.append([point7_index, point3_index, point5_index, point2_index])
                        tetrahedra_quadruples.append([point7_index, point1_index, point4_index, point2_index])
                        tetrahedra_quadruples.append([point7_index, point1_index, point3_index, point2_index])


    #determine energy band levels for each of the k points in the submesh
    pseudopotential1_3 = pseudopotential_array[0, 0]
    pseudopotential2_3 = pseudopotential_array[0, 1]
    pseudopotential1_4 = pseudopotential_array[1, 0]
    pseudopotential2_4 = pseudopotential_array[1, 1]
    pseudopotential1_11 = pseudopotential_array[2, 0]
    pseudopotential2_11 = pseudopotential_array[2, 1]

    energy_bands = np.empty([k, 8])
    energy_band_array = np.empty((0,3), float)

    for m in range(-10, 10):
        for n in range(-10, 10):
            for l in range(-10, 10):
                h = m*b1 + n*b2 + l*b3

                if np.dot(h, h) <= radius**2:
                    energy_band_array = np.append(energy_band_array, [h], axis=0)

    for m in range(k):
        k_vec = grid_points[m,:]
        energy_bands_for_point = energy_bands_calculation.energy_band_values(pseudopotential1_3, pseudopotential1_4,
                                                                             pseudopotential1_11, pseudopotential2_3,
                                                                             pseudopotential2_4, pseudopotential2_11,
                                                                             k_vec, energy_band_array, c, atomic_basis_vector)
        energy_bands[m,:] = energy_bands_for_point


    #The Fermi energy level is iteratively determined from the number of states function
    band_index = math.ceil(valence_electrons/2)
    """int: the index of the energy band containing the Fermi energy level."""
    band_minima = np.amin(energy_bands, axis=0)
    """NumPy array: the smallest energy value contained in each energy band."""
    lower_bound = band_minima[band_index - 1]
    """float: an initial lower bound on the Fermi energy level."""
    band_maxima = np.amax(energy_bands, axis=0)
    """NumPy array: the largest energy value contained in each energy band."""
    upper_bound = band_maxima[band_index - 1]
    """float: an initial upper bound on the Fermi energy level."""
    E_Fermi = (lower_bound + upper_bound)/2
    """float: an initial value for the Fermi energy level that will be iteratively refined."""

    V_G = np.dot(b1, np.cross(b2, b3))
    """float: the volume of the reciprocal unit cell."""
    V_T = np.dot(B1, np.cross(B2, B3))/6
    """float: the volume of each tetrahedron in reciprocal space. Each tetrahedron has an equal volume."""

    E_values_by_tetrahedron = np.empty([len(tetrahedra_quadruples)*8, 10])
    """NumPy array: the corner energy values for each band of each tetrahedron."""

    #number of states function generation
    theoretical_number_of_states = valence_electrons/2
    """float: the actual total number of states (integrated density of states) for the reciprocal unit cell."""
    total_number_of_states = 0
    """float: the total number of states (integrated density of states) for the reciprocal unit cell that is calculated 
    for the given Fermi energy level. This value is compared to the actual number of states to iteratively refine the 
    Fermi energy level."""
    number_of_states_error_threshold = .00001
    """float: the Fermi energy level is iteratively refined until the calculated number of states varies by less than 
    this amount from the calculated number of states from the previous iteration."""
    new_number_of_states = 2*number_of_states_error_threshold
    """float: the calculated number of states from the most recent iteration. This value is compared to the calculated 
    number of states from the previous iteration to know whether or not to further refine the Fermi energy level."""
    old_number_of_states = 0
    """float: the calculated number of states from the second most recent iteration. This value is compared to the 
    calculated number of states from the most recent iteration to know whether or not to further refine the Fermi energy 
    level."""

    while abs(new_number_of_states - old_number_of_states) > number_of_states_error_threshold:
        old_number_of_states = total_number_of_states
        total_number_of_states = 0

        for m in range(len(tetrahedra_quadruples)):
            #Each energy band is looped over
            for n in range(8):
                M = 1 + n + 8*m
                "int: a unique index for the current band of the current tetrahedron."

                #The energy at each corner for a given band for a given tetrahedron is determined.
                E_at_corners = np.array([energy_bands[tetrahedra_quadruples[m][0] - 1, n],
                                         energy_bands[tetrahedra_quadruples[m][1] - 1, n],
                                         energy_bands[tetrahedra_quadruples[m][2] - 1, n],
                                         energy_bands[tetrahedra_quadruples[m][3] - 1, n]])
                E_at_corners = np.sort(E_at_corners, axis=0) #This reorders the energy at each corner from least to greatest
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

                E_values_by_tetrahedron[M - 1,:] = [E_1, E_2, E_3, E_4, E_21, E_31, E_32, E_41, E_42, E_43]

                #The number of states for a given band for a given tetrahedron is determined
                number_of_states = 0 #number-of-states function for a given tetrahedron

                if E_Fermi < E_1:
                    number_of_states = 0
                elif E_Fermi >= E_1 and E_Fermi < E_2:
                    number_of_states = V_T/V_G*(E_Fermi - E_1)**3/(E_21*E_31*E_41)
                elif E_Fermi >= E_2 and E_Fermi < E_3:
                    number_of_states = V_T/(V_G*E_31*E_41)*(E_21**2 + 3*E_21*(E_Fermi - E_2) + 3*(E_Fermi - E_2)**2 -
                                                            (E_31 + E_42)*(E_Fermi - E_2)**3/(E_32*E_42))
                elif E_Fermi >= E_3 and E_Fermi < E_4:
                    number_of_states = V_T/V_G*(1 - (E_4 - E_Fermi)**3/(E_41*E_42*E_43))
                elif E_Fermi >= E_4:
                    number_of_states = V_T/V_G
                total_number_of_states = total_number_of_states + number_of_states

        #Adjust the Fermi level
        if total_number_of_states > theoretical_number_of_states:
            #E_Fermi needs to be lowered to in between its current value and lower bound. The current value should be the new upper bound
            upper_bound = E_Fermi
        elif total_number_of_states < theoretical_number_of_states:
            lower_bound = E_Fermi

        E_Fermi = (upper_bound + lower_bound)/2

        new_number_of_states = total_number_of_states

    print("The calculated Fermi Energy Level is:", E_Fermi)


    #The density of states function is determined for each band of each tetrahedron
    density_by_tetrahedron = []
    """list of floats: the density of states at the Fermi energy level for each band of each tetrahedron."""
    for m in range(len(tetrahedra_quadruples)):
        #Each energy band is looped over
        for n in range(8):
            M = 1 + n + 8*m

            E_values = E_values_by_tetrahedron[M - 1,:]
            E_1 = E_values[0]
            E_2 = E_values[1]
            E_3 = E_values[2]
            E_4 = E_values[3]
            E_21 = E_values[4]
            E_31 = E_values[5]
            E_32 = E_values[6]
            E_41 = E_values[7]
            E_42 = E_values[8]
            E_43 = E_values[9]

            #The density of states at the fermi energy level for a given band for a given tetrahedron is determined
            density_of_states = 0
            """float: the density of states at the Fermi energy level for a given band of a given tetrahedron."""

            if E_Fermi < E_1:
                density_of_states = 0
            elif E_Fermi >= E_1 and E_Fermi < E_2:
                density_of_states = V_T/V_G*3*(E_Fermi - E_1)**2/(E_21*E_31*E_41)
            elif E_Fermi >= E_2 and E_Fermi < E_3:
                density_of_states = V_T/(V_G*E_31*E_41)*(3*E_21 + 6*(E_Fermi - E_2) - 3*(E_31 + E_42)*(E_Fermi - E_2)**2/(E_32*E_42))
            elif E_Fermi >= E_3 and E_Fermi < E_4:
                density_of_states = V_T/V_G*3*(E_4 - E_Fermi)**2/(E_41*E_42*E_43)
            elif E_Fermi >= E_4:
                density_of_states = 0

            density_by_tetrahedron.append(density_of_states)

    tetrahedra_by_point = []
    """list of list of ints: for each k point in the grid, a list of the indices of each tetrahedron containing that k 
    point is given."""

    for m in range(k):
        adjacent_tetrahedra = []
        """list of ints: the indices of each tetrahedron containing the given k point in the grid."""

        #find all tetrahedra containing the k point
        for n in range(len(tetrahedra_quadruples)):
            for l in range(4):
                if tetrahedra_quadruples[n][l] == m + 1:
                    adjacent_tetrahedra.append(n + 1)

        tetrahedra_by_point.append(adjacent_tetrahedra)


    #Calculating integration weights at the corners of each tetrahedron from the fermi level and energy levels for each band
    total_energy = 0

    for m in range(len(tetrahedra_quadruples)):
        corners = tetrahedra_quadruples[m] #The corner points for the tetrahedron are called.

        #Each energy band is looped over
        for n in range(8):
            #The energy at each corner for a given band for a given tetrahedron is determined.
            E_at_corners = np.array([energy_bands[corners[0] - 1, n], energy_bands[corners[1] - 1, n],
                                     energy_bands[corners[2] - 1, n], energy_bands[corners[3] - 1, n]])
            corners.sort(key=dict(zip(corners, E_at_corners)).get) #This reorders the corners list according to the energies
            E_at_corners = np.sort(E_at_corners, axis=0) #This reorders the energy at each corner from least to greatest

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

            #The weightings for each corner of the tetrahedron are determined
            w_1 = 0
            w_2 = 0
            w_3 = 0
            w_4 = 0

            if E_Fermi < E_1:
                w_1 = 0
                w_2 = 0
                w_3 = 0
                w_4 = 0
            elif E_Fermi >= E_1 and E_Fermi < E_2:
                C = V_T/(4*V_G)*(E_Fermi - E_1)**3/(E_21*E_31*E_41)

                w_1 = C*(4 - (E_Fermi- E_1)*(1/E_21 + 1/E_31 + 1/E_41))
                w_2 = C*(E_Fermi - E_1)/E_21
                w_3 = C*(E_Fermi - E_1)/E_41
                w_4 = C*(E_Fermi - E_1)/E_41
            elif E_Fermi >= E_2 and E_Fermi < E_3:
                C_1 = V_T/(4*V_G)*(E_Fermi - E_1)**2/(E_41*E_31)
                C_2 = V_T/(4*V_G)*(E_Fermi - E_1)*(E_Fermi - E_2)*(E_3 - E_Fermi)/(E_41*E_32*E_31)
                C_3 = V_T/(4*V_G)*(E_Fermi - E_2)**2*(E_4 - E_Fermi)/(E_42*E_32*E_41)

                w_1 = C_1 + (C_1 + C_2)*(E_3 - E_Fermi)/E_31 + (C_1 + C_2 + C_3)*(E_4 - E_Fermi)/E_41
                w_2 = C_1 + C_2 + C_3 + (C_2 + C_3)*(E_3 - E_Fermi)/E_32 + C_3*(E_4 - E_Fermi)/E_42
                w_3 = (C_1 + C_2)*(E_Fermi - E_1)/E_41 + (C_2 + C_3)*(E_Fermi - E_2)/E_32
                w_4 = (C_1 + C_2 + C_3)*(E_Fermi - E_1)/E_41 + C_3*(E_Fermi - E_2)/E_42
            elif E_Fermi >= E_3 and E_Fermi < E_4:
                C = V_T/(4*V_G)*(E_4 - E_Fermi)**3/(E_41*E_42*E_43)

                w_1 = V_T/(4*V_G) - C*(E_4 - E_Fermi)/E_41
                w_2 = V_T/(4*V_G) - C*(E_4 - E_Fermi)/E_42
                w_3 = V_T/(4*V_G) - C*(E_4 - E_Fermi)/E_43
                w_4 = V_T/(4*V_G) - C*(4 - (1/E_41 + 1/E_42 + 1/ E_43)*(E_4 - E_Fermi))
            elif E_Fermi >= E_4:
                w_1 = V_T/(4*V_G)
                w_2 = V_T/(4*V_G)
                w_3 = V_T/(4*V_G)
                w_4 = V_T/(4*V_G)

            #The weighting corrections are applied
            if apply_weight_correction == True:
                #corrections for w_1
                adjacent_tetrahedra1 = tetrahedra_by_point[corners[0] - 1]
                weight_correction1 = 0

                for p in range(len(adjacent_tetrahedra1)):
                    E_for_adjacent_tetrahedron = E_values_by_tetrahedron[n + 8 * (adjacent_tetrahedra1[p] - 1), :]
                    density_of_states = density_by_tetrahedron[n + 8*(adjacent_tetrahedra1[p] - 1)]
                    corner_E_sum = 0

                    for q in range(4):
                        corner_E_sum += E_for_adjacent_tetrahedron[q] - E_1

                    weight_correction1 += density_of_states/40*corner_E_sum

                w_1 += weight_correction1

                #corrections for w_2
                adjacent_tetrahedra2 = tetrahedra_by_point[corners[1] - 1]
                weight_correction2 = 0

                for p in range(len(adjacent_tetrahedra2)):
                    E_for_adjacent_tetrahedron = E_values_by_tetrahedron[n + 8*(adjacent_tetrahedra2[p] - 1), :]
                    density_of_states = density_by_tetrahedron[n + 8*(adjacent_tetrahedra2[p] - 1)]
                    corner_E_sum = 0

                    for q in range(4):
                        corner_E_sum += E_for_adjacent_tetrahedron[q] - E_2

                    weight_correction2 += density_of_states/40*corner_E_sum

                w_2 += weight_correction2

                #corrections for w_3
                adjacent_tetrahedra3 = tetrahedra_by_point[corners[2] - 1]
                weight_correction3 = 0

                for p in range(len(adjacent_tetrahedra3)):
                    E_for_adjacent_tetrahedron = E_values_by_tetrahedron[n + 8*(adjacent_tetrahedra3[p] - 1), :]
                    density_of_states = density_by_tetrahedron[n + 8*(adjacent_tetrahedra3[p] - 1)]
                    corner_E_sum = 0

                    for q in range(4):
                        corner_E_sum += E_for_adjacent_tetrahedron[q] - E_3

                    weight_correction3 += density_of_states/40*corner_E_sum

                w_3 += weight_correction3

                #corrections for w_4
                adjacent_tetrahedra4 = tetrahedra_by_point[corners[3] - 1]
                weight_correction4 = 0

                for p in range(len(adjacent_tetrahedra4)):
                    E_for_adjacent_tetrahedron = E_values_by_tetrahedron[n + 8*(adjacent_tetrahedra4[p] - 1), :]
                    density_of_states = density_by_tetrahedron[n + 8*(adjacent_tetrahedra4[p] - 1)]
                    corner_E_sum = 0

                    for q in range(4):
                        corner_E_sum += E_for_adjacent_tetrahedron[q] - E_4

                    weight_correction4 += density_of_states/40*corner_E_sum

                w_4 += weight_correction4

            #use weights for integration
            tetrahedra_integral_contribution = w_1*E_1 + w_2*E_2 + w_3*E_3 + w_4*E_4
            total_energy += tetrahedra_integral_contribution

    print ("The integral result is:", total_energy)

    return E_Fermi, total_energy