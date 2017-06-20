import numpy as np
import time
import tetrahedron_method

r_lattice_vectors2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
grid_vecs = np.array([[.16180339887, 0, 0], [0, .14159265358979, 0], [0, 0, .11111111]])
offset = np.array([.05, .05, .05])
grid = []

for m in range(2):
    for n in range(2):
        for l in range(2):
            grid.append((grid_vecs[:,0] * m + grid_vecs[:,1] * n + grid_vecs[:,2] * l + offset).tolist())

start = time.time()
def index_grid_points(grid, grid_vecs, offset):
    grid_inverse = np.linalg.inv(grid_vecs)
    max_m = 0
    max_n = 0
    max_l = 0
    index_values = []

    for grid_point in grid:
        grid_point_not_offset = np.asarray(grid_point) - offset
        grid_indices = np.dot(grid_inverse, grid_point_not_offset).T

        m = int(grid_indices[0] + .5)
        n = int(grid_indices[1] + .5)
        l = int(grid_indices[2] + .5)

        index_values.append([m, n, l])

        if m > max_m:
            max_m = m

        if n > max_n:
            max_n = n

        if l > max_l:
            max_l = l

    limit_on_number_of_points = (max_m + 1) * (max_n + 1) * (max_l + 1)
    grid_points = np.zeros((limit_on_number_of_points, 3))
    max_indices = [max_m, max_n, max_l]

    for p in range(len(grid)):
        [m, n, l] = index_values[p]
        grid_point = grid[p]

        N = l + (max_l + 1) * (n + (max_n + 1) * m)

        grid_points[N,:] = grid_point

    indexed_grid = grid_points.tolist()

    return (indexed_grid, max_indices)

grid, max_indices = index_grid_points(grid, grid_vecs, offset)


def generate_tetrahedra(grid, B1, B2, B3, shortest_diagonal, max_indices):
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

    [max_m, max_n, max_l] = max_indices

    number_of_points = len(grid)

    tetrahedra_by_point = [[] for x in range(number_of_points)]
    """list of list of ints: for each k point in the grid, a list of the 
    indices of each tetrahedron containing that k point is given."""

    for N in range(number_of_points):
        while np.allclose(grid[N], [0, 0, 0]) == False or N == 0:
            point1_index = N + 1
            point2_index = N + 2
            point3_index = N + 2 + max_l
            point4_index = N + 3 + max_l
            point5_index = N + 1 + (max_l + 1) * (max_n + 1)
            point6_index = N + 2 + (max_l + 1) * (max_n + 1)
            point7_index = N + 2 + max_l + (max_l + 1) * (max_n + 1)
            point8_index = N + 3 + max_l + (max_l + 1) * (max_n + 1)

            point1 = np.asarray(grid[N])
            """NumPy array: the coordinates of the first corner of a parallelepiped 
            from the grid."""
            point2, point3, point4, point5, point6, point7, point8 = \
                tetrahedron_method.determine_parallelepiped_corners(point1, B1, B2, B3)
            """tuple of NumPy arrays: the coordinates of the second through eighth 
            corners of a parallelepiped from the grid respectively."""

            if point2_index > number_of_points or not np.array_equal(grid[point2_index - 1], point2):
                break

            if point3_index > number_of_points or not np.array_equal(grid[point3_index - 1], point3):
                break

            if point4_index > number_of_points or not np.allclose(grid[point4_index - 1], point4):
                break

            if point5_index > number_of_points or not np.array_equal(grid[point5_index - 1], point5):
                break

            if point6_index > number_of_points or not np.array_equal(grid[point6_index - 1], point6):
                break

            if point7_index > number_of_points or not np.array_equal(grid[point7_index - 1], point7):
                break

            if point8_index > number_of_points or not np.array_equal(grid[point8_index - 1], point8):
                break

            # create quadruples for the corner points of each tetrahedra
            point_indices = np.array([point1_index, point2_index, point3_index,
                                      point4_index, point5_index, point6_index,
                                      point7_index, point8_index])
            """NumPy array: the indices of the corner points of a 
            parallelepiped in the grid."""
            tetrahedra_quadruples, tetrahedra_by_point = tetrahedron_method.add_tetrahedron(tetrahedra_quadruples,
                                                                       shortest_diagonal, point_indices, tetrahedra_by_point)
            break

    print(tetrahedra_by_point)

    return tetrahedra_quadruples

tetrahedra_quadruples = generate_tetrahedra(grid, grid_vecs[:,0], grid_vecs[:,1], grid_vecs[:,2], 3, max_indices)
end = time.time()
#print (end-start)

test_grid1 = [[0, 0, 0], [.5, 0, 1], [-.5, 1, 0], [1, 0, 0], [0, 1, 1],
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

test_grid_vecs = np.array([[1, -.5, .5], [0, 1, 0], [0, 0, 1]])

grid1, max_indices1 = index_grid_points(test_grid1, test_grid_vecs, np.array([0, 0, 0]))

def generate_tetrahedra1():
    # Checking if the correct tetrahedra quadruples are generated.
    return generate_tetrahedra(grid1, test_B1, test_B2,
        test_B3, test_shortest_diagonal, max_indices1)
"""== [[7, 8, 6, 2], [7, 8, 5, 2],
    [7, 4, 6, 2], [7, 3, 5, 2], [7, 1, 4, 2], [7, 1, 3, 2]]"""

#print(generate_tetrahedra1())


def cluster_tetrahedra_by_point(tetrahedra_quadruples, k):
    """For each k point in the grid, this function generates a list of 
    tetrahedra indices for each tetrahedron containing the k point.

    Args:
        tetrahedra_quadruples (list of lists of ints): a list of quadruples. 
            There is exactly one quadruple for every tetrahedron. Each 
            quadruple is a list of the grid_points indices for the corners of 
            the tetrahedron.
        k (int): the total number of points in the grid.

    Returns:
        tetrahedra_by_point (list of list of ints): for each k point in the 
            grid, a list of the indices of each tetrahedron containing that 
            k point is given.
    """

    tetrahedra_by_point = []
    """list of list of ints: for each k point in the grid, a list of the 
    indices of each tetrahedron containing that k point is given."""

    for m in range(k):
        adjacent_tetrahedra = []
        """list of ints: the indices of each tetrahedron containing the given k point in the grid."""

        # find all tetrahedra containing the k point
        for n in range(len(tetrahedra_quadruples)):
            for l in range(4):
                if tetrahedra_quadruples[n][l] == m + 1:
                    adjacent_tetrahedra.append(n + 1)

        tetrahedra_by_point.append(adjacent_tetrahedra)

    return tetrahedra_by_point