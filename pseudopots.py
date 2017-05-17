"""Construct pseudopotentials for various elements. This approach is taken from
`Grosso <http://www.sciencedirect.com/science/article/pii/B9780123850300000050>`_.
"""

import itertools
import numpy as np
from numpy.linalg import norm
from scipy import linalg
from BZI.symmetry import shells, make_ptvecs, make_rptvecs
from BZI.sampling import sphere_pts

angstrom_to_Bohr = 1.889725989

# Define the reciprocal lattice points that contribute.
def find_intvecs(a):
    """Find all integer vectors whose squared magnitude is a given value.
    
    Args:
        a (int): the squared magnitude of the desired vectors.
    
    Returns:
        vecs (list): a list of vectors with the provided squared magnitude
    """

    if a == 0:
        return [[0,0,0]]
    # Determine what integers to use for the components of the vectors.
    allowed_ints = []
    for i,j,k in itertools.product(range(-a, a), repeat=3):
        # Exit the loop if the component 
        if (i**2 + j**2 + k**2) == a:
            allowed_ints.append([i,j,k])
    return allowed_ints

#### Toy Pseudopotential ####
Toy_lat_type = "sc"
Toy_lat_const = 1.
Toy_lv = make_ptvecs(Toy_lat_type, Toy_lat_const) # toy lattice vectors

Toy_pff = [0.2]
Toy_shells = [[0.,0.,0], [0.,0.,1.]]
nested_shells = [shells(i, Toy_lv) for i in Toy_shells]
Toy_rlat_pts = np.array(list(itertools.chain(*nested_shells)))

# The number of contributing reciprocal lattice points determines the size
# of the Hamiltonian.
nToy = len(Toy_rlat_pts)

def ToyPP(kpt):
    """Evaluate a Toy pseudopotential at a given k-point.

    Args:
        kpoint (numpy.ndarray): a sampling point.

    Return:
        (numpy.ndarray): the sorted eigenvalues of the Hamiltonian at the provided
        sampling point.
    """
    
    # Initialize the Toy pseudopotential Hamiltonian.
    Toy_H = np.empty([nToy, nToy])

    # Construct the Toy Hamiltonian.
    for (i,k1) in enumerate(Toy_rlat_pts):
        for (j,k2) in enumerate(Toy_rlat_pts):
            if np.isclose(norm(k2 - k1), 1.) == True:
                Toy_H[i,j] = Toy_pff[0]
            elif i == j:
                Toy_H[i,j] = np.linalg.norm(kpt + k1)**2
            else:
                Toy_H[i,j] = 0.
                
    return np.sort(np.linalg.eigvals(Toy_H))

#### Free electron Pseudopotential ####
Free_lat_type = "sc"
Free_lat_const = 1.
Free_lv = make_ptvecs(Free_lat_type, Free_lat_const) # free lattice vectors

Free_pff = [0.2]
Free_shells = [[0.,0.,0], [0.,0.,1.]]
nested_shells = [shells(i, Free_lv) for i in Free_shells]
Free_rlat_pts = np.array(list(itertools.chain(*nested_shells)))

def FreePP(pt):
    """Evaluate the free-electron pseudopotential at a given point. The method
    employed here ignores band sorting issues.
    """
    return np.asarray([np.linalg.norm(pt + rpt)**2 for rpt in Free_rlat_pts])


#### W pseudopotentials ####
def W1(spt):
    """W1 is another toy model that we often work with. It is also convenient
    for unit testing since integrating it can be performed analytically.
    
    Args:
        spt (list or numpy.ndarray): a sampling point
    """
    return [np.product(np.exp([np.cos(2*np.pi*pt) for pt in spt]))]

def W2(spt):
    """W2 is another toy model. It is also convenient for unit testing since 
    integrating it can be performed analytically.
    
    Args:
        spt (list or numpy.ndarray): a sampling point
    """
    
    return[-np.cos(np.sum([np.cos(2*np.pi*pt) for pt in spt]))]

#### Pseudopotential of Al ####

# Define the pseudopotential form factors taken from Ashcroft for Al.
Al_pff = [0.0179, 0.0562]
Al_lat_const = 7.65339025545
Al_lat_type = "fcc"
Al_rlat_vecs = make_rptvecs(Al_lat_type, Al_lat_const)
Al_cutoff = 4*(2*np.pi/Al_lat_const)**2
Al_rlat_pts = sphere_pts(Al_rlat_vecs, Al_cutoff, [0,0,0])

# The number of contributing reciprocal lattice points determines the size
# of the Hamiltonian.
Al_size = len(Al_rlat_pts)
Ry_to_eV = 13.605698066
# Al_shift = 10.65942710893091139

def Al_PP(kpoint,neigvals):
    """Evaluate an Al pseudopotential at a given k-point. The pseudopotential
    form factors were taken from 
    `Ashcroft <http://journals.aps.org/pr/abstract/10.1103/PhysRev.116.555`_.

    Args:
        kpoint (numpy.array): a sampling point in k-space.

    Return:
        (numpy.array): the sorted eigenvalues of the Hamiltonian at the provided
        k-point.
    """
    # Initialize the Al Hamiltonian.
    Al_H = np.zeros([Al_size, Al_size])
    
    # Construct the Al Hamiltonian.
    for i in range(Al_size):
        k1 = np.asarray(Al_rlat_pts[i])
        for j in range(i + 1):
            k2 = np.asarray(Al_rlat_pts[j])
            n2 = norm(k2 - k1)**2
            if i == j:
                Al_H[i,j] = np.dot(kpoint + k1, kpoint + k1)
            elif np.isclose(n2, 3*(2*np.pi/Al_lat_const)**2) == True:
                Al_H[i,j] = Al_pff[0]
            elif np.isclose(n2, 4*(2*np.pi/Al_lat_const)**2) == True:
                Al_H[i,j] = Al_pff[1]
            else:
                continue
    return np.sort(np.linalg.eigvalsh(Al_H))[:neigvals]*Ry_to_eV # - Al_shift

def customAl_PP(kpoint, neigvals, Al_cutoff=21*(2*np.pi/Al_lat_const)**2,
               shift=True, matrix=False):
    """Evaluate an Al pseudopotential at a given k-point. The pseudopotential
    form factors were taken from Ashcroft.

    Args:
        kpoint (numpy.array): a sampling point in k-space.
        neigvals (int): the number of eigenvalues returned
        Al_cutoff (float): the cutoff energy for the expansion. This value 
            determines the size of the Hamiltonian.
        shift (bool): if true, the basis expansion will include points centered
            about the provided kpoint.
        matrix (bool): if true, the Hamiltonian is returned instead of its
            eigenvalues.

    Return:
        (numpy.array): the sorted eigenvalues of the Hamiltonian or the Hamiltonian
            itself at the the provided k-point.
    """

    if shift:
        Al_rlat_pts = sphere_pts(Al_rlat_vecs, Al_cutoff, kpoint)
    else:
        Al_rlat_pts = sphere_pts(Al_rlat_vecs, Al_cutoff, [0,0,0])
    Al_size = len(Al_rlat_pts)

    # Initialize the Al Hamiltonian.
    Al_H = np.zeros([Al_size, Al_size])
    
    # Populate the Al Hamiltonian.
    for i in range(Al_size):
        k1 = np.asarray(Al_rlat_pts[i])
        for j in range(i + 1):
            k2 = np.asarray(Al_rlat_pts[j])
            n2 = norm(k2 - k1)**2
            if i == j:
                Al_H[i,j] = np.dot(kpoint + k1, kpoint + k1)
            elif np.isclose(n2, 3*(2*np.pi/Al_lat_const)**2) == True:
                Al_H[i,j] = Al_pff[0]
            elif np.isclose(n2, 4*(2*np.pi/Al_lat_const)**2) == True:
                Al_H[i,j] = Al_pff[1]
            else:
                continue
    if matrix:
        return Al_H*Ry_to_eV # - Al_shift
    else:
        return np.sort(np.linalg.eigvalsh(Al_H))[:neigvals]*Ry_to_eV # - Al_shift

#### 
# Pseudo-potential from Band Structures and Pseudopotential Form Factors for
# Fourteen Semiconductors of the Diamond. and. Zinc-blende Structures* by
# Cohen and Bergstresser
#### 

# Define the pseudo-potential form factors taken from Cohen and Bergstesser
# V3S stands for the symmetric form factor for reciprocal lattice vectors of
# squared magnitude 3.
# V4A stands for the antisymmetric form factor for reciprocal lattice vectors
# of squared magnitude 4.
# These are assigned to new variables below and never get referenced.
#            V3S   V8S   V11S  V3A   V4A   V11A
Si_pff =   [-0.21, 0.04, 0.08, 0.00, 0.00, 0.00]
Ge_pff =   [-0.23, 0.01, 0.06, 0.00, 0.00, 0.00]
Sn_pff =   [-0.20, 0.00, 0.04, 0.00, 0.00, 0.00]
GaP_pff =  [-0.22, 0.03, 0.07, 0.12, 0.07, 0.02]
GaAs_pff = [-0.23, 0.01, 0.06, 0.07, 0.05, 0.01]
AlSb_pff = [-0.21, 0.02, 0.06, 0.06, 0.04, 0.02]
InP_pff =  [-0.23, 0.01, 0.06, 0.07, 0.05, 0.01]
GaSb_pff = [-0.22, 0.00, 0.05, 0.06, 0.05, 0.01]
InAs_pff = [-0.22, 0.00, 0.05, 0.08, 0.05, 0.03]
InSb_pff = [-0.20, 0.00, 0.04, 0.06, 0.05, 0.01]
ZnS_pff =  [-0.22, 0.03, 0.07, 0.24, 0.14, 0.04]
ZnSe_pff = [-0.23, 0.01, 0.06, 0.18, 0.12, 0.03]
ZnTe_pff = [-0.22, 0.00, 0.05, 0.13, 0.10, 0.01]
CdTe_pff = [-0.20, 0.00, 0.04, 0.15, 0.09, 0.04]    

#### Pseudopotential of Si ####
Si_lat_type = "fcc"
Si_lat_const = 5.43*angstrom_to_Bohr # the lattice constant in Bohr
Si_rlat_vecs = make_rptvecs(Si_lat_type, Si_lat_const)
Si_tau = Si_lat_const/8.*np.array([1,1,1]) # one atomic basis vector

# The pseudopotential form factors of Si
Si_pff =   [-0.21, 0.04, 0.08]

# The cutoff energy for the fourier expansion of wavefunction
Si_r = 11.*(2*np.pi/Si_lat_const)**2
Si_rlat_pts = sphere_pts(Si_rlat_vecs, Si_r, [0,0,0]) # reciprocal lattice points
Si_size = len(Si_rlat_pts)
# Si_shift = 10.4864461

def Si_PP(kpoint, neigvals):
    """Find the eigenvalues of a Hamiltonian matrix built to match the
    band structure of Si.
    
    Args:
        kpoint (list or numpy.ndarray): a point in k-space.
        neigvals (int): the number of returned eigenvalues in increasing
            order.
        
    Returns:
        eigvals (np.ndarray): the eigenvalues of the Hamiltonian matrix.
    """

    # Initialize the Hamiltonian.
    H = np.zeros((Si_size, Si_size),dtype=complex)

    # Populate the Hamiltonian.
    for i in range(Si_size):
        for j in range(i + 1):
            h = (Si_rlat_pts[j] - Si_rlat_pts[i])
            if i == j:
                H[i,j] = np.dot(kpoint + Si_rlat_pts[i], kpoint + Si_rlat_pts[i])
            if np.isclose(np.dot(h,h), 3*(2*np.pi/Si_lat_const)**2):
                H[i,j] = Si_pff[0]*np.cos(np.dot(h,Si_tau))
            if np.isclose(np.dot(h,h), 8*(2*np.pi/Si_lat_const)**2):
                H[i,j] = Si_pff[1]*np.cos(np.dot(h,Si_tau))
            if np.isclose(np.dot(h,h), 11*(2*np.pi/Si_lat_const)**2):
                H[i,j] = Si_pff[2]*np.cos(np.dot(h,Si_tau))
            else:
                continue
            
    return np.linalg.eigvalsh(H)[:neigvals]*Ry_to_eV # - Si_shift

def customSi_PP(kpoint, neigvals, Si_cutoff, shift=True, matrix=False):
    """Evaluate a Si pseudopotential at a given k-point. The pseudopotential
    form factors were taken from Cohen.

    Args:
        kpoint (numpy.array): a sampling point in k-space.
        neigvals (int): the number of eigenvalues returned
        Al_cutoff (float): the cutoff energy for the expansion. This value 
            determines the size of the Hamiltonian.
        shift (bool): if true, the basis expansion will include points centered
            about the provided kpoint.
        matrix (bool): if true, the Hamiltonian is returned instead of its
            eigenvalues.

    Return:
        (numpy.array): the sorted eigenvalues of the Hamiltonian or the Hamiltonian
            itself at the the provided k-point.
    """

    if shift:
        Si_rlat_pts = sphere_pts(Si_rlat_vecs, Si_r, kpoint) # reciprocal lattice points
    else:
        Si_rlat_pts = sphere_pts(Si_rlat_vecs, Si_r, [0,0,0]) # reciprocal lattice points

    Si_size = len(Si_rlat_pts)
    H = np.zeros((Si_size, Si_size),dtype=complex)
    
    for i in range(Si_size):
        for j in range(i + 1):
            h = (Si_rlat_pts[j] - Si_rlat_pts[i])
            if i == j:
                H[i,j] = np.dot(kpoint + Si_rlat_pts[i], kpoint + Si_rlat_pts[i])
            if np.isclose(np.dot(h,h), 3*(2*np.pi/Si_lat_const)**2):
                H[i,j] = Si_pff[0]*np.cos(np.dot(h,Si_tau))
            if np.isclose(np.dot(h,h), 8*(2*np.pi/Si_lat_const)**2):
                H[i,j] = Si_pff[1]*np.cos(np.dot(h,Si_tau))
            if np.isclose(np.dot(h,h), 11*(2*np.pi/Si_lat_const)**2):
                H[i,j] = Si_pff[2]*np.cos(np.dot(h,Si_tau))
            else:
                continue
    if matrix:
        return H*Ry_to_eV # - Si_shift
    else:
        return np.linalg.eigvalsh(H)[:neigvals]*Ry_to_eV # - Si_shift


#### Pseudopotential of Ge ####
Ge_lat_type = "fcc"
Ge_lat_const = 5.66*angstrom_to_Bohr # the lattice constant in Bohr
Ge_rlat_vecs = make_rptvecs(Ge_lat_type, Ge_lat_const)
Ge_tau = Ge_lat_const/8.*np.array([1,1,1]) # one atomic basis vector

# The pseudopotential form factors of Ge
Ge_pff =   [-0.23, 0.01, 0.06]

# The cutoff energy for the fourier expansion of wavefunction
Ge_r = 11.*(2*np.pi/Ge_lat_const)**2
Ge_rlat_pts = sphere_pts(Ge_rlat_vecs, Ge_r, [0,0,0]) # reciprocal lattice points
Ge_size = len(Ge_rlat_pts)
# Ge_shift = 10.4864461

def Ge_PP(kpoint, neigvals):
    """Find the eigenvalues of a Hamiltonian matrix built to match the
    band structure of Ge.
    
    Args:
        kpoint (list or numpy.ndarray): a point in k-space.
        neigvals (int): the number of returned eigenvalues in increasing
            order.
        
    Returns:
        eigvals (np.ndarray): the eigenvalues of the Hamiltonian matrix.
    """

    # Initialize the Hamiltonian.
    H = np.zeros((Ge_size, Ge_size),dtype=complex)

    # Populate the Hamiltonian.
    for i in range(Ge_size):
        for j in range(i + 1):
            h = (Ge_rlat_pts[j] - Ge_rlat_pts[i])
            if i == j:
                H[i,j] = np.dot(kpoint + Ge_rlat_pts[i], kpoint + Ge_rlat_pts[i])
            if np.isclose(np.dot(h,h), 3*(2*np.pi/Ge_lat_const)**2):
                H[i,j] = Ge_pff[0]*np.cos(np.dot(h,Ge_tau))
            if np.isclose(np.dot(h,h), 8*(2*np.pi/Ge_lat_const)**2):
                H[i,j] = Ge_pff[1]*np.cos(np.dot(h,Ge_tau))
            if np.isclose(np.dot(h,h), 11*(2*np.pi/Ge_lat_const)**2):
                H[i,j] = Ge_pff[2]*np.cos(np.dot(h,Ge_tau))
            else:
                continue
            
    return np.linalg.eigvalsh(H)[:neigvals]*Ry_to_eV # - Ge_shift

def customGe_PP(kpoint, neigvals, Ge_cutoff, shift=True, matrix=False):
    """Evaluate a Ge pseudopotential at a given k-point. The pseudopotential
    form factors were taken from Cohen.

    Args:
        kpoint (numpy.array): a sampling point in k-space.
        neigvals (int): the number of eigenvalues returned
        Al_cutoff (float): the cutoff energy for the expansion. This value 
            determines the size of the Hamiltonian.
        shift (bool): if true, the basis expansion will include points centered
            about the provided kpoint.
        matrix (bool): if true, the Hamiltonian is returned instead of its
            eigenvalues.

    Return:
        (numpy.array): the sorted eigenvalues of the Hamiltonian or the Hamiltonian
            itself at the the provided k-point.
    """

    if shift:
        Ge_rlat_pts = sphere_pts(Ge_rlat_vecs, Ge_r, kpoint) # reciprocal lattice points
    else:
        Ge_rlat_pts = sphere_pts(Ge_rlat_vecs, Ge_r, [0,0,0]) # reciprocal lattice points

    Ge_size = len(Ge_rlat_pts)
    H = np.zeros((Ge_size, Ge_size),dtype=complex)
    
    for i in range(Ge_size):
        for j in range(i + 1):
            h = (Ge_rlat_pts[j] - Ge_rlat_pts[i])
            if i == j:
                H[i,j] = np.dot(kpoint + Ge_rlat_pts[i], kpoint + Ge_rlat_pts[i])
            if np.isclose(np.dot(h,h), 3*(2*np.pi/Ge_lat_const)**2):
                H[i,j] = Ge_pff[0]*np.cos(np.dot(h,Ge_tau))
            if np.isclose(np.dot(h,h), 8*(2*np.pi/Ge_lat_const)**2):
                H[i,j] = Ge_pff[1]*np.cos(np.dot(h,Ge_tau))
            if np.isclose(np.dot(h,h), 11*(2*np.pi/Ge_lat_const)**2):
                H[i,j] = Ge_pff[2]*np.cos(np.dot(h,Ge_tau))
            else:
                continue
    if matrix:
        return H*Ry_to_eV # - Ge_shift
    else:
        return np.linalg.eigvalsh(H)[:neigvals]*Ry_to_eV # - Ge_shift


#### Pseudopotential of Sn ####
Sn_lat_type = "fcc"
Sn_lat_const = 6.49*angstrom_to_Bohr # the lattice constant in Bohr
Sn_rlat_vecs = make_rptvecs(Sn_lat_type, Sn_lat_const)
Sn_tau = Sn_lat_const/8.*np.array([1,1,1]) # one atomic basis vector

# The pseudopotential form factors of Sn
Sn_pff =   [-0.20, 0.00, 0.04]

# The cutoff energy for the fourier expansion of wavefunction
Sn_r = 11.*(2*np.pi/Sn_lat_const)**2
Sn_rlat_pts = sphere_pts(Sn_rlat_vecs, Sn_r, [0,0,0]) # reciprocal lattice points
Sn_size = len(Sn_rlat_pts)
# Sn_shift = 10.4864461

def Sn_PP(kpoint, neigvals):
    """Find the eigenvalues of a Hamiltonian matrix built to match the
    band structure of Sn.
    
    Args:
        kpoint (list or numpy.ndarray): a point in k-space.
        neigvals (int): the number of returned eigenvalues in increasing
            order.
        
    Returns:
        eigvals (np.ndarray): the eigenvalues of the Hamiltonian matrix.
    """

    # Initialize the Hamiltonian.
    H = np.zeros((Sn_size, Sn_size),dtype=complex)

    # Populate the Hamiltonian.
    for i in range(Sn_size):
        for j in range(i + 1):
            h = (Sn_rlat_pts[j] - Sn_rlat_pts[i])
            if i == j:
                H[i,j] = np.dot(kpoint + Sn_rlat_pts[i], kpoint + Sn_rlat_pts[i])
            if np.isclose(np.dot(h,h), 3*(2*np.pi/Sn_lat_const)**2):
                H[i,j] = Sn_pff[0]*np.cos(np.dot(h,Sn_tau))
            if np.isclose(np.dot(h,h), 11*(2*np.pi/Sn_lat_const)**2):
                H[i,j] = Sn_pff[2]*np.cos(np.dot(h,Sn_tau))
            else:
                continue
            
    return np.linalg.eigvalsh(H)[:neigvals]*Ry_to_eV # - Sn_shift

def customSn_PP(kpoint, neigvals, Sn_cutoff, shift=True, matrix=False):
    """Evaluate a Sn pseudopotential at a given k-point. The pseudopotential
    form factors were taken from Cohen.

    Args:
        kpoint (numpy.array): a sampling point in k-space.
        neigvals (int): the number of eigenvalues returned
        Al_cutoff (float): the cutoff energy for the expansion. This value 
            determines the size of the Hamiltonian.
        shift (bool): if true, the basis expansion will include points centered
            about the provided kpoint.
        matrix (bool): if true, the Hamiltonian is returned instead of its
            eigenvalues.

    Return:
        (numpy.array): the sorted eigenvalues of the Hamiltonian or the Hamiltonian
            itself at the the provided k-point.
    """

    if shift:
        Sn_rlat_pts = sphere_pts(Sn_rlat_vecs, Sn_r, kpoint) # reciprocal lattice points
    else:
        Sn_rlat_pts = sphere_pts(Sn_rlat_vecs, Sn_r, [0,0,0]) # reciprocal lattice points

    Sn_size = len(Sn_rlat_pts)
    H = np.zeros((Sn_size, Sn_size),dtype=complex)
    
    for i in range(Sn_size):
        for j in range(i + 1):
            h = (Sn_rlat_pts[j] - Sn_rlat_pts[i])
            if i == j:
                H[i,j] = np.dot(kpoint + Sn_rlat_pts[i], kpoint + Sn_rlat_pts[i])
            if np.isclose(np.dot(h,h), 3*(2*np.pi/Sn_lat_const)**2):
                H[i,j] = Sn_pff[0]*np.cos(np.dot(h,Sn_tau))
            if np.isclose(np.dot(h,h), 11*(2*np.pi/Sn_lat_const)**2):
                H[i,j] = Sn_pff[2]*np.cos(np.dot(h,Sn_tau))
            else:
                continue
    if matrix:
        return H*Ry_to_eV # - Sn_shift
    else:
        return np.linalg.eigvalsh(H)[:neigvals]*Ry_to_eV # - Sn_shift


#### Pseudopotential of GaP ####
GaP_lat_type = "fcc"
GaP_lat_const = 5.44*angstrom_to_Bohr # the lattice constant in Bohr
GaP_rlat_vecs = make_rptvecs(GaP_lat_type, GaP_lat_const)
GaP_tau = GaP_lat_const/8.*np.array([1,1,1]) # one atomic basis vector

# The pseudopotential form factors of GaP
sGaP_pff =   [-0.22, 0.03, 0.07] # symmetric
aGaP_pff =  [0.12, 0.07, 0.02] # anti-symmetric


# The cutoff energy for the fourier expansion of wavefunction
GaP_r = 11.*(2*np.pi/GaP_lat_const)**2
GaP_rlat_pts = sphere_pts(GaP_rlat_vecs, GaP_r, [0,0,0]) # reciprocal lattice points
GaP_size = len(GaP_rlat_pts)
# GaP_shift = 10.4864461

def GaP_PP(kpoint, neigvals):
    """Find the eigenvalues of a Hamiltonian matrix built to match the
    band structure of GaP.
    
    Args:
        kpoint (list or numpy.ndarray): a point in k-space.
        neigvals (int): the number of returned eigenvalues in increasing
            order.
        
    Returns:
        eigvals (np.ndarray): the eigenvalues of the Hamiltonian matrix.
    """

    # Initialize the Hamiltonian.
    H = np.zeros((GaP_size, GaP_size),dtype=complex)

    # Populate the Hamiltonian.
    for i in range(GaP_size):
        for j in range(i + 1):
            h = (GaP_rlat_pts[j] - GaP_rlat_pts[i])
            if i == j:
                H[i,j] = np.dot(kpoint + GaP_rlat_pts[i], kpoint + GaP_rlat_pts[i])
            if np.isclose(np.dot(h,h), 3*(2*np.pi/GaP_lat_const)**2):
                H[i,j] = sGaP_pff[0]*np.cos(np.dot(h,GaP_tau)) + (
                    1j*aGaP_pff[0]*np.sin(np.dot(h,GaP_tau)))
            if np.isclose(np.dot(h,h), 4*(2*np.pi/GaP_lat_const)**2):
                H[i,j] = 1j*GaP_pff[0]*np.cos(np.dot(h,GaP_tau))
            if np.isclose(np.dot(h,h), 8*(2*np.pi/GaP_lat_const)**2):
                H[i,j] = GaP_pff[1]*np.cos(np.dot(h,GaP_tau))
            if np.isclose(np.dot(h,h), 11*(2*np.pi/GaP_lat_const)**2):
                H[i,j] = sGaP_pff[2]*np.cos(np.dot(h,GaP_tau)) + (
                    1j*aGaP_pff[2]*np.sin(np.dot(h,GaP_tau)))
            else:
                continue
            
    return np.linalg.eigvalsh(H)[:neigvals]*Ry_to_eV # - GaP_shift

def customGaP_PP(kpoint, neigvals, GaP_cutoff, shift=True, matrix=False):
    """Evaluate a GaP pseudopotential at a given k-point. The pseudopotential
    form factors were taken from Cohen.

    Args:
        kpoint (numpy.array): a sampling point in k-space.
        neigvals (int): the number of eigenvalues returned
        Al_cutoff (float): the cutoff energy for the expansion. This value 
            determines the size of the Hamiltonian.
        shift (bool): if true, the basis expansion will include points centered
            about the provided kpoint.
        matrix (bool): if true, the Hamiltonian is returned instead of its
            eigenvalues.

    Return:
        (numpy.array): the sorted eigenvalues of the Hamiltonian or the Hamiltonian
            itself at the the provided k-point.
    """

    if shift:
        GaP_rlat_pts = sphere_pts(GaP_rlat_vecs, GaP_r, kpoint) # reciprocal lattice points
    else:
        GaP_rlat_pts = sphere_pts(GaP_rlat_vecs, GaP_r, [0,0,0]) # reciprocal lattice points

    GaP_size = len(GaP_rlat_pts)
    H = np.zeros((GaP_size, GaP_size),dtype=complex)
    
    for i in range(GaP_size):
        for j in range(i + 1):
            h = (GaP_rlat_pts[j] - GaP_rlat_pts[i])
            if i == j:
                H[i,j] = np.dot(kpoint + GaP_rlat_pts[i], kpoint + GaP_rlat_pts[i])
            if np.isclose(np.dot(h,h), 3*(2*np.pi/GaP_lat_const)**2):
                H[i,j] = sGaP_pff[0]*np.cos(np.dot(h,GaP_tau)) + (
                    1j*aGaP_pff[0]*np.sin(np.dot(h,GaP_tau)))
            if np.isclose(np.dot(h,h), 4*(2*np.pi/GaP_lat_const)**2):
                H[i,j] = GaP_pff[0]*np.cos(np.dot(h,GaP_tau))
            if np.isclose(np.dot(h,h), 8*(2*np.pi/GaP_lat_const)**2):
                H[i,j] = GaP_pff[1]*np.cos(np.dot(h,GaP_tau))
            if np.isclose(np.dot(h,h), 11*(2*np.pi/GaP_lat_const)**2):
                H[i,j] = sGaP_pff[2]*np.cos(np.dot(h,GaP_tau)) + (
                    1j*aGaP_pff[2]*np.sin(np.dot(h,GaP_tau)))
            else:
                continue
    if matrix:
        return H*Ry_to_eV # - GaP_shift
    else:
        return np.linalg.eigvalsh(H)[:neigvals]*Ry_to_eV # - GaP_shift    

#### Pseudopotential of GaAs ####
GaAs_lat_type = "fcc"
GaAs_lat_const = 5.64*angstrom_to_Bohr # the lattice constant in Bohr
GaAs_rlat_vecs = make_rptvecs(GaAs_lat_type, GaAs_lat_const)
GaAs_tau = GaAs_lat_const/8.*np.array([1,1,1]) # one atomic basis vector

# The pseudopotential form factors of GaAs
sGaAs_pff =   [-0.23, 0.01, 0.06] # symmetric
aGaAs_pff =  [0.07, 0.05, 0.01] # anti-symmetric


# The cutoff energy for the fourier expansion of wavefunction
GaAs_r = 11.*(2*np.pi/GaAs_lat_const)**2
GaAs_rlat_pts = sphere_pts(GaAs_rlat_vecs, GaAs_r, [0,0,0]) # reciprocal lattice points
GaAs_size = len(GaAs_rlat_pts)
# GaAs_shift = 10.4864461

def GaAs_PP(kpoint, neigvals):
    """Find the eigenvalues of a Hamiltonian matrix built to match the
    band structure of GaAs.
    
    Args:
        kpoint (list or numpy.ndarray): a point in k-space.
        neigvals (int): the number of returned eigenvalues in increasing
            order.
        
    Returns:
        eigvals (np.ndarray): the eigenvalues of the Hamiltonian matrix.
    """

    # Initialize the Hamiltonian.
    H = np.zeros((GaAs_size, GaAs_size),dtype=complex)

    # Populate the Hamiltonian.
    for i in range(GaAs_size):
        for j in range(i + 1):
            h = (GaAs_rlat_pts[j] - GaAs_rlat_pts[i])
            if i == j:
                H[i,j] = np.dot(kpoint + GaAs_rlat_pts[i], kpoint + GaAs_rlat_pts[i])
            if np.isclose(np.dot(h,h), 3*(2*np.pi/GaAs_lat_const)**2):
                H[i,j] = sGaAs_pff[0]*np.cos(np.dot(h,GaAs_tau)) + (
                    1j*aGaAs_pff[0]*np.sin(np.dot(h,GaAs_tau)))
            if np.isclose(np.dot(h,h), 4*(2*np.pi/GaAs_lat_const)**2):
                H[i,j] = GaAs_pff[0]*np.cos(np.dot(h,GaAs_tau))
            if np.isclose(np.dot(h,h), 8*(2*np.pi/GaAs_lat_const)**2):
                H[i,j] = GaAs_pff[1]*np.cos(np.dot(h,GaAs_tau))
            if np.isclose(np.dot(h,h), 11*(2*np.pi/GaAs_lat_const)**2):
                H[i,j] = sGaAs_pff[2]*np.cos(np.dot(h,GaAs_tau)) + (
                    1j*aGaAs_pff[2]*np.sin(np.dot(h,GaAs_tau)))
            else:
                continue
            
    return np.linalg.eigvalsh(H)[:neigvals]*Ry_to_eV # - GaAs_shift

def customGaAs_PP(kpoint, neigvals, GaAs_cutoff, shift=True, matrix=False):
    """Evaluate a GaAs pseudopotential at a given k-point. The pseudopotential
    form factors were taken from Cohen.

    Args:
        kpoint (numpy.array): a sampling point in k-space.
        neigvals (int): the number of eigenvalues returned
        Al_cutoff (float): the cutoff energy for the expansion. This value 
            determines the size of the Hamiltonian.
        shift (bool): if true, the basis expansion will include points centered
            about the provided kpoint.
        matrix (bool): if true, the Hamiltonian is returned instead of its
            eigenvalues.

    Return:
        (numpy.array): the sorted eigenvalues of the Hamiltonian or the Hamiltonian
            itself at the the provided k-point.
    """

    if shift:
        GaAs_rlat_pts = sphere_pts(GaAs_rlat_vecs, GaAs_r, kpoint) # reciprocal lattice points
    else:
        GaAs_rlat_pts = sphere_pts(GaAs_rlat_vecs, GaAs_r, [0,0,0]) # reciprocal lattice points

    GaAs_size = len(GaAs_rlat_pts)
    H = np.zeros((GaAs_size, GaAs_size),dtype=complex)
    
    for i in range(GaAs_size):
        for j in range(i + 1):
            h = (GaAs_rlat_pts[j] - GaAs_rlat_pts[i])
            if i == j:
                H[i,j] = np.dot(kpoint + GaAs_rlat_pts[i], kpoint + GaAs_rlat_pts[i])
            if np.isclose(np.dot(h,h), 3*(2*np.pi/GaAs_lat_const)**2):
                H[i,j] = sGaAs_pff[0]*np.cos(np.dot(h,GaAs_tau)) + (
                    1j*aGaAs_pff[0]*np.sin(np.dot(h,GaAs_tau)))
            if np.isclose(np.dot(h,h), 4*(2*np.pi/GaAs_lat_const)**2):
                H[i,j] = GaAs_pff[0]*np.cos(np.dot(h,GaAs_tau))
            if np.isclose(np.dot(h,h), 8*(2*np.pi/GaAs_lat_const)**2):
                H[i,j] = GaAs_pff[1]*np.cos(np.dot(h,GaAs_tau))
            if np.isclose(np.dot(h,h), 11*(2*np.pi/GaAs_lat_const)**2):
                H[i,j] = sGaAs_pff[2]*np.cos(np.dot(h,GaAs_tau)) + (
                    1j*aGaAs_pff[2]*np.sin(np.dot(h,GaAs_tau)))
            else:
                continue
    if matrix:
        return H*Ry_to_eV # - GaAs_shift
    else:
        return np.linalg.eigvalsh(H)[:neigvals]*Ry_to_eV # - GaAs_shift

#### Pseudopotential of AlSb ####
AlSb_lat_type = "fcc"
AlSb_lat_const = 6.13*angstrom_to_Bohr # the lattice constant in Bohr
AlSb_rlat_vecs = make_rptvecs(AlSb_lat_type, AlSb_lat_const)
AlSb_tau = AlSb_lat_const/8.*np.array([1,1,1]) # one atomic basis vector

# The pseudopotential form factors of AlSb
sAlSb_pff =   [-0.21, 0.02, 0.06] # symmetric
aAlSb_pff =  [0.06, 0.04, 0.02] # anti-symmetric


# The cutoff energy for the fourier expansion of wavefunction
AlSb_r = 11.*(2*np.pi/AlSb_lat_const)**2
AlSb_rlat_pts = sphere_pts(AlSb_rlat_vecs, AlSb_r, [0,0,0]) # reciprocal lattice points
AlSb_size = len(AlSb_rlat_pts)

def AlSb_PP(kpoint, neigvals):
    """Find the eigenvalues of a Hamiltonian matrix built to match the
    band structure of AlSb.
    
    Args:
        kpoint (list or numpy.ndarray): a point in k-space.
        neigvals (int): the number of returned eigenvalues in increasing
            order.
        
    Returns:
        eigvals (np.ndarray): the eigenvalues of the Hamiltonian matrix.
    """

    # Initialize the Hamiltonian.
    H = np.zeros((AlSb_size, AlSb_size),dtype=complex)

    # Populate the Hamiltonian.
    for i in range(AlSb_size):
        for j in range(i + 1):
            h = (AlSb_rlat_pts[j] - AlSb_rlat_pts[i])
            if i == j:
                H[i,j] = np.dot(kpoint + AlSb_rlat_pts[i], kpoint + AlSb_rlat_pts[i])
            if np.isclose(np.dot(h,h), 3*(2*np.pi/AlSb_lat_const)**2):
                H[i,j] = sAlSb_pff[0]*np.cos(np.dot(h,AlSb_tau)) + (
                    1j*aAlSb_pff[0]*np.sin(np.dot(h,AlSb_tau)))
            if np.isclose(np.dot(h,h), 4*(2*np.pi/AlSb_lat_const)**2):
                H[i,j] = AlSb_pff[0]*np.cos(np.dot(h,AlSb_tau))
            if np.isclose(np.dot(h,h), 8*(2*np.pi/AlSb_lat_const)**2):
                H[i,j] = AlSb_pff[1]*np.cos(np.dot(h,AlSb_tau))
            if np.isclose(np.dot(h,h), 11*(2*np.pi/AlSb_lat_const)**2):
                H[i,j] = sAlSb_pff[2]*np.cos(np.dot(h,AlSb_tau)) + (
                    1j*aAlSb_pff[2]*np.sin(np.dot(h,AlSb_tau)))
            else:
                continue
            
    return np.linalg.eigvalsh(H)[:neigvals]*Ry_to_eV # - AlSb_shift

def customAlSb_PP(kpoint, neigvals, AlSb_cutoff, shift=True, matrix=False):
    """Evaluate a AlSb pseudopotential at a given k-point. The pseudopotential
    form factors were taken from Cohen.

    Args:
        kpoint (numpy.array): a sampling point in k-space.
        neigvals (int): the number of eigenvalues returned
        Al_cutoff (float): the cutoff energy for the expansion. This value 
            determines the size of the Hamiltonian.
        shift (bool): if true, the basis expansion will include points centered
            about the provided kpoint.
        matrix (bool): if true, the Hamiltonian is returned instead of its
            eigenvalues.

    Return:
        (numpy.array): the sorted eigenvalues of the Hamiltonian or the Hamiltonian
            itself at the the provided k-point.
    """

    if shift:
        AlSb_rlat_pts = sphere_pts(AlSb_rlat_vecs, AlSb_r, kpoint) # reciprocal lattice points
    else:
        AlSb_rlat_pts = sphere_pts(AlSb_rlat_vecs, AlSb_r, [0,0,0]) # reciprocal lattice points

    AlSb_size = len(AlSb_rlat_pts)
    H = np.zeros((AlSb_size, AlSb_size),dtype=complex)
    
    for i in range(AlSb_size):
        for j in range(i + 1):
            h = (AlSb_rlat_pts[j] - AlSb_rlat_pts[i])
            if i == j:
                H[i,j] = np.dot(kpoint + AlSb_rlat_pts[i], kpoint + AlSb_rlat_pts[i])
            if np.isclose(np.dot(h,h), 3*(2*np.pi/AlSb_lat_const)**2):
                H[i,j] = sAlSb_pff[0]*np.cos(np.dot(h,AlSb_tau)) + (
                    1j*aAlSb_pff[0]*np.sin(np.dot(h,AlSb_tau)))
            if np.isclose(np.dot(h,h), 4*(2*np.pi/AlSb_lat_const)**2):
                H[i,j] = AlSb_pff[0]*np.cos(np.dot(h,AlSb_tau))
            if np.isclose(np.dot(h,h), 8*(2*np.pi/AlSb_lat_const)**2):
                H[i,j] = AlSb_pff[1]*np.cos(np.dot(h,AlSb_tau))
            if np.isclose(np.dot(h,h), 11*(2*np.pi/AlSb_lat_const)**2):
                H[i,j] = sAlSb_pff[2]*np.cos(np.dot(h,AlSb_tau)) + (
                    1j*aAlSb_pff[2]*np.sin(np.dot(h,AlSb_tau)))
            else:
                continue
    if matrix:
        return H*Ry_to_eV # - AlSb_shift
    else:
        return np.linalg.eigvalsh(H)[:neigvals]*Ry_to_eV # - AlSb_shift    

#### Pseudopotential of InP ####
InP_lat_type = "fcc"
InP_lat_const = 5.86*angstrom_to_Bohr # the lattice constant in Bohr
InP_rlat_vecs = make_rptvecs(InP_lat_type, InP_lat_const)
InP_tau = InP_lat_const/8.*np.array([1,1,1]) # one atomic basis vector

# The pseudopotential form factors of InP
sInP_pff =   [-0.23, 0.01, 0.06] # symmetric
aInP_pff =  [0.07, 0.05, 0.01] # anti-symmetric


# The cutoff energy for the fourier expansion of wavefunction
InP_r = 11.*(2*np.pi/InP_lat_const)**2
InP_rlat_pts = sphere_pts(InP_rlat_vecs, InP_r, [0,0,0]) # reciprocal lattice points
InP_size = len(InP_rlat_pts)

def InP_PP(kpoint, neigvals):
    """Find the eigenvalues of a Hamiltonian matrix built to match the
    band structure of InP.
    
    Args:
        kpoint (list or numpy.ndarray): a point in k-space.
        neigvals (int): the number of returned eigenvalues in increasing
            order.
        
    Returns:
        eigvals (np.ndarray): the eigenvalues of the Hamiltonian matrix.
    """

    # Initialize the Hamiltonian.
    H = np.zeros((InP_size, InP_size),dtype=complex)

    # Populate the Hamiltonian.
    for i in range(InP_size):
        for j in range(i + 1):
            h = (InP_rlat_pts[j] - InP_rlat_pts[i])
            if i == j:
                H[i,j] = np.dot(kpoint + InP_rlat_pts[i], kpoint + InP_rlat_pts[i])
            if np.isclose(np.dot(h,h), 3*(2*np.pi/InP_lat_const)**2):
                H[i,j] = sInP_pff[0]*np.cos(np.dot(h,InP_tau)) + (
                    1j*aInP_pff[0]*np.sin(np.dot(h,InP_tau)))
            if np.isclose(np.dot(h,h), 4*(2*np.pi/InP_lat_const)**2):
                H[i,j] = InP_pff[0]*np.cos(np.dot(h,InP_tau))
            if np.isclose(np.dot(h,h), 8*(2*np.pi/InP_lat_const)**2):
                H[i,j] = InP_pff[1]*np.cos(np.dot(h,InP_tau))
            if np.isclose(np.dot(h,h), 11*(2*np.pi/InP_lat_const)**2):
                H[i,j] = sInP_pff[2]*np.cos(np.dot(h,InP_tau)) + (
                    1j*aInP_pff[2]*np.sin(np.dot(h,InP_tau)))
            else:
                continue
            
    return np.linalg.eigvalsh(H)[:neigvals]*Ry_to_eV # - InP_shift

def customInP_PP(kpoint, neigvals, InP_cutoff, shift=True, matrix=False):
    """Evaluate a InP pseudopotential at a given k-point. The pseudopotential
    form factors were taken from Cohen.

    Args:
        kpoint (numpy.array): a sampling point in k-space.
        neigvals (int): the number of eigenvalues returned
        Al_cutoff (float): the cutoff energy for the expansion. This value 
            determines the size of the Hamiltonian.
        shift (bool): if true, the basis expansion will include points centered
            about the provided kpoint.
        matrix (bool): if true, the Hamiltonian is returned instead of its
            eigenvalues.

    Return:
        (numpy.array): the sorted eigenvalues of the Hamiltonian or the Hamiltonian
            itself at the the provided k-point.
    """

    if shift:
        InP_rlat_pts = sphere_pts(InP_rlat_vecs, InP_r, kpoint) # reciprocal lattice points
    else:
        InP_rlat_pts = sphere_pts(InP_rlat_vecs, InP_r, [0,0,0]) # reciprocal lattice points

    InP_size = len(InP_rlat_pts)
    H = np.zeros((InP_size, InP_size),dtype=complex)
    
    for i in range(InP_size):
        for j in range(i + 1):
            h = (InP_rlat_pts[j] - InP_rlat_pts[i])
            if i == j:
                H[i,j] = np.dot(kpoint + InP_rlat_pts[i], kpoint + InP_rlat_pts[i])
            if np.isclose(np.dot(h,h), 3*(2*np.pi/InP_lat_const)**2):
                H[i,j] = sInP_pff[0]*np.cos(np.dot(h,InP_tau)) + (
                    1j*aInP_pff[0]*np.sin(np.dot(h,InP_tau)))
            if np.isclose(np.dot(h,h), 4*(2*np.pi/InP_lat_const)**2):
                H[i,j] = InP_pff[0]*np.cos(np.dot(h,InP_tau))
            if np.isclose(np.dot(h,h), 8*(2*np.pi/InP_lat_const)**2):
                H[i,j] = InP_pff[1]*np.cos(np.dot(h,InP_tau))
            if np.isclose(np.dot(h,h), 11*(2*np.pi/InP_lat_const)**2):
                H[i,j] = sInP_pff[2]*np.cos(np.dot(h,InP_tau)) + (
                    1j*aInP_pff[2]*np.sin(np.dot(h,InP_tau)))
            else:
                continue
    if matrix:
        return H*Ry_to_eV # - InP_shift
    else:
        return np.linalg.eigvalsh(H)[:neigvals]*Ry_to_eV # - InP_shift    

#### Pseudopotential of GaSb ####
GaSb_lat_type = "fcc"
GaSb_lat_const = 6.12*angstrom_to_Bohr # the lattice constant in Bohr
GaSb_rlat_vecs = make_rptvecs(GaSb_lat_type, GaSb_lat_const)
GaSb_tau = GaSb_lat_const/8.*np.array([1,1,1]) # one atomic basis vector

# The pseudopotential form factors of GaSb
sGaSb_pff =   [-0.22, 0.00, 0.05] # symmetric
aGaSb_pff =  [0.06, 0.05, 0.01] # anti-symmetric


# The cutoff energy for the fourier expansion of wavefunction
GaSb_r = 11.*(2*np.pi/GaSb_lat_const)**2
GaSb_rlat_pts = sphere_pts(GaSb_rlat_vecs, GaSb_r, [0,0,0]) # reciprocal lattice points
GaSb_size = len(GaSb_rlat_pts)

def GaSb_PP(kpoint, neigvals):
    """Find the eigenvalues of a Hamiltonian matrix built to match the
    band structure of GaSb.
    
    Args:
        kpoint (list or numpy.ndarray): a point in k-space.
        neigvals (int): the number of returned eigenvalues in increasing
            order.
        
    Returns:
        eigvals (np.ndarray): the eigenvalues of the Hamiltonian matrix.
    """

    # Initialize the Hamiltonian.
    H = np.zeros((GaSb_size, GaSb_size),dtype=complex)

    # Populate the Hamiltonian.
    for i in range(GaSb_size):
        for j in range(i + 1):
            h = (GaSb_rlat_pts[j] - GaSb_rlat_pts[i])
            if i == j:
                H[i,j] = np.dot(kpoint + GaSb_rlat_pts[i], kpoint + GaSb_rlat_pts[i])
            if np.isclose(np.dot(h,h), 3*(2*np.pi/GaSb_lat_const)**2):
                H[i,j] = sGaSb_pff[0]*np.cos(np.dot(h,GaSb_tau)) + (
                    1j*aGaSb_pff[0]*np.sin(np.dot(h,GaSb_tau)))
            if np.isclose(np.dot(h,h), 4*(2*np.pi/GaSb_lat_const)**2):
                H[i,j] = GaSb_pff[0]*np.cos(np.dot(h,GaSb_tau))
            if np.isclose(np.dot(h,h), 11*(2*np.pi/GaSb_lat_const)**2):
                H[i,j] = sGaSb_pff[2]*np.cos(np.dot(h,GaSb_tau)) + (
                    1j*aGaSb_pff[2]*np.sin(np.dot(h,GaSb_tau)))
            else:
                continue
            
    return np.linalg.eigvalsh(H)[:neigvals]*Ry_to_eV # - GaSb_shift

def customGaSb_PP(kpoint, neigvals, GaSb_cutoff, shift=True, matrix=False):
    """Evaluate a GaSb pseudopotential at a given k-point. The pseudopotential
    form factors were taken from Cohen.

    Args:
        kpoint (numpy.array): a sampling point in k-space.
        neigvals (int): the number of eigenvalues returned
        Al_cutoff (float): the cutoff energy for the expansion. This value 
            determines the size of the Hamiltonian.
        shift (bool): if true, the basis expansion will include points centered
            about the provided kpoint.
        matrix (bool): if true, the Hamiltonian is returned instead of its
            eigenvalues.

    Return:
        (numpy.array): the sorted eigenvalues of the Hamiltonian or the Hamiltonian
            itself at the the provided k-point.
    """

    if shift:
        GaSb_rlat_pts = sphere_pts(GaSb_rlat_vecs, GaSb_r, kpoint) # reciprocal lattice points
    else:
        GaSb_rlat_pts = sphere_pts(GaSb_rlat_vecs, GaSb_r, [0,0,0]) # reciprocal lattice points

    GaSb_size = len(GaSb_rlat_pts)
    H = np.zeros((GaSb_size, GaSb_size),dtype=complex)
    
    for i in range(GaSb_size):
        for j in range(i + 1):
            h = (GaSb_rlat_pts[j] - GaSb_rlat_pts[i])
            if i == j:
                H[i,j] = np.dot(kpoint + GaSb_rlat_pts[i], kpoint + GaSb_rlat_pts[i])
            if np.isclose(np.dot(h,h), 3*(2*np.pi/GaSb_lat_const)**2):
                H[i,j] = sGaSb_pff[0]*np.cos(np.dot(h,GaSb_tau)) + (
                    1j*aGaSb_pff[0]*np.sin(np.dot(h,GaSb_tau)))
            if np.isclose(np.dot(h,h), 4*(2*np.pi/GaSb_lat_const)**2):
                H[i,j] = GaSb_pff[0]*np.cos(np.dot(h,GaSb_tau))
            if np.isclose(np.dot(h,h), 11*(2*np.pi/GaSb_lat_const)**2):
                H[i,j] = sGaSb_pff[2]*np.cos(np.dot(h,GaSb_tau)) + (
                    1j*aGaSb_pff[2]*np.sin(np.dot(h,GaSb_tau)))
            else:
                continue
    if matrix:
        return H*Ry_to_eV # - GaSb_shift
    else:
        return np.linalg.eigvalsh(H)[:neigvals]*Ry_to_eV # - GaSb_shift    

#### Pseudopotential of InAs ####
InAs_lat_type = "fcc"
InAs_lat_const = 6.04*angstrom_to_Bohr # the lattice constant in Bohr
InAs_rlat_vecs = make_rptvecs(InAs_lat_type, InAs_lat_const)
InAs_tau = InAs_lat_const/8.*np.array([1,1,1]) # one atomic basis vector

# The pseudopotential form factors of InAs
sInAs_pff =   [-0.22, 0.00, 0.05] # symmetric
aInAs_pff =  [0.08, 0.05, 0.03] # anti-symmetric

# The cutoff energy for the fourier expansion of wavefunction
InAs_r = 11.*(2*np.pi/InAs_lat_const)**2
InAs_rlat_pts = sphere_pts(InAs_rlat_vecs, InAs_r, [0,0,0]) # reciprocal lattice points
InAs_size = len(InAs_rlat_pts)

def InAs_PP(kpoint, neigvals):
    """Find the eigenvalues of a Hamiltonian matrix built to match the
    band structure of InAs.
    
    Args:
        kpoint (list or numpy.ndarray): a point in k-space.
        neigvals (int): the number of returned eigenvalues in increasing
            order.
        
    Returns:
        eigvals (np.ndarray): the eigenvalues of the Hamiltonian matrix.
    """

    # Initialize the Hamiltonian.
    H = np.zeros((InAs_size, InAs_size),dtype=complex)

    # Populate the Hamiltonian.
    for i in range(InAs_size):
        for j in range(i + 1):
            h = (InAs_rlat_pts[j] - InAs_rlat_pts[i])
            if i == j:
                H[i,j] = np.dot(kpoint + InAs_rlat_pts[i], kpoint + InAs_rlat_pts[i])
            if np.isclose(np.dot(h,h), 3*(2*np.pi/InAs_lat_const)**2):
                H[i,j] = sInAs_pff[0]*np.cos(np.dot(h,InAs_tau)) + (
                    1j*aInAs_pff[0]*np.sin(np.dot(h,InAs_tau)))
            if np.isclose(np.dot(h,h), 4*(2*np.pi/InAs_lat_const)**2):
                H[i,j] = InAs_pff[0]*np.cos(np.dot(h,InAs_tau))
            if np.isclose(np.dot(h,h), 11*(2*np.pi/InAs_lat_const)**2):
                H[i,j] = sInAs_pff[2]*np.cos(np.dot(h,InAs_tau)) + (
                    1j*aInAs_pff[2]*np.sin(np.dot(h,InAs_tau)))
            else:
                continue
            
    return np.linalg.eigvalsh(H)[:neigvals]*Ry_to_eV # - InAs_shift

def customInAs_PP(kpoint, neigvals, InAs_cutoff, shift=True, matrix=False):
    """Evaluate a InAs pseudopotential at a given k-point. The pseudopotential
    form factors were taken from Cohen.

    Args:
        kpoint (numpy.array): a sampling point in k-space.
        neigvals (int): the number of eigenvalues returned
        Al_cutoff (float): the cutoff energy for the expansion. This value 
            determines the size of the Hamiltonian.
        shift (bool): if true, the basis expansion will include points centered
            about the provided kpoint.
        matrix (bool): if true, the Hamiltonian is returned instead of its
            eigenvalues.

    Return:
        (numpy.array): the sorted eigenvalues of the Hamiltonian or the Hamiltonian
            itself at the the provided k-point.
    """

    if shift:
        InAs_rlat_pts = sphere_pts(InAs_rlat_vecs, InAs_r, kpoint) # reciprocal lattice points
    else:
        InAs_rlat_pts = sphere_pts(InAs_rlat_vecs, InAs_r, [0,0,0]) # reciprocal lattice points

    InAs_size = len(InAs_rlat_pts)
    H = np.zeros((InAs_size, InAs_size),dtype=complex)
    
    for i in range(InAs_size):
        for j in range(i + 1):
            h = (InAs_rlat_pts[j] - InAs_rlat_pts[i])
            if i == j:
                H[i,j] = np.dot(kpoint + InAs_rlat_pts[i], kpoint + InAs_rlat_pts[i])
            if np.isclose(np.dot(h,h), 3*(2*np.pi/InAs_lat_const)**2):
                H[i,j] = sInAs_pff[0]*np.cos(np.dot(h,InAs_tau)) + (
                    1j*aInAs_pff[0]*np.sin(np.dot(h,InAs_tau)))
            if np.isclose(np.dot(h,h), 4*(2*np.pi/InAs_lat_const)**2):
                H[i,j] = InAs_pff[0]*np.cos(np.dot(h,InAs_tau))
            if np.isclose(np.dot(h,h), 11*(2*np.pi/InAs_lat_const)**2):
                H[i,j] = sInAs_pff[2]*np.cos(np.dot(h,InAs_tau)) + (
                    1j*aInAs_pff[2]*np.sin(np.dot(h,InAs_tau)))
            else:
                continue
    if matrix:
        return H*Ry_to_eV # - InAs_shift
    else:
        return np.linalg.eigvalsh(H)[:neigvals]*Ry_to_eV # - InAs_shift    

#### Pseudopotential of InSb ####
InSb_lat_type = "fcc"
InSb_lat_const = 6.48*angstrom_to_Bohr # the lattice constant in Bohr
InSb_rlat_vecs = make_rptvecs(InSb_lat_type, InSb_lat_const)
InSb_tau = InSb_lat_const/8.*np.array([1,1,1]) # one atomic basis vector

# The pseudopotential form factors of InSb
sInSb_pff =   [-0.20, 0.00, 0.04] # symmetric
aInSb_pff =  [0.06, 0.05, 0.01] # anti-symmetric

# The cutoff energy for the fourier expansion of wavefunction
InSb_r = 11.*(2*np.pi/InSb_lat_const)**2
InSb_rlat_pts = sphere_pts(InSb_rlat_vecs, InSb_r, [0,0,0]) # reciprocal lattice points
InSb_size = len(InSb_rlat_pts)

def InSb_PP(kpoint, neigvals):
    """Find the eigenvalues of a Hamiltonian matrix built to match the
    band structure of InSb.
    
    Args:
        kpoint (list or numpy.ndarray): a point in k-space.
        neigvals (int): the number of returned eigenvalues in increasing
            order.
        
    Returns:
        eigvals (np.ndarray): the eigenvalues of the Hamiltonian matrix.
    """

    # Initialize the Hamiltonian.
    H = np.zeros((InSb_size, InSb_size),dtype=complex)

    # Populate the Hamiltonian.
    for i in range(InSb_size):
        for j in range(i + 1):
            h = (InSb_rlat_pts[j] - InSb_rlat_pts[i])
            if i == j:
                H[i,j] = np.dot(kpoint + InSb_rlat_pts[i], kpoint + InSb_rlat_pts[i])
            if np.isclose(np.dot(h,h), 3*(2*np.pi/InSb_lat_const)**2):
                H[i,j] = sInSb_pff[0]*np.cos(np.dot(h,InSb_tau)) + (
                    1j*aInSb_pff[0]*np.sin(np.dot(h,InSb_tau)))
            if np.isclose(np.dot(h,h), 4*(2*np.pi/InSb_lat_const)**2):
                H[i,j] = InSb_pff[0]*np.cos(np.dot(h,InSb_tau))
            if np.isclose(np.dot(h,h), 11*(2*np.pi/InSb_lat_const)**2):
                H[i,j] = sInSb_pff[2]*np.cos(np.dot(h,InSb_tau)) + (
                    1j*aInSb_pff[2]*np.sin(np.dot(h,InSb_tau)))
            else:
                continue
            
    return np.linalg.eigvalsh(H)[:neigvals]*Ry_to_eV # - InSb_shift

def customInSb_PP(kpoint, neigvals, InSb_cutoff, shift=True, matrix=False):
    """Evaluate a InSb pseudopotential at a given k-point. The pseudopotential
    form factors were taken from Cohen.

    Args:
        kpoint (numpy.array): a sampling point in k-space.
        neigvals (int): the number of eigenvalues returned
        Al_cutoff (float): the cutoff energy for the expansion. This value 
            determines the size of the Hamiltonian.
        shift (bool): if true, the basis expansion will include points centered
            about the provided kpoint.
        matrix (bool): if true, the Hamiltonian is returned instead of its
            eigenvalues.

    Return:
        (numpy.array): the sorted eigenvalues of the Hamiltonian or the Hamiltonian
            itself at the the provided k-point.
    """

    if shift:
        InSb_rlat_pts = sphere_pts(InSb_rlat_vecs, InSb_r, kpoint) # reciprocal lattice points
    else:
        InSb_rlat_pts = sphere_pts(InSb_rlat_vecs, InSb_r, [0,0,0]) # reciprocal lattice points

    InSb_size = len(InSb_rlat_pts)
    H = np.zeros((InSb_size, InSb_size),dtype=complex)
    
    for i in range(InSb_size):
        for j in range(i + 1):
            h = (InSb_rlat_pts[j] - InSb_rlat_pts[i])
            if i == j:
                H[i,j] = np.dot(kpoint + InSb_rlat_pts[i], kpoint + InSb_rlat_pts[i])
            if np.isclose(np.dot(h,h), 3*(2*np.pi/InSb_lat_const)**2):
                H[i,j] = sInSb_pff[0]*np.cos(np.dot(h,InSb_tau)) + (
                    1j*aInSb_pff[0]*np.sin(np.dot(h,InSb_tau)))
            if np.isclose(np.dot(h,h), 4*(2*np.pi/InSb_lat_const)**2):
                H[i,j] = InSb_pff[0]*np.cos(np.dot(h,InSb_tau))
            if np.isclose(np.dot(h,h), 11*(2*np.pi/InSb_lat_const)**2):
                H[i,j] = sInSb_pff[2]*np.cos(np.dot(h,InSb_tau)) + (
                    1j*aInSb_pff[2]*np.sin(np.dot(h,InSb_tau)))
            else:
                continue
    if matrix:
        return H*Ry_to_eV # - InSb_shift
    else:
        return np.linalg.eigvalsh(H)[:neigvals]*Ry_to_eV # - InSb_shift    

#### Pseudopotential of ZnS ####
ZnS_lat_type = "fcc"
ZnS_lat_const = 5.41*angstrom_to_Bohr # the lattice constant in Bohr
ZnS_rlat_vecs = make_rptvecs(ZnS_lat_type, ZnS_lat_const)
ZnS_tau = ZnS_lat_const/8.*np.array([1,1,1]) # one atomic basis vector

# The pseudopotential form factors of ZnS
sZnS_pff =   [-0.22, 0.03, 0.07] # symmetric
aZnS_pff =  [0.24, 0.14, 0.04] # anti-symmetric

# The cutoff energy for the fourier expansion of wavefunction
ZnS_r = 11.*(2*np.pi/ZnS_lat_const)**2
ZnS_rlat_pts = sphere_pts(ZnS_rlat_vecs, ZnS_r, [0,0,0]) # reciprocal lattice points
ZnS_size = len(ZnS_rlat_pts)

def ZnS_PP(kpoint, neigvals):
    """Find the eigenvalues of a Hamiltonian matrix built to match the
    band structure of ZnS.
    
    Args:
        kpoint (list or numpy.ndarray): a point in k-space.
        neigvals (int): the number of returned eigenvalues in increasing
            order.
        
    Returns:
        eigvals (np.ndarray): the eigenvalues of the Hamiltonian matrix.
    """

    # Initialize the Hamiltonian.
    H = np.zeros((ZnS_size, ZnS_size),dtype=complex)

    # Populate the Hamiltonian.
    for i in range(ZnS_size):
        for j in range(i + 1):
            h = (ZnS_rlat_pts[j] - ZnS_rlat_pts[i])
            if i == j:
                H[i,j] = np.dot(kpoint + ZnS_rlat_pts[i], kpoint + ZnS_rlat_pts[i])
            if np.isclose(np.dot(h,h), 3*(2*np.pi/ZnS_lat_const)**2):
                H[i,j] = sZnS_pff[0]*np.cos(np.dot(h,ZnS_tau)) + (
                    1j*aZnS_pff[0]*np.sin(np.dot(h,ZnS_tau)))
            if np.isclose(np.dot(h,h), 4*(2*np.pi/ZnS_lat_const)**2):
                H[i,j] = ZnS_pff[0]*np.cos(np.dot(h,ZnS_tau))
            if np.isclose(np.dot(h,h), 8*(2*np.pi/ZnS_lat_const)**2):
                H[i,j] = ZnS_pff[1]*np.cos(np.dot(h,ZnS_tau))
            if np.isclose(np.dot(h,h), 11*(2*np.pi/ZnS_lat_const)**2):
                H[i,j] = sZnS_pff[2]*np.cos(np.dot(h,ZnS_tau)) + (
                    1j*aZnS_pff[2]*np.sin(np.dot(h,ZnS_tau)))
            else:
                continue
            
    return np.linalg.eigvalsh(H)[:neigvals]*Ry_to_eV # - ZnS_shift

def customZnS_PP(kpoint, neigvals, ZnS_cutoff, shift=True, matrix=False):
    """Evaluate a ZnS pseudopotential at a given k-point. The pseudopotential
    form factors were taken from Cohen.

    Args:
        kpoint (numpy.array): a sampling point in k-space.
        neigvals (int): the number of eigenvalues returned
        Al_cutoff (float): the cutoff energy for the expansion. This value 
            determines the size of the Hamiltonian.
        shift (bool): if true, the basis expansion will include points centered
            about the provided kpoint.
        matrix (bool): if true, the Hamiltonian is returned instead of its
            eigenvalues.

    Return:
        (numpy.array): the sorted eigenvalues of the Hamiltonian or the Hamiltonian
            itself at the the provided k-point.
    """

    if shift:
        ZnS_rlat_pts = sphere_pts(ZnS_rlat_vecs, ZnS_r, kpoint) # reciprocal lattice points
    else:
        ZnS_rlat_pts = sphere_pts(ZnS_rlat_vecs, ZnS_r, [0,0,0]) # reciprocal lattice points

    ZnS_size = len(ZnS_rlat_pts)
    H = np.zeros((ZnS_size, ZnS_size),dtype=complex)
    
    for i in range(ZnS_size):
        for j in range(i + 1):
            h = (ZnS_rlat_pts[j] - ZnS_rlat_pts[i])
            if i == j:
                H[i,j] = np.dot(kpoint + ZnS_rlat_pts[i], kpoint + ZnS_rlat_pts[i])
            if np.isclose(np.dot(h,h), 3*(2*np.pi/ZnS_lat_const)**2):
                H[i,j] = sZnS_pff[0]*np.cos(np.dot(h,ZnS_tau)) + (
                    1j*aZnS_pff[0]*np.sin(np.dot(h,ZnS_tau)))
            if np.isclose(np.dot(h,h), 4*(2*np.pi/ZnS_lat_const)**2):
                H[i,j] = ZnS_pff[0]*np.cos(np.dot(h,ZnS_tau))
            if np.isclose(np.dot(h,h), 8*(2*np.pi/ZnS_lat_const)**2):
                H[i,j] = ZnS_pff[1]*np.cos(np.dot(h,ZnS_tau))
            if np.isclose(np.dot(h,h), 11*(2*np.pi/ZnS_lat_const)**2):
                H[i,j] = sZnS_pff[2]*np.cos(np.dot(h,ZnS_tau)) + (
                    1j*aZnS_pff[2]*np.sin(np.dot(h,ZnS_tau)))
            else:
                continue
    if matrix:
        return H*Ry_to_eV # - ZnS_shift
    else:
        return np.linalg.eigvalsh(H)[:neigvals]*Ry_to_eV # - ZnS_shift    

#### Pseudopotential of ZnSe ####
ZnSe_lat_type = "fcc"
ZnSe_lat_const = 5.65*angstrom_to_Bohr # the lattice constant in Bohr
ZnSe_rlat_vecs = make_rptvecs(ZnSe_lat_type, ZnSe_lat_const)
ZnSe_tau = ZnSe_lat_const/8.*np.array([1,1,1]) # one atomic basis vector

# The pseudopotential form factors of ZnSe
sZnSe_pff =   [-0.23, 0.01, 0.06] # symmetric
aZnSe_pff =  [0.18, 0.12, 0.03] # anti-symmetric

# The cutoff energy for the fourier expansion of wavefunction
ZnSe_r = 11.*(2*np.pi/ZnSe_lat_const)**2
ZnSe_rlat_pts = sphere_pts(ZnSe_rlat_vecs, ZnSe_r, [0,0,0]) # reciprocal lattice points
ZnSe_size = len(ZnSe_rlat_pts)

def ZnSe_PP(kpoint, neigvals):
    """Find the eigenvalues of a Hamiltonian matrix built to match the
    band structure of ZnSe.
    
    Args:
        kpoint (list or numpy.ndarray): a point in k-space.
        neigvals (int): the number of returned eigenvalues in increasing
            order.
        
    Returns:
        eigvals (np.ndarray): the eigenvalues of the Hamiltonian matrix.
    """

    # Initialize the Hamiltonian.
    H = np.zeros((ZnSe_size, ZnSe_size),dtype=complex)

    # Populate the Hamiltonian.
    for i in range(ZnSe_size):
        for j in range(i + 1):
            h = (ZnSe_rlat_pts[j] - ZnSe_rlat_pts[i])
            if i == j:
                H[i,j] = np.dot(kpoint + ZnSe_rlat_pts[i], kpoint + ZnSe_rlat_pts[i])
            if np.isclose(np.dot(h,h), 3*(2*np.pi/ZnSe_lat_const)**2):
                H[i,j] = sZnSe_pff[0]*np.cos(np.dot(h,ZnSe_tau)) + (
                    1j*aZnSe_pff[0]*np.sin(np.dot(h,ZnSe_tau)))
            if np.isclose(np.dot(h,h), 4*(2*np.pi/ZnSe_lat_const)**2):
                H[i,j] = ZnSe_pff[0]*np.cos(np.dot(h,ZnSe_tau))
            if np.isclose(np.dot(h,h), 8*(2*np.pi/ZnSe_lat_const)**2):
                H[i,j] = ZnSe_pff[1]*np.cos(np.dot(h,ZnSe_tau))
            if np.isclose(np.dot(h,h), 11*(2*np.pi/ZnSe_lat_const)**2):
                H[i,j] = sZnSe_pff[2]*np.cos(np.dot(h,ZnSe_tau)) + (
                    1j*aZnSe_pff[2]*np.sin(np.dot(h,ZnSe_tau)))
            else:
                continue
            
    return np.linalg.eigvalsh(H)[:neigvals]*Ry_to_eV # - ZnSe_shift

def customZnSe_PP(kpoint, neigvals, ZnSe_cutoff, shift=True, matrix=False):
    """Evaluate a ZnSe pseudopotential at a given k-point. The pseudopotential
    form factors were taken from Cohen.

    Args:
        kpoint (numpy.array): a sampling point in k-space.
        neigvals (int): the number of eigenvalues returned
        Al_cutoff (float): the cutoff energy for the expansion. This value 
            determines the size of the Hamiltonian.
        shift (bool): if true, the basis expansion will include points centered
            about the provided kpoint.
        matrix (bool): if true, the Hamiltonian is returned instead of its
            eigenvalues.

    Return:
        (numpy.array): the sorted eigenvalues of the Hamiltonian or the Hamiltonian
            itself at the the provided k-point.
    """

    if shift:
        ZnSe_rlat_pts = sphere_pts(ZnSe_rlat_vecs, ZnSe_r, kpoint) # reciprocal lattice points
    else:
        ZnSe_rlat_pts = sphere_pts(ZnSe_rlat_vecs, ZnSe_r, [0,0,0]) # reciprocal lattice points

    ZnSe_size = len(ZnSe_rlat_pts)
    H = np.zeros((ZnSe_size, ZnSe_size),dtype=complex)
    
    for i in range(ZnSe_size):
        for j in range(i + 1):
            h = (ZnSe_rlat_pts[j] - ZnSe_rlat_pts[i])
            if i == j:
                H[i,j] = np.dot(kpoint + ZnSe_rlat_pts[i], kpoint + ZnSe_rlat_pts[i])
            if np.isclose(np.dot(h,h), 3*(2*np.pi/ZnSe_lat_const)**2):
                H[i,j] = sZnSe_pff[0]*np.cos(np.dot(h,ZnSe_tau)) + (
                    1j*aZnSe_pff[0]*np.sin(np.dot(h,ZnSe_tau)))
            if np.isclose(np.dot(h,h), 4*(2*np.pi/ZnSe_lat_const)**2):
                H[i,j] = ZnSe_pff[0]*np.cos(np.dot(h,ZnSe_tau))
            if np.isclose(np.dot(h,h), 8*(2*np.pi/ZnSe_lat_const)**2):
                H[i,j] = ZnSe_pff[1]*np.cos(np.dot(h,ZnSe_tau))
            if np.isclose(np.dot(h,h), 11*(2*np.pi/ZnSe_lat_const)**2):
                H[i,j] = sZnSe_pff[2]*np.cos(np.dot(h,ZnSe_tau)) + (
                    1j*aZnSe_pff[2]*np.sin(np.dot(h,ZnSe_tau)))
            else:
                continue
    if matrix:
        return H*Ry_to_eV # - ZnSe_shift
    else:
        return np.linalg.eigvalsh(H)[:neigvals]*Ry_to_eV # - ZnSe_shift    

#### Pseudopotential of ZnTe ####
ZnTe_lat_type = "fcc"
ZnTe_lat_const = 6.07*angstrom_to_Bohr # the lattice constant in Bohr
ZnTe_rlat_vecs = make_rptvecs(ZnTe_lat_type, ZnTe_lat_const)
ZnTe_tau = ZnTe_lat_const/8.*np.array([1,1,1]) # one atomic basis vector

# The pseudopotential form factors of ZnTe
sZnTe_pff =   [-0.22, 0.00, 0.05] # symmetric
aZnTe_pff =  [0.13, 0.10, 0.01] # anti-symmetric

# The cutoff energy for the fourier expansion of wavefunction
ZnTe_r = 11.*(2*np.pi/ZnTe_lat_const)**2
ZnTe_rlat_pts = sphere_pts(ZnTe_rlat_vecs, ZnTe_r, [0,0,0]) # reciprocal lattice points
ZnTe_size = len(ZnTe_rlat_pts)

def ZnTe_PP(kpoint, neigvals):
    """Find the eigenvalues of a Hamiltonian matrix built to match the
    band structure of ZnTe.
    
    Args:
        kpoint (list or numpy.ndarray): a point in k-space.
        neigvals (int): the number of returned eigenvalues in increasing
            order.
        
    Returns:
        eigvals (np.ndarray): the eigenvalues of the Hamiltonian matrix.
    """

    # Initialize the Hamiltonian.
    H = np.zeros((ZnTe_size, ZnTe_size),dtype=complex)

    # Populate the Hamiltonian.
    for i in range(ZnTe_size):
        for j in range(i + 1):
            h = (ZnTe_rlat_pts[j] - ZnTe_rlat_pts[i])
            if i == j:
                H[i,j] = np.dot(kpoint + ZnTe_rlat_pts[i], kpoint + ZnTe_rlat_pts[i])
            if np.isclose(np.dot(h,h), 3*(2*np.pi/ZnTe_lat_const)**2):
                H[i,j] = sZnTe_pff[0]*np.cos(np.dot(h,ZnTe_tau)) + (
                    1j*aZnTe_pff[0]*np.sin(np.dot(h,ZnTe_tau)))
            if np.isclose(np.dot(h,h), 4*(2*np.pi/ZnTe_lat_const)**2):
                H[i,j] = ZnTe_pff[0]*np.cos(np.dot(h,ZnTe_tau))
            if np.isclose(np.dot(h,h), 11*(2*np.pi/ZnTe_lat_const)**2):
                H[i,j] = sZnTe_pff[2]*np.cos(np.dot(h,ZnTe_tau)) + (
                    1j*aZnTe_pff[2]*np.sin(np.dot(h,ZnTe_tau)))
            else:
                continue
            
    return np.linalg.eigvalsh(H)[:neigvals]*Ry_to_eV # - ZnTe_shift

def customZnTe_PP(kpoint, neigvals, ZnTe_cutoff, shift=True, matrix=False):
    """Evaluate a ZnTe pseudopotential at a given k-point. The pseudopotential
    form factors were taken from Cohen.

    Args:
        kpoint (numpy.array): a sampling point in k-space.
        neigvals (int): the number of eigenvalues returned
        Al_cutoff (float): the cutoff energy for the expansion. This value 
            determines the size of the Hamiltonian.
        shift (bool): if true, the basis expansion will include points centered
            about the provided kpoint.
        matrix (bool): if true, the Hamiltonian is returned instead of its
            eigenvalues.

    Return:
        (numpy.array): the sorted eigenvalues of the Hamiltonian or the Hamiltonian
            itself at the the provided k-point.
    """

    if shift:
        ZnTe_rlat_pts = sphere_pts(ZnTe_rlat_vecs, ZnTe_r, kpoint) # reciprocal lattice points
    else:
        ZnTe_rlat_pts = sphere_pts(ZnTe_rlat_vecs, ZnTe_r, [0,0,0]) # reciprocal lattice points

    ZnTe_size = len(ZnTe_rlat_pts)
    H = np.zeros((ZnTe_size, ZnTe_size),dtype=complex)
    
    for i in range(ZnTe_size):
        for j in range(i + 1):
            h = (ZnTe_rlat_pts[j] - ZnTe_rlat_pts[i])
            if i == j:
                H[i,j] = np.dot(kpoint + ZnTe_rlat_pts[i], kpoint + ZnTe_rlat_pts[i])
            if np.isclose(np.dot(h,h), 3*(2*np.pi/ZnTe_lat_const)**2):
                H[i,j] = sZnTe_pff[0]*np.cos(np.dot(h,ZnTe_tau)) + (
                    1j*aZnTe_pff[0]*np.sin(np.dot(h,ZnTe_tau)))
            if np.isclose(np.dot(h,h), 4*(2*np.pi/ZnTe_lat_const)**2):
                H[i,j] = ZnTe_pff[0]*np.cos(np.dot(h,ZnTe_tau))
            if np.isclose(np.dot(h,h), 11*(2*np.pi/ZnTe_lat_const)**2):
                H[i,j] = sZnTe_pff[2]*np.cos(np.dot(h,ZnTe_tau)) + (
                    1j*aZnTe_pff[2]*np.sin(np.dot(h,ZnTe_tau)))
            else:
                continue
    if matrix:
        return H*Ry_to_eV # - ZnTe_shift
    else:
        return np.linalg.eigvalsh(H)[:neigvals]*Ry_to_eV # - ZnTe_shift    

#### Pseudopotential of CdTe ####
CdTe_lat_type = "fcc"
CdTe_lat_const = 6.07*angstrom_to_Bohr # the lattice constant in Bohr
CdTe_rlat_vecs = make_rptvecs(CdTe_lat_type, CdTe_lat_const)
CdTe_tau = CdTe_lat_const/8.*np.array([1,1,1]) # one atomic basis vector

# The pseudopotential form factors of CdTe
sCdTe_pff =   [-0.20, 0.00, 0.04] # symmetric
aCdTe_pff =  [0.15, 0.09, 0.04] # anti-symmetric

# The cutoff energy for the fourier expansion of wavefunction
CdTe_r = 11.*(2*np.pi/CdTe_lat_const)**2
CdTe_rlat_pts = sphere_pts(CdTe_rlat_vecs, CdTe_r, [0,0,0]) # reciprocal lattice points
CdTe_size = len(CdTe_rlat_pts)

def CdTe_PP(kpoint, neigvals):
    """Find the eigenvalues of a Hamiltonian matrix built to match the
    band structure of CdTe.
    
    Args:
        kpoint (list or numpy.ndarray): a point in k-space.
        neigvals (int): the number of returned eigenvalues in increasing
            order.
        
    Returns:
        eigvals (np.ndarray): the eigenvalues of the Hamiltonian matrix.
    """

    # Initialize the Hamiltonian.
    H = np.zeros((CdTe_size, CdTe_size),dtype=complex)

    # Populate the Hamiltonian.
    for i in range(CdTe_size):
        for j in range(i + 1):
            h = (CdTe_rlat_pts[j] - CdTe_rlat_pts[i])
            if i == j:
                H[i,j] = np.dot(kpoint + CdTe_rlat_pts[i], kpoint + CdTe_rlat_pts[i])
            if np.isclose(np.dot(h,h), 3*(2*np.pi/CdTe_lat_const)**2):
                H[i,j] = sCdTe_pff[0]*np.cos(np.dot(h,CdTe_tau)) + (
                    1j*aCdTe_pff[0]*np.sin(np.dot(h,CdTe_tau)))
            if np.isclose(np.dot(h,h), 4*(2*np.pi/CdTe_lat_const)**2):
                H[i,j] = CdTe_pff[0]*np.cos(np.dot(h,CdTe_tau))
            if np.isclose(np.dot(h,h), 11*(2*np.pi/CdTe_lat_const)**2):
                H[i,j] = sCdTe_pff[2]*np.cos(np.dot(h,CdTe_tau)) + (
                    1j*aCdTe_pff[2]*np.sin(np.dot(h,CdTe_tau)))
            else:
                continue
            
    return np.linalg.eigvalsh(H)[:neigvals]*Ry_to_eV # - CdTe_shift

def customCdTe_PP(kpoint, neigvals, CdTe_cutoff, shift=True, matrix=False):
    """Evaluate a CdTe pseudopotential at a given k-point. The pseudopotential
    form factors were taken from Cohen.

    Args:
        kpoint (numpy.array): a sampling point in k-space.
        neigvals (int): the number of eigenvalues returned
        Al_cutoff (float): the cutoff energy for the expansion. This value 
            determines the size of the Hamiltonian.
        shift (bool): if true, the basis expansion will include points centered
            about the provided kpoint.
        matrix (bool): if true, the Hamiltonian is returned instead of its
            eigenvalues.

    Return:
        (numpy.array): the sorted eigenvalues of the Hamiltonian or the Hamiltonian
            itself at the the provided k-point.
    """

    if shift:
        CdTe_rlat_pts = sphere_pts(CdTe_rlat_vecs, CdTe_r, kpoint) # reciprocal lattice points
    else:
        CdTe_rlat_pts = sphere_pts(CdTe_rlat_vecs, CdTe_r, [0,0,0]) # reciprocal lattice points

    CdTe_size = len(CdTe_rlat_pts)
    H = np.zeros((CdTe_size, CdTe_size),dtype=complex)
    
    for i in range(CdTe_size):
        for j in range(i + 1):
            h = (CdTe_rlat_pts[j] - CdTe_rlat_pts[i])
            if i == j:
                H[i,j] = np.dot(kpoint + CdTe_rlat_pts[i], kpoint + CdTe_rlat_pts[i])
            if np.isclose(np.dot(h,h), 3*(2*np.pi/CdTe_lat_const)**2):
                H[i,j] = sCdTe_pff[0]*np.cos(np.dot(h,CdTe_tau)) + (
                    1j*aCdTe_pff[0]*np.sin(np.dot(h,CdTe_tau)))
            if np.isclose(np.dot(h,h), 4*(2*np.pi/CdTe_lat_const)**2):
                H[i,j] = CdTe_pff[0]*np.cos(np.dot(h,CdTe_tau))
            if np.isclose(np.dot(h,h), 11*(2*np.pi/CdTe_lat_const)**2):
                H[i,j] = sCdTe_pff[2]*np.cos(np.dot(h,CdTe_tau)) + (
                    1j*aCdTe_pff[2]*np.sin(np.dot(h,CdTe_tau)))
            else:
                continue
    if matrix:
        return H*Ry_to_eV # - CdTe_shift
    else:
        return np.linalg.eigvalsh(H)[:neigvals]*Ry_to_eV # - CdTe_shift    
    
#### 
# Pseudo-potential from Band Structures and Pseudopotential Form Factors for
# Fourteen Semiconductors of the Diamond. and. Zinc-blende Structures* by
# Cohen and Bergstresser
#### 

# Define the pseudo-potential form factors taken from Cohen and Bergstesser
# V3S stands for the symmetric form factor for reciprocal lattice vectors of
# squared magnitude 3.
# V4A stands for the antisymmetric form factor for reciprocal lattice vectors
# of squared magnitude 4.
#            V3S   V8S   V11S  V3A   V4A   V11A
Si_pff =   [-0.21, 0.04, 0.08, 0.00, 0.00, 0.00]
Ge_pff =   [-0.23, 0.01, 0.06, 0.00, 0.00, 0.00]
Sn_pff =   [-0.20, 0.00, 0.04, 0.00, 0.00, 0.00]
GaP_pff =  [-0.22, 0.03, 0.07, 0.12, 0.07, 0.02]
GaAs_pff = [-0.23, 0.01, 0.06, 0.07, 0.05, 0.01]
AlSb_pff = [-0.21, 0.02, 0.06, 0.06, 0.04, 0.02]
InP_pff =  [-0.23, 0.01, 0.06, 0.07, 0.05, 0.01]
GaSb_pff = [-0.22, 0.00, 0.05, 0.06, 0.05, 0.01]
InAs_pff = [-0.22, 0.00, 0.05, 0.08, 0.05, 0.03]
InSb_pff = [-0.20, 0.00, 0.04, 0.06, 0.05, 0.01]
ZnS_pff =  [-0.22, 0.03, 0.07, 0.24, 0.14, 0.04]
ZnSe_pff = [-0.23, 0.01, 0.06, 0.18, 0.12, 0.03]
ZnTe_pff = [-0.22, 0.00, 0.05, 0.13, 0.10, 0.01]
CdTe_pff = [-0.20, 0.00, 0.04, 0.15, 0.09, 0.04]

# def SiPP(kpoint, matrix=False):
#     """Evaluate an Si pseudopotential at a given k-point. The pseudopotential
#     form factors were taken from 
#     `Harrison <https://journals.aps.org/pr/abstract/10.1103/PhysRev.141.789`_.

#     Args:
#         a (float): the lattice constant
#         kpoint (numpy.array): a sampling point in k-space in cartesian coordinates.

#     Return:
#         (numpy.array): the sorted eigenvalues of the Hamiltonian at the provided
#         k-point.
#     """
    
#     # Lattice constant of Si.
#     a = 5.43
#     lat_type = "fcc"
    
#     kpoint = np.asarray(kpoint)
#     tau = a/8*np.array([1,1,1])
#     atomic_basis = [tau, -tau]
#     # These reciprocal lattice points are in Cartesian coordinates but have had
#     # 2*np.pi/a factored out so that they are integers.
#     Si_rlat_pts = find_intvecs(0) + find_intvecs(3) + find_intvecs(8) + find_intvecs(11)
    
#     # Initialize the Si pseudopotential Hamiltonian.
#     Si_H = np.zeros([len(Si_rlat_pts), len(Si_rlat_pts)], dtype=complex)
#     # Construct the Al Hamiltonian.
#     for d_nu in atomic_basis:
#         for (i, k1) in enumerate(Si_rlat_pts):
#             k1 = np.asarray(k1)
#             for (j, k2) in enumerate(Si_rlat_pts):
#                 k2 = np.asarray(k2)
#                 h = k2 - k1
#                 n2 = norm(h)**2
#                 # V(h_j)
#                 # if i == j:
#                 #     Si_H[i,i] += norm(kpoint + 2*np.pi/a*k1)**2/len(atomic_basis)
#                 # elif i == 0 or j == 0:
#                 #     if np.isclose(n2, 3) == True:
#                 #         Si_H[i,j] += Si_pff[0]*np.exp(-1j*2*np.pi/a*np.dot(h, d_nu))
#                 #     elif np.isclose(n2, 8) == True:
#                 #         Si_H[i,j] += Si_pff[1]*np.exp(-1j*2*np.pi/a*np.dot(h, d_nu))
#                 #     elif np.isclose(n2, 11) == True:
#                 #         Si_H[i,j] += Si_pff[2]*np.exp(-1j*2*np.pi/a*np.dot(h, d_nu))
#                 #     else:
#                 #         pass
#                 # else:
#                 #     pass
                
#                 # V(h_i - h_j)
#                 if np.isclose(n2, 3) == True:
#                     Si_H[i,j] += Si_pff[0]*np.exp(-1j*2*np.pi/a*np.dot(h, d_nu))
#                 elif np.isclose(n2, 8) == True:
#                     Si_H[i,j] += Si_pff[1]*np.exp(-1j*2*np.pi/a*np.dot(h, d_nu))
#                 elif np.isclose(n2, 11) == True:
#                     Si_H[i,j] += Si_pff[2]*np.exp(-1j*2*np.pi/a*np.dot(h, d_nu))
#                 elif i == j:
#                     Si_H[i,i] += norm(kpoint + 2*np.pi/a*k1)**2/len(atomic_basis)
#                 else:
#                     pass

#                 # # V(h_i - h_j)
#                 # if np.isclose(n2, 3) == True:
#                 #     Si_H[i,j] += Si_pff[0]*np.exp(-1j*2*np.pi/a*np.dot(h, d_nu))
#                 # elif np.isclose(n2, 8) == True:
#                 #     Si_H[i,j] += Si_pff[1]*np.exp(-1j*2*np.pi/a*np.dot(h, d_nu))
#                 # elif np.isclose(n2, 11) == True:
#                 #     Si_H[i,j] += Si_pff[2]*np.exp(-1j*2*np.pi/a*np.dot(h, d_nu))
#                 # elif np.isclose(n2, 0) == True:
#                 #     Si_H[i,i] += norm(kpoint + 2*np.pi/a*k1)**2/len(atomic_basis)
#                 # else:
#                 #     pass
                
#     if matrix == True:
#         return Si_H
#     else:
#         return np.sort(np.linalg.eigvals(Si_H))
