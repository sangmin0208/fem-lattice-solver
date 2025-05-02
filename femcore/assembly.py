import numpy as np
from scipy.sparse import coo_matrix
from numba import njit, prange

def define_material(E=1770.0, nu=0.3):
    """
    Define material constants (lame parameters) from Young's modulus (E)
    and Poisson's ratio (v)
    """
    lambda_ = (E * nu) / ((1 + nu) * (1 - 2 * nu))  # First Lamé parameter
    mu = E / (2 * (1 + nu)) # Shear modulus (second Lamé parameter)
    return lambda_, mu

# Compute element stiffness matrix for 4-node tetrahedron using numerical integration
@njit(cache=True, fastmath=True)
def compute_tetra4_stiffness_numba(Xe, lambda_, mu):
    # Gradients of shape functions w.r.t. reference coordinates (natural coordinates)
    dNdxi = np.array([[-1.0, -1.0, -1.0],
                      [ 1.0,  0.0,  0.0],
                      [ 0.0,  1.0,  0.0],
                      [ 0.0,  0.0,  1.0]], dtype=np.float64)
    
    # Jacobian matrix: transforms from reference to physical coordinates
    J = Xe.T @ dNdxi
    detJ = np.linalg.det(J)
    #if abs(detJ) < 1e-12:
    #    return np.zeros((12, 12))  # Ignore degenerate elements

    invJ = np.linalg.inv(J)
    dNdX = dNdxi @ invJ
    volume = detJ / 6.0

    B = np.zeros((6, 12), dtype=np.float64)
    for i in range(4):
        ix = i * 3
        B[0, ix + 0] = dNdX[i, 0]   # ε_xx
        B[1, ix + 1] = dNdX[i, 1]   # ε_yy
        B[2, ix + 2] = dNdX[i, 2]   # ε_zz
        B[3, ix + 0] = dNdX[i, 1]   # ε_xy
        B[3, ix + 1] = dNdX[i, 0]
        B[4, ix + 0] = dNdX[i, 2]   # ε_xz
        B[4, ix + 2] = dNdX[i, 0]
        B[5, ix + 1] = dNdX[i, 2]   # ε_yz
        B[5, ix + 2] = dNdX[i, 1]

    # Construct material constitutive matrix C for linear isotropic elasticity (6×6)
    C = np.zeros((6, 6), dtype=np.float64)
    for i in range(3):
        C[i, i] = lambda_ + 2.0 * mu    # Normal stress terms
        for j in range(3):
            if i != j:
                C[i, j] = lambda_   # Coupling terms
    for i in range(3, 6):
        C[i, i] = mu

    # Compute the element stiffness matrix: Ke = Bᵀ * C * B * volume
    Ke = B.T @ C @ B * volume
    return Ke

# Assemble global stiffness matrix using triplet format
@njit(parallel=True, cache=True, fastmath=True)
def assemble_triplets_numba(nodes, elements, lambda_, mu):
    #n_nodes = nodes.shape[0]
    nodes_xyz = nodes[:, 1:4]   # Extract only x, y, z coordinates (ignore node ID)
    n_elems = len(elements)
    total_nnz = n_elems * 144
    rows = np.empty(total_nnz, dtype=np.int32)
    cols = np.empty_like(rows)
    data = np.empty_like(rows, dtype=np.float64)

    for ei in prange(n_elems):
        e = elements[ei]
        Xe = nodes_xyz[e]
        Ke = compute_tetra4_stiffness_numba(Xe, lambda_, mu)
        
        # Construct global DOF mapping (3 DOFs per node)
        dof = np.array([3*e[0], 3*e[0]+1, 3*e[0]+2,
                        3*e[1], 3*e[1]+1, 3*e[1]+2,
                        3*e[2], 3*e[2]+1, 3*e[2]+2,
                        3*e[3], 3*e[3]+1, 3*e[3]+2], dtype=np.int32)
        
        # Flatten element stiffness matrix into triplet format
        base = ei * 144
        for i in range(12):
            for j in range(12):
                idx = base + i * 12 + j
                rows[idx] = dof[i]
                cols[idx] = dof[j]
                data[idx] = Ke[i, j]

    return rows, cols, data

# Convert triplets to global sparse stiffness matrix (CSR format)
def assemble_global_stiffness(nodes, elements, lambda_, mu):
    rows, cols, data = assemble_triplets_numba(nodes, elements, lambda_, mu)
    n_dofs = 3 * nodes.shape[0]
    return coo_matrix((data, (rows, cols)), shape=(n_dofs, n_dofs)).tocsr()