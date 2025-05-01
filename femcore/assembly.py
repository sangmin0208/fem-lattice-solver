import numpy as np
from scipy.sparse import coo_matrix
from numba import njit, prange

# Define material properties from Young's modulus and Poisson's ratio
def define_material(E=1770.0, nu=0.3):
    lambda_ = (E * nu) / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return lambda_, mu

# Compute element stiffness matrix for 4-node tetrahedron using numerical integration
@njit(cache=True, fastmath=True)
def compute_tetra4_stiffness_numba(Xe, lambda_, mu):
    dNdxi = np.array([[-1.0, -1.0, -1.0],
                      [ 1.0,  0.0,  0.0],
                      [ 0.0,  1.0,  0.0],
                      [ 0.0,  0.0,  1.0]], dtype=np.float64)

    J = Xe.T @ dNdxi
    detJ = np.linalg.det(J)
    if abs(detJ) < 1e-12:
        return np.zeros((12, 12))  # Ignore degenerate elements

    invJ = np.linalg.inv(J)
    dNdX = dNdxi @ invJ
    volume = detJ / 6.0

    B = np.zeros((6, 12), dtype=np.float64)
    for i in range(4):
        ix = i * 3
        B[0, ix + 0] = dNdX[i, 0]
        B[1, ix + 1] = dNdX[i, 1]
        B[2, ix + 2] = dNdX[i, 2]
        B[3, ix + 0] = dNdX[i, 1]
        B[3, ix + 1] = dNdX[i, 0]
        B[4, ix + 0] = dNdX[i, 2]
        B[4, ix + 2] = dNdX[i, 0]
        B[5, ix + 1] = dNdX[i, 2]
        B[5, ix + 2] = dNdX[i, 1]

    C = np.zeros((6, 6), dtype=np.float64)
    for i in range(3):
        C[i, i] = lambda_ + 2.0 * mu
        for j in range(3):
            if i != j:
                C[i, j] = lambda_
    for i in range(3, 6):
        C[i, i] = mu

    Ke = B.T @ C @ B * volume
    return Ke

# Assemble global stiffness matrix using triplet format
@njit(parallel=True, cache=True, fastmath=True)
def assemble_triplets_numba(nodes, elements, lambda_, mu):
    n_nodes = nodes.shape[0]
    nodes_xyz = nodes[:, 1:4]
    n_elems = len(elements)
    total_nnz = n_elems * 144
    rows = np.empty(total_nnz, dtype=np.int32)
    cols = np.empty_like(rows)
    data = np.empty_like(rows, dtype=np.float64)

    for ei in prange(n_elems):
        e = elements[ei]
        Xe = nodes_xyz[e]
        Ke = compute_tetra4_stiffness_numba(Xe, lambda_, mu)

        dof = np.array([3*e[0], 3*e[0]+1, 3*e[0]+2,
                        3*e[1], 3*e[1]+1, 3*e[1]+2,
                        3*e[2], 3*e[2]+1, 3*e[2]+2,
                        3*e[3], 3*e[3]+1, 3*e[3]+2], dtype=np.int32)

        base = ei * 144
        for i in range(12):
            for j in range(12):
                idx = base + i * 12 + j
                rows[idx] = dof[i]
                cols[idx] = dof[j]
                data[idx] = Ke[i, j]

    return rows, cols, data

# Final function to build global stiffness matrix
def assemble_global_stiffness(nodes, elements, lambda_, mu):
    rows, cols, data = assemble_triplets_numba(nodes, elements, lambda_, mu)
    n_dofs = 3 * nodes.shape[0]
    return coo_matrix((data, (rows, cols)), shape=(n_dofs, n_dofs)).tocsr()