import numpy as np
from numba import njit, prange
from vedo import Mesh, show

@njit(parallel=True, fastmath=True)
def compute_mesh_volume(nodes_xyz, elements):
    total_volume = 0.0
    for i in prange(elements.shape[0]):
        e = elements[i]
        v0, v1, v2, v3 = nodes_xyz[e[0]], nodes_xyz[e[1]], nodes_xyz[e[2]], nodes_xyz[e[3]]
        vol = np.abs(np.dot(np.cross(v1 - v0, v2 - v0), v3 - v0)) / 6.0
        total_volume += vol
    return total_volume

# bc for 100x100x100 lattice structure compression simulation 
def prepare_boundary_conditions(node_xyz, nodes):
    F = np.zeros(3 * nodes.shape[0])
    dof_x = 3 * np.arange(len(nodes))
    dof_y = dof_x + 1
    dof_z = dof_x + 2

    y0_mask = np.isclose(node_xyz[:, 1], 0.0, atol=1e-8)
    y100_mask = np.isclose(node_xyz[:, 1], 100.0, atol=1e-8)
    x_fix_mask = (
        (np.isclose(node_xyz[:, 0], 100.0, atol=1e-8)) &
        ((np.isclose(node_xyz[:, 1], 0.0, atol=1e-8) | np.isclose(node_xyz[:, 1], 100.0, atol=1e-8)) &
         (np.isclose(node_xyz[:, 2], 0.0, atol=1e-8) | np.isclose(node_xyz[:, 2], 100.0, atol=1e-8)))
    )
    z_fix_mask = (
        (np.isclose(node_xyz[:, 2], 100.0, atol=1e-8)) &
        ((np.isclose(node_xyz[:, 1], 0.0, atol=1e-8) | np.isclose(node_xyz[:, 1], 100.0, atol=1e-8)) &
         (np.isclose(node_xyz[:, 0], 0.0, atol=1e-8) | np.isclose(node_xyz[:, 0], 100.0, atol=1e-8)))
    )

    fixed_dofs = np.concatenate([
        dof_y[y0_mask],
        dof_y[y100_mask],
        dof_x[x_fix_mask],
        dof_z[z_fix_mask]
    ])
    u_fixed = np.concatenate([
        np.zeros(np.count_nonzero(y0_mask)),
        -0.2 * np.ones(np.count_nonzero(y100_mask)),
        np.zeros(np.count_nonzero(x_fix_mask)),
        np.zeros(np.count_nonzero(z_fix_mask))
    ])

    return F, fixed_dofs, u_fixed, y0_mask, y100_mask, x_fix_mask, z_fix_mask

def visualize_displacement(nodes, elements, u, show_edges=True, scale_factor=1.0):
    """
    Visualize FEM displacement result using vedo.

    Parameters:
        nodes (ndarray): (n_nodes, 4) array with [id, x, y, z]
        elements (ndarray): (n_elements, 4) array with node indices (zero-based)
        u (ndarray): (n_dofs,) global displacement vector
        show_edges (bool): If True, show element edges
        scale_factor (float): Factor to exaggerate deformation
    """
    points = nodes[:, 1:4]
    displacements = u.reshape((-1, 3))
    displaced = points + displacements * scale_factor
    values = np.linalg.norm(displacements, axis=1)

    mesh = Mesh([displaced, elements])
    mesh.cmap("viridis", values, on="points")
    mesh.add_scalarbar(title="Displacement [mm]")

    show(mesh, axes=1, viewup="z", title="FEM Displacement", bg="white")



def master_slave_match_nodes(idx_a, idx_b, dim, node_xyz):
    """
    Match all slave nodes to master nodes based on closest proximity (in dim directions).
    All slave nodes are guaranteed to be matched. If needed, unmatched nodes are linked
    to nearest already-matched slave nodes' master.

    Args:
        idx_a (ndarray): Node indices of face A.
        idx_b (ndarray): Node indices of face B.
        dim (list[int]): Dimensions to match in (e.g., [1,2] for y,z).
        node_xyz (ndarray): (N, 3) array of node coordinates.

    Returns:
        pairs (list[tuple[int, int]]): (master, slave) index pairs.
    """
    if len(idx_a) <= len(idx_b):
        master_idx, slave_idx = idx_a, idx_b
    else:
        master_idx, slave_idx = idx_b, idx_a

    master_coords = node_xyz[master_idx][:, dim]
    slave_coords = node_xyz[slave_idx][:, dim]

    pairs = []
    used_slave = set()

    # Direct 1:1 matching
    for m_idx, m_coord in zip(master_idx, master_coords):
        dists = np.linalg.norm(slave_coords - m_coord, axis=1)
        sorted_indices = np.argsort(dists)
        for si in sorted_indices:
            s_idx = slave_idx[si]
            if s_idx not in used_slave:
                pairs.append((m_idx, s_idx))
                used_slave.add(s_idx)
                break

    matched_slave = np.array([s for _, s in pairs], dtype=int)
    unmatched_slave = np.setdiff1d(slave_idx, matched_slave, assume_unique=True)

    # Fallback: match unmatched_slave → nearest matched_slave's master
    if unmatched_slave.size > 0:
        slave_coords_matched = node_xyz[matched_slave][:, dim]
        for s_idx in unmatched_slave:
            s_coord = node_xyz[s_idx, dim]
            dists = np.linalg.norm(slave_coords_matched - s_coord, axis=1)
            nearest_idx = np.argmin(dists)
            matched_slave_idx = matched_slave[nearest_idx]

            # Find its master from existing pairs
            for m, s in pairs:
                if s == matched_slave_idx:
                    pairs.append((m, s_idx))
                    break

    # Final check
    all_slave_indices = set(slave_idx)
    matched_now = set(s for _, s in pairs)
    if matched_now != all_slave_indices:
        missing = all_slave_indices - matched_now
        raise ValueError(f"Matching failed for {len(missing)} slave nodes: {sorted(missing)[:5]}...")

    return pairs


def prepare_shear_boundary_conditions(node_xyz):
    """
    Define shear boundary conditions:
    - y=0: fully fixed (ux, uy, uz = 0)
    - y=100: uz = -0.2
    - x/z opposing faces: periodic matching of (ux, uy)

    Args:
        node_xyz (ndarray): (N, 3) node coordinates.

    Returns:
        F (ndarray): Zero force vector.
        fixed_dofs (ndarray): Indices of fixed DOFs.
        u_fixed (ndarray): Prescribed displacement values.
    """
    N = node_xyz.shape[0]
    tol = 1e-8
    F = np.zeros(3 * N)
    all_dofs = []
    all_disp = []

    # y=0 fixed
    y0_nodes = np.where(np.abs(node_xyz[:, 1]) < tol)[0]
    for n in y0_nodes:
        all_dofs += [3*n, 3*n+1, 3*n+2]
        all_disp += [0.0, 0.0, 0.0]

    # y=100 displacement
    y100_nodes = np.where(np.abs(node_xyz[:, 1] - 100.0) < tol)[0]
    for n in y100_nodes:
        all_dofs.append(3*n + 2)
        all_disp.append(-0.2)

    # x-face matching
    x0_idx = np.where(np.abs(node_xyz[:, 0]) < tol)[0]
    x100_idx = np.where(np.abs(node_xyz[:, 0] - 100.0) < tol)[0]
    pairs_x = master_slave_match_nodes(x0_idx, x100_idx, [1, 2], node_xyz)
    for i, j in pairs_x:
        all_dofs += [3*i, 3*i+1, 3*j, 3*j+1]
        all_disp += [0.0, 0.0, 0.0, 0.0]

    # z-face matching
    z0_idx = np.where(np.abs(node_xyz[:, 2]) < tol)[0]
    z100_idx = np.where(np.abs(node_xyz[:, 2] - 100.0) < tol)[0]
    pairs_z = master_slave_match_nodes(z0_idx, z100_idx, [0, 1], node_xyz)
    for i, j in pairs_z:
        all_dofs += [3*i, 3*i+1, 3*j, 3*j+1]
        all_disp += [0.0, 0.0, 0.0, 0.0]

    # Remove duplicates
    all_dofs = np.array(all_dofs)
    all_disp = np.array(all_disp)
    unique_dofs, idx = np.unique(all_dofs, return_index=True)
    u_fixed = all_disp[idx]

    return F, unique_dofs, u_fixed
