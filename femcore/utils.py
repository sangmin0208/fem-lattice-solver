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
