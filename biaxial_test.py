import time
import os
import numpy as np
from scipy.sparse.linalg import spsolve
from femcore import read_gmsh_mesh, define_material, prepare_boundary_conditions, compute_mesh_volume, assemble_global_stiffness


def main():
    total_start = time.time()
    runtime_log = {}

    path = "./examples/fcc[r1=4.575_r2=9.407_r3=6.630_r4=6.979_r5=6.881_vol=122030.945].msh"
    scale = 1.0

    print("Importing mesh...")
    t0 = time.time()
    nodes, elements = read_gmsh_mesh(path, scale)
    runtime_log['mesh_import'] = time.time() - t0
    print(f"Mesh import time: {runtime_log['mesh_import']:.4f} sec")
    print(f"Number of nodes       : {nodes.shape[0]}")
    print(f"Number of elements    : {elements.shape[0]}")

    node_xyz = nodes[:, 1:4]
    t0 = time.time()
    total_volume = compute_mesh_volume(node_xyz, elements)
    runtime_log['volume_calc'] = time.time() - t0
    print(f"Computed mesh volume  : {total_volume:.4f} mm³")

    print("Defining material properties...")
    t0 = time.time()
    lam, mu = define_material(E=775.43, nu=0.3)
    runtime_log['material'] = time.time() - t0
    print(f"Lamé constants        : λ = {lam:.3f}, μ = {mu:.3f}")

    print("Assembling global stiffness matrix...")
    t0 = time.time()
    K = assemble_global_stiffness(nodes, elements, lam, mu)
    runtime_log['stiffness_assembly'] = time.time() - t0
    print(f"Stiffness assembly time: {runtime_log['stiffness_assembly']:.4f} sec")
    print(f"Global matrix shape    : {K.shape}, nonzeros: {K.nnz}")

    print("Preparing force vector and DOF indices...")
    t0 = time.time()
    F, fixed_dofs, u_fixed, y0_mask, y100_mask, x_fix_mask, z_fix_mask = prepare_boundary_conditions(node_xyz, nodes)
    runtime_log['bc_dof_setup'] = time.time() - t0

    print("Dirichlet boundary conditions summary:")
    print(f"  Nodes on y=0 face (fixed)      : {np.count_nonzero(y0_mask)}")
    print(f"  Nodes on y=100 face (-0.2 disp): {np.count_nonzero(y100_mask)}")
    print(f"  Nodes fixed in x-direction     : {np.count_nonzero(x_fix_mask)}")
    print(f"  Nodes fixed in z-direction     : {np.count_nonzero(z_fix_mask)}")
    print(f"  Total fixed DOFs               : {len(fixed_dofs)}")
    print(f"  Total free DOFs                : {3 * nodes.shape[0] - len(fixed_dofs)}")

    print("Applying boundary conditions...")
    t0 = time.time()
    Kff = K.copy()
    Kfc = K[:, fixed_dofs].tocsc()[~np.isin(np.arange(F.shape[0]), fixed_dofs), :]
    Ff = F.copy()
    Ff[~np.isin(np.arange(F.shape[0]), fixed_dofs)] -= Kfc @ u_fixed
    runtime_log['apply_bc'] = time.time() - t0

    print("Solving linear system...")
    t0 = time.time()
    os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
    os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())
    os.environ["NUMEXPR_NUM_THREADS"] = str(os.cpu_count())

    try:
        from pypardiso import spsolve as direct_solve
        print("Using PyPardiso solver")
    except ImportError:
        print("PyPardiso not found. Falling back to SciPy solver")
        direct_solve = spsolve

    u = np.zeros_like(F)
    free_dofs = np.setdiff1d(np.arange(F.shape[0]), fixed_dofs, assume_unique=True)
    u[fixed_dofs] = u_fixed
    u[free_dofs] = direct_solve(K[free_dofs, :][:, free_dofs], Ff[free_dofs])
    runtime_log['solve'] = time.time() - t0

    print("Linear system solved.")
    print(f"Solve time: {runtime_log['solve']:.4f} sec")

    print("Calculating reaction force at y=0 face...")
    t0 = time.time()
    mask_y0 = np.isclose(nodes[:, 2], 0.0, atol=1e-8)
    dof_y_indices = 3 * np.nonzero(mask_y0)[0] + 1
    reaction_force_y = K[dof_y_indices, :].dot(u).sum()
    runtime_log['reaction'] = time.time() - t0
    print(f"Total reaction force (y=0 face, y-dir): {reaction_force_y:.6f} N")

    print("Runtime summary:")
    for k, v in runtime_log.items():
        print(f"  {k:<24}: {v:.4f} sec")
    print(f"Total execution time: {time.time() - total_start:.4f} sec")
    
if __name__ == '__main__':
    main()
