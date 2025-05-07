import time
import numpy as np
from scipy.sparse.linalg import spsolve

from femcore import (
    read_gmsh_mesh,
    define_material,
    compute_mesh_volume,
    assemble_global_stiffness,
    prepare_shear_boundary_conditions
)

def main():
    total_start = time.time()
    log = {}

    # Path to Gmsh-generated mesh
    path = "./examples/fcc[r1=9.546_r2=9.117_r3=9.853_r4=3.733_r5=6.414_vol=175722.536].msh"
    scale = 1.0

    print("Loading mesh...")
    t0 = time.time()
    nodes, elements = read_gmsh_mesh(path, scale)
    log['mesh_import'] = time.time() - t0
    print(f" - Nodes    : {nodes.shape[0]}")
    print(f" - Elements : {elements.shape[0]}")

    node_xyz = nodes[:, 1:4]

    print("Computing mesh volume...")
    t0 = time.time()
    volume = compute_mesh_volume(node_xyz, elements)
    log['volume'] = time.time() - t0
    print(f" - Volume = {volume:.3f} mm³")

    print("Defining material...")
    t0 = time.time()
    lam, mu = define_material(E=775.43, nu=0.3)
    log['material'] = time.time() - t0
    print(f" - Lamé parameters: λ = {lam:.3f}, μ = {mu:.3f}")

    print("Assembling global stiffness matrix...")
    t0 = time.time()
    K = assemble_global_stiffness(nodes, elements, lam, mu)
    log['assemble'] = time.time() - t0
    print(f" - K shape: {K.shape}, nnz: {K.nnz}")

    print("Applying shear boundary conditions...")
    t0 = time.time()
    F, fixed_dofs, u_fixed = prepare_shear_boundary_conditions(node_xyz)
    log['bc'] = time.time() - t0
    print(f" - Total fixed DOFs: {len(fixed_dofs)}")

    print("Solving linear system...")
    t0 = time.time()
    try:
        from pypardiso import spsolve as direct_solve
        print(" - Using PyPardiso solver")
    except ImportError:
        print(" - Using SciPy solver")
        direct_solve = spsolve

    u = np.zeros_like(F)
    free_dofs = np.setdiff1d(np.arange(F.shape[0]), fixed_dofs, assume_unique=True)

    Kff = K[free_dofs, :][:, free_dofs]
    Kfc = K[free_dofs, :][:, fixed_dofs]
    Ff = F[free_dofs] - Kfc @ u_fixed

    u[fixed_dofs] = u_fixed
    u[free_dofs] = direct_solve(Kff, Ff)
    log['solve'] = time.time() - t0

    print("Computing reaction force at y=0 face...")
    t0 = time.time()
    y0_nodes = np.where(np.abs(node_xyz[:, 1]) < 1e-6)[0]
    dof_z = 3 * y0_nodes + 2
    reaction_force_z = K[dof_z, :].dot(u).sum()
    log['reaction'] = time.time() - t0
    print(f" - Total reaction force (z @ y=0): {reaction_force_z:.6f} N")

    print("\nRuntime summary:")
    for k, v in log.items():
        print(f" - {k:<15}: {v:.4f} sec")
    print(f" - TOTAL           : {time.time() - total_start:.4f} sec")

if __name__ == "__main__":
    main()
