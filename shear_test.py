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

import sys
sys.path.append("./cpp/build")
import read_gmsh_cpp

def main():
    total_start = time.time()
    runtime_log = {}

    # Path to Gmsh-generated mesh
    path = "./examples/fcc[r1=4.575_r2=9.407_r3=6.630_r4=6.979_r5=6.881_vol=122030.945].msh"
    scale = 1.0

    print("Importing mesh via C++ module...")
    t0 = time.time()
    nodes, elements = read_gmsh_cpp.read_gmsh_tetra4_mesh(path, scale)
    runtime_log['mesh_import'] = time.time() - t0
    print(f"Mesh import time: {runtime_log['mesh_import']:.4f} sec")
    print(f"Number of nodes       : {nodes.shape[0]}")
    print(f"Number of elements    : {elements.shape[0]}")
    
    node_xyz = nodes[:, 1:4]

    print("Computing mesh volume...")
    t0 = time.time()
    volume = compute_mesh_volume(node_xyz, elements)
    runtime_log['volume'] = time.time() - t0
    print(f" - Volume = {volume:.3f} mm³")

    print("Defining material...")
    t0 = time.time()
    lam, mu = define_material(E=775.43, nu=0.3)
    runtime_log['material'] = time.time() - t0
    print(f" - Lamé parameters: λ = {lam:.3f}, μ = {mu:.3f}")

    print("Assembling global stiffness matrix...")
    t0 = time.time()
    K = assemble_global_stiffness(nodes, elements, lam, mu)
    runtime_log['assemble'] = time.time() - t0
    print(f" - K shape: {K.shape}, nnz: {K.nnz}")

    print("Applying shear boundary conditions...")
    t0 = time.time()
    F, fixed_dofs, u_fixed = prepare_shear_boundary_conditions(node_xyz)
    runtime_log['bc'] = time.time() - t0
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
    runtime_log['solve'] = time.time() - t0

    print("Computing reaction force at y=0 face...")
    t0 = time.time()
    y0_nodes = np.where(np.abs(node_xyz[:, 1]) < 1e-6)[0]
    dof_z = 3 * y0_nodes + 2
    reaction_force_z = K[dof_z, :].dot(u).sum()
    runtime_log['reaction'] = time.time() - t0
    print(f" - Total reaction force (z @ y=0): {reaction_force_z:.6f} N")

    print("\nRuntime summary:")
    for k, v in runtime_log.items():
        print(f" - {k:<15}: {v:.4f} sec")
    print(f" - TOTAL           : {time.time() - total_start:.4f} sec")

if __name__ == "__main__":
    main()
