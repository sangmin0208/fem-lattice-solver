import os
import multiprocessing
n_threads = multiprocessing.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(n_threads)   
os.environ["MKL_NUM_THREADS"] = str(n_threads)   
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import time
import numpy as np
from scipy.sparse import coo_matrix
from femcore import define_material, prepare_boundary_conditions, prepare_shear_boundary_conditions, compute_mesh_volume
import sys
sys.path.append("./cpp/build")
from tetra4_fem_core import read_gmsh_tetra4_mesh, assemble_tetra4_triplets
from pypardiso import spsolve as direct_solve


print(f"\nn_threads = {n_threads}")

def preprocess_fem_system(mesh_path, scale=1.0, E=775.43, nu=0.3):
    runtime_log = {}
    total_start = time.time()

    print("Importing mesh...")
    t0 = time.time()
    nodes, elements = read_gmsh_tetra4_mesh(mesh_path, scale)
    runtime_log['mesh_import'] = time.time() - t0
    n_nodes = nodes.shape[0]
    n_elements = elements.shape[0]

    print("Calculating volume...")
    t0 = time.time()
    node_xyz = nodes[:, 1:4]
    total_volume = compute_mesh_volume(node_xyz, elements)
    runtime_log['volume_calc'] = time.time() - t0

    print("Defining material...")
    t0 = time.time()
    lam, mu = define_material(E, nu)
    runtime_log['material'] = time.time() - t0

    print("Assembling stiffness matrix...")
    t0 = time.time()
    n_dofs = 3 * n_nodes
    rows, cols, data = assemble_tetra4_triplets(nodes, elements, lam, mu)
    
    K = coo_matrix((data, (rows, cols)), shape=(n_dofs, n_dofs)).tocsr()
    
    runtime_log['stiffness_assembly'] = time.time() - t0

    runtime_log['preprocess_total'] = time.time() - total_start
    runtime_log['n_nodes'] = n_nodes
    runtime_log['n_elements'] = n_elements
    runtime_log['n_dofs'] = n_dofs
    runtime_log['volume'] = total_volume

    return node_xyz, nodes, total_volume, K, n_dofs, runtime_log

def solve_fem_system(K, F, fixed_dofs, u_fixed):
    n_dofs = F.shape[0]
    u = np.zeros(n_dofs)
    mask = np.ones(n_dofs, dtype=bool)
    mask[fixed_dofs] = False
    free_dofs = np.nonzero(mask)[0]
    
    Kff = K[free_dofs][:, free_dofs]
    Kfc = K[free_dofs][:, fixed_dofs]
    Ff = F[free_dofs] - Kfc @ u_fixed

    u[fixed_dofs] = u_fixed
    u[free_dofs] = direct_solve(Kff, Ff)
    return u

def compute_reaction(K, u, node_xyz, axis='y', face='y0'):
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    idx = axis_map[axis]

    if face == 'y0':
        face_nodes = np.where(np.abs(node_xyz[:, 1]) < 1e-8)[0]  # y=0 face
    elif face == 'y100':
        face_nodes = np.where(np.abs(node_xyz[:, 1] - 100.0) < 1e-8)[0]
    else:
        raise ValueError(f"Unknown face: {face}")

    dof_indices = 3 * face_nodes + idx
    reaction = K[dof_indices, :].dot(u).sum()
    return reaction

def run_dual_analysis(mesh_path):
    print(f"\n===== Running Dual FEM Analysis on {mesh_path} =====\n")
    total_start = time.time()

    print("Preprocessing FEM system...")
    node_xyz, nodes, volume, K, n_dofs, pre_log = preprocess_fem_system(mesh_path)

    results = {}

    # --- Preprocess Summary ---
    print("\n--- Preprocess Result ---")
    print(f"  Mesh file           : {mesh_path}")
    print(f"  Number of nodes     : {pre_log['n_nodes']}")
    print(f"  Number of elements  : {pre_log['n_elements']}")
    print(f"  Degrees of freedom  : {pre_log['n_dofs']}")
    print(f"  Mesh volume         : {pre_log['volume']:.4f} mm³")
    print(f"  Stiffness matrix    : {K.shape[0]} x {K.shape[1]}, nnz = {K.nnz}")
    print(f"  Preprocess time     : {pre_log['preprocess_total']:.4f} sec")

    # --- Biaxial Test ---
    print("\n--- Biaxial Test ---")
    print("Preparing boundary conditions...")
    t0 = time.time()
    F_b, fixed_b, u_b, y0_mask, y100_mask, x_fix_mask, z_fix_mask = prepare_boundary_conditions(node_xyz, nodes)
    print("Solving linear system...")
    u_b_full = solve_fem_system(K, F_b, fixed_b, u_b)
    print("Computing reaction force...")
    reaction_b = compute_reaction(K, u_b_full, node_xyz, axis='y', face='y0')

    results['biaxial'] = {
        'reaction': reaction_b,
        'runtime': time.time() - t0,
        'bc_nodes': {
            'y=0 (fixed)': np.count_nonzero(y0_mask),
            'y=100 (disp)': np.count_nonzero(y100_mask),
            'x-dir fixed': np.count_nonzero(x_fix_mask),
            'z-dir fixed': np.count_nonzero(z_fix_mask),
            'total fixed DOFs': len(fixed_b),
        }
    }
    print(f"Biaxial reaction (y @ y=0): {reaction_b:.6f} N")

    # --- Shear Test ---
    print("\n--- Shear Test ---")
    print("Preparing boundary conditions...")
    t0 = time.time()
    F_s, fixed_s, u_s = prepare_shear_boundary_conditions(node_xyz)
    print("Solving linear system...")
    u_s_full = solve_fem_system(K, F_s, fixed_s, u_s)
    print("Computing reaction force...")
    reaction_s = compute_reaction(K, u_s_full, node_xyz, axis='z', face='y0')

    results['shear'] = {
        'reaction': reaction_s,
        'runtime': time.time() - t0,
        'bc_nodes': {
            'total fixed DOFs': len(fixed_s),
        }
    }
    print(f"Shear reaction (z @ y=0): {reaction_s:.6f} N")

    total_runtime = time.time() - total_start

    # --- Runtime Summary ---
    print("\n--- Runtime Summary ---")
    for test, info in results.items():
        print(f"\n  [{test.capitalize()} Test]")
        print(f"    Reaction Force     : {info['reaction']:.6f} N")
        print(f"    Solve Time         : {info['runtime']:.4f} sec")
        print(f"    Boundary Nodes     :")
        for label, count in info['bc_nodes'].items():
            print(f"      - {label:<18}: {count}")
    print(f"\n  Preprocessing Time   : {pre_log['preprocess_total']:.4f} sec")
    print(f"  TOTAL Runtime        : {total_runtime:.4f} sec\n")
    results['volume'] = volume
    results['preprocess'] = pre_log
    return results

"""
mesh_path = "./examples/fcc_mesh/fcc[r1=8.871_r2=2.471_r3=8.721_r4=8.098_r5=4.138_vol=133998.761].msh"
run_dual_analysis(mesh_path)
"""


#/home/sangmin/fem-lattice-solver/examples/fcc_mesh/fcc[r1=8.477_r2=8.891_r3=9.780_r4=9.440_r5=4.884_vol=188795.886].msh