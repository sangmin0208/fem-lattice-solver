import sys
sys.path.append("./cpp/build")

import time
import numpy as np
from scipy.sparse import coo_matrix
from femcore.assembly import assemble_global_stiffness, define_material
from read_gmsh_cpp import read_gmsh_tetra4_mesh, assemble_tetra4_triplets

# 설정
path = "./examples/fcc[r1=3.254_r2=6.388_r3=8.775_r4=8.351_r5=8.427_vol=117916.853].msh"
scale = 1.0
E, nu = 775.43, 0.3
lam, mu = define_material(E, nu)

# 메쉬 불러오기
nodes, elements = read_gmsh_tetra4_mesh(path, scale)
n_dofs = 3 * nodes.shape[0]

# 방법 1: Python (Numba)
t0 = time.time()
K_py = assemble_global_stiffness(nodes, elements, lam, mu, verbose=False)
t1 = time.time()

# 방법 2: C++ (Pybind11)
t2 = time.time()
rows, cols, data = assemble_tetra4_triplets(nodes, elements, lam, mu)
K_cpp = coo_matrix((data, (rows, cols)), shape=(n_dofs, n_dofs)).tocsr()
t3 = time.time()

# 결과 비교
print("===== Stiffness Assembly Comparison =====")
print(f"[Python] Time   : {t1 - t0:.4f} sec")
print(f"[C++]    Time   : {t3 - t2:.4f} sec")
print(f"Nonzeros (py)   : {K_py.nnz}")
print(f"Nonzeros (cpp)  : {K_cpp.nnz}")

fro_diff = (K_py - K_cpp).power(2).sum() ** 0.5
print(f"Frobenius norm  : {fro_diff:.6e}")