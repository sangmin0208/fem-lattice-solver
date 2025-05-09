import sys
sys.path.append("./cpp/build")

import numpy as np
from femcore.assembly import compute_tetra4_stiffness_numba, define_material
from read_gmsh_cpp import read_gmsh_tetra4_mesh, compute_tetra4_stiffness_cpp

# 경로 및 재료 상수 설정
path = "./examples/fcc[r1=3.254_r2=6.388_r3=8.775_r4=8.351_r5=8.427_vol=117916.853].msh"
scale = 1.0
E, nu = 775.43, 0.3
lam, mu = define_material(E, nu)

# 메쉬 로딩
nodes, elements = read_gmsh_tetra4_mesh(path, scale)
Xe = nodes[elements[0], 1:4]  # 첫 번째 요소 좌표만 추출

# Ke 계산
Ke_py = compute_tetra4_stiffness_numba(Xe, lam, mu)
Ke_cpp = compute_tetra4_stiffness_cpp(Xe, lam, mu)

# 비교
diff = np.linalg.norm(Ke_py - Ke_cpp)
is_close = np.allclose(Ke_py, Ke_cpp, rtol=1e-10, atol=1e-12)

print("===== Single Element Ke Comparison =====")
print(f"Frobenius norm  : {diff:.6e}")
print(f"allclose match  : {is_close}")
