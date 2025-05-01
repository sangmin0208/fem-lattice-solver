import numpy as np

def read_gmsh_mesh(filename, scale=1.0):
    nodes = []
    elements = []
    with open(filename, 'r') as f:
        line = f.readline()
        while line:
            if line.strip() == "$Nodes":
                num_nodes = int(f.readline())
                for _ in range(num_nodes):
                    parts = f.readline().split()
                    node_id = int(parts[0])
                    x, y, z = map(float, parts[1:4])
                    nodes.append([node_id, x * scale, y * scale, z * scale])
            elif line.strip() == "$Elements":
                num_elements = int(f.readline())
                for _ in range(num_elements):
                    parts = f.readline().split()
                    elem_type = int(parts[1])
                    if elem_type == 4:
                        tet = list(map(int, parts[-4:]))
                        elements.append([v-1 for v in tet])
                break
            line = f.readline()
    return np.array(nodes, dtype=np.float64), np.array(elements, dtype=np.int32)

