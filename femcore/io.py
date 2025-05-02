import numpy as np

def read_gmsh_mesh(filename, scale=1.0):
    nodes = []      # List to store node data [ID, x, y, z]
    elements = []   # List to store tetrahedral elements (node indices)
    
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
            
            # Parse element section (only type 4: tetrahedron)
            elif line.strip() == "$Elements":
                num_elements = int(f.readline())
                for _ in range(num_elements):
                    parts = f.readline().split()
                    elem_type = int(parts[1])
                    if elem_type == 4:  # 4 = 4-node tetrahedron
                        tet = list(map(int, parts[-4:]))    # Last 4 entries are node indices
                        elements.append([v-1 for v in tet]) # Convert to 0-based indexing
                break
            line = f.readline()
    return np.array(nodes, dtype=np.float64), np.array(elements, dtype=np.int32)

