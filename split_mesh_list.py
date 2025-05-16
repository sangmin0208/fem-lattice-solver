import os
from math import ceil

def split_files(mesh_dir, out1="node30_list.txt", out2="node31_list.txt"):
    files = sorted(f for f in os.listdir(mesh_dir) if f.endswith(".msh"))
    half = ceil(len(files) / 2)

    node30_files = files[:half]
    node31_files = files[half:]

    with open(out1, "w") as f:
        f.writelines(f"{fn}\n" for fn in node30_files)

    with open(out2, "w") as f:
        f.writelines(f"{fn}\n" for fn in node31_files)

    print(f"🔹 {out1}: {len(node30_files)} files")
    print(f"🔹 {out2}: {len(node31_files)} files")

if __name__ == "__main__":
    split_files("./examples/fcc_mesh")
