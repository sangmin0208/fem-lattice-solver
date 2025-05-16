# mesh_file_list.py

import os

def save_mesh_filenames(mesh_dir, output_txt="mesh_file_list.txt"):
    files = sorted(f for f in os.listdir(mesh_dir) if f.endswith(".msh"))
    with open(output_txt, "w") as fout:
        for f in files:
            fout.write(f + "\n")
    print(f"✅ Saved {len(files)} mesh filenames to {output_txt}")

if __name__ == "__main__":
    save_mesh_filenames("./examples/fcc_mesh")
