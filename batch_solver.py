import os
import re
import time
import sys
from tqdm import tqdm
from solver import run_dual_analysis

def extract_metadata_from_filename(filename):
    pattern = r"r1=([\d.]+)_r2=([\d.]+)_r3=([\d.]+)_r4=([\d.]+)_r5=([\d.]+)_vol=([\d.]+)"
    match = re.search(pattern, filename)
    if match:
        return [float(val) for val in match.groups()]
    else:
        return [None] * 6

def run_batch_analysis(mesh_dir, file_list, output_path):
    header = [
        "filename", "r1", "r2", "r3", "r4", "r5", "step_volume", "gmsh_volume",
        "nodes", "elements", "pre_time", "biax_time", "shear_time",
        "total_time", "biax_react", "E_eff", "shear_react", "G_eff"
    ]
    col_widths = {
        "filename": 75,
        "r1": 10, "r2": 10, "r3": 10, "r4": 10, "r5": 10,
        "step_volume": 14, "gmsh_volume": 14,
        "nodes": 10, "elements": 10,
        "pre_time": 12, "biax_time": 12, "shear_time": 12,
        "total_time": 12, "biax_react": 14, "E_eff": 14,
        "shear_react": 14, "G_eff": 14
    }

    def format_row(values, is_header=False):
        row = []
        for i, val in enumerate(values):
            key = header[i]
            width = col_widths[key]
            if is_header:
                row.append(f"{val:>{width}}")
            elif isinstance(val, float):
                row.append(f"{val:>{width}.4f}")
            else:
                row.append(f"{str(val):>{width}}")
        return " ".join(row)

    with open(output_path, "w") as fout:
        fout.write(format_row(header, is_header=True) + "\n")
        fout.write("-" * (sum(col_widths.values()) + len(col_widths) - 1) + "\n")

        for idx, fname in enumerate(tqdm(file_list, desc="Running FEM Batch")):
            mesh_path = os.path.join(mesh_dir, fname)
            try:
                t_start = time.time()
                results = run_dual_analysis(mesh_path)
                total_time = time.time() - t_start

                r1, r2, r3, r4, r5, step_vol = extract_metadata_from_filename(fname)
                pre_log = results["preprocess"]
                biaxial_reaction = results["biaxial"]["reaction"]
                shear_reaction = results["shear"]["reaction"]

                with open(mesh_path, 'r') as f:
                    for line in f:
                        if line.strip() == "$Nodes":
                            n_nodes = int(next(f).strip())
                        elif line.strip() == "$Elements":
                            n_elements = int(next(f).strip())
                            break

                E_eff = biaxial_reaction / (100 * 100 * 0.002)
                G_eff = shear_reaction / (100 * 100 * 0.002)

                row = [
                    fname, r1, r2, r3, r4, r5, step_vol,
                    results["volume"], n_nodes, n_elements,
                    pre_log["preprocess_total"],
                    results["biaxial"]["runtime"], results["shear"]["runtime"],
                    total_time, biaxial_reaction, E_eff, shear_reaction, G_eff
                ]
                fout.write(format_row(row) + "\n")
                fout.flush()
            except Exception as e:
                print(f"[ERROR] Failed to process {fname}: {e}")

if __name__ == "__main__":
    # ex(command): python batch_solver.py node31_list.txt result_node31.txt
    if len(sys.argv) != 3:
        print("Usage: python batch_solver.py <file_list.txt> <output_result.txt>")
        sys.exit(1)

    file_list_path = sys.argv[1]
    output_path = sys.argv[2]

    with open(file_list_path, "r") as f:
        mesh_files = [line.strip() for line in f if line.strip()]

    run_batch_analysis("./examples/fcc_mesh", mesh_files, output_path)
