# FEM Project Setup Instructions (macOS / Linux + conda)

This guide describes how to set up the environment and dependencies required to run the FEM lattice simulation code on macOS or Linux using `conda`.

---

## 1. Clone Project Structure
Assume the directory structure:

```
FEM_Project/
├── main.py
├── batch_solver.py
├── meshes/                # place your .msh files here
├── femcore/
│   ├── __init__.py
│   ├── io.py
│   ├── assembly.py
│   └── utils.py
```

---

## 2. Set Up Conda Environment

```bash
conda create -n fem_env python=3.10 -y
conda activate fem_env
```

Install required packages:

```bash
conda install numpy scipy numba -y
conda install conda-forge::pypardiso
```

If `pypardiso` installation fails or is unavailable on your system, the code will fallback to SciPy's solver.

Optional (for faster solves on supported CPUs):
```bash
conda install anaconda::mkl-service
```

---

## 3. Running the Code

### Run single case:
```bash
python main.py
```

### Run batch solver:
```bash
python batch_solver.py
```

Ensure that `.msh` files are located in the `./meshes/` directory.

---

## 4. Notes

- All modules are located under `femcore/`.
- The environment is cross-platform: the same setup works on both macOS and most Linux distributions.
- No GPU-specific libraries are required in this baseline version.
- All path references in code are relative (recommended to run scripts from project root).

---

## 5. Future Extension
If running on Linux cluster or HPC:
- Use `joblib` or `multiprocessing` in `batch_solver.py` for parallel runs.
- Set up conda environment as part of job submission script (e.g., SLURM).

---

### Confirmed Working Platforms:

