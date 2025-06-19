# ReVISE: Learning to Refine at Test-Time via Intrinsic Self-Verification

This is the official implementation of ReVISE, a method for test-time refinement through intrinsic self-verification.

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/revise.git
cd revise
```

### 2. Set up your Python environment

You can use either `venv` or Conda to set up your environment. Here are the differences:
- **venv**: A lightweight, built-in Python tool for creating virtual environments. Recommended if you already have Python installed and prefer minimal dependencies.
- **Conda**: A more robust environment manager that handles both Python and non-Python dependencies. Recommended if you are working on a system without Python pre-installed or need to manage complex dependencies.

For this project, we recommend using Conda for its ease of use and compatibility with the provided `environment.yml` file. However, you can use `venv` if you prefer.

#### Using venv
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
# or
conda env create -f environment.yml
conda activate revise

# Install flash-attn
pip install flash-attn --no-build-isolation
```
