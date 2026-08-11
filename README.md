# Overset Grid Interface Reconstruction – Proof of Concept

This repository contains a collection of one-dimensional proof-of-concept implementations for studying interface-error reconstruction in overset/domain-decomposed PDE problems.

The main idea is that errors introduced by artificial interface conditions can be interpreted as missing homogeneous solutions of the governing operator and subsequently reconstructed.

Two reconstruction strategies are included:

1. **Solution-Based Reconstruction**
   - Uses local observations away from the interface.
   - Reconstructs homogeneous modes explicitly.
   - Supports:
     - Single-mode reconstruction.
     - Multiple independent modes.
     - Multiple modes with a single amplitude.

2. **Flux-Based Reconstruction**
   - Uses interface solution and flux jumps.
   - Solves a small 2×2 reconstruction system.
   - Does not require observation points.

---

# Repository Structure

```text
.
├── overset1d.py
├── validation1d_flux.py
├── utils/
│   ├── interface_correction_1d.py
│   └── logger.py
├── runs/
└── pyproject.toml
```

Main scripts:

| File | Description |
|--------|-------------|
| `overset1d.py` | Solution-based reconstruction |
| `validation1d_flux.py` | Flux-based reconstruction |

---

# Requirements

The project uses:

- Python 3.11+
- NumPy
- SciPy
- Matplotlib
- Pandas
- scikit-fem

Recommended package manager:

**uv**

---

# Installing uv

Linux/macOS:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify installation:

```bash
uv --version
```

---

# Create Environment

From the repository root:

```bash
uv venv
```

Activate:

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows:

```powershell
.venv\Scripts\activate
```

---

# Install Dependencies

If a `pyproject.toml` is available:

```bash
uv sync
```

Otherwise:

```bash
uv pip install \
numpy \
scipy \
matplotlib \
pandas \
scikit-fem
```

---

# Running the Solution-Based Reconstruction

Run:

```bash
python overset1d.py
```

or

```bash
uv run python overset1d.py
```

The script:

1. Computes a reference solution on the full domain.
2. Splits the domain into two subdomains.
3. Perturbs the interface boundary condition.
4. Solves each subdomain independently.
5. Reconstructs the missing homogeneous component.
6. Compares the reconstructed solution with the reference solution.

---

# Running the Flux-Based Reconstruction

Run:

```bash
python validation1d_flux.py
```

or

```bash
uv run python validation1d_flux.py
```

The script:

1. Computes the interface solution jump

```math
J_u = u_A(x_I)-u_B(x_I)
```

2. Computes the interface flux jump

```math
J_q = q_A-q_B
```

3. Constructs unit homogeneous responses.

4. Solves the interface system

```math
\begin{bmatrix}
1 & -1 \\
q_{\phi A} & -q_{\phi B}
\end{bmatrix}
\begin{bmatrix}
\varepsilon_A \\
\varepsilon_B
\end{bmatrix}
=
\begin{bmatrix}
J_u\\
J_q
\end{bmatrix}
```

5. Applies the correction.

---

# Output

Each run generates a timestamped directory:

```text
runs/
└── run_YYYYMMDD_HHMMSS/
```

containing:

```text
results.csv
run.log
solution_with_error.png
exponential_fit_comparison.png
```

For the flux method:

```text
solution_with_error_flux.png
exponential_fit_comparison_flux.png
```

---

# Key Parameters

The main parameters are located near the beginning of each script.

```python
xI = 64
```

Artificial interface location.

```python
n_left = 400
n_right = 400
```

Subdomain resolutions.

```python
noise_amplitude = 6.4
```

Magnitude of interface perturbation.

```python
k_reaction = 25.0
```

Reaction coefficient.

---

# Solution-Based Reconstruction Modes

Single mode:

```python
nmodes = 1
single_amplitude = False
```

Multiple independent modes:

```python
nmodes = 3
single_amplitude = False
```

Multiple modes with one amplitude:

```python
nmodes = 3
single_amplitude = True
```

Recommended observation-point spacing:

```python
2*dx
```

between neighboring observation points.

Observation points placed too far from the interface tend to contain stronger contributions from the particular solution, leading to poorer modal reconstruction.

---

# Typical Results

Constant coefficients:

```text
L2 before correction ≈ 1.38e+01
L2 after correction  ≈ 4.40e−03
```

Variable coefficients:

```text
L2 before correction ≈ 1.31e+01
L2 after correction  ≈ 2.27e−01
```

Three modes per side:

```text
L2 after correction ≈ 1.6e−01
```

---
