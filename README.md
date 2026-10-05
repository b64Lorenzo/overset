# Overset Grid Interface Reconstruction

Numerical proof-of-concept repository accompanying the MSc research work

> **Reconstruction Techniques for Elliptic Problems on Overlapping and Disjoint Domains**
>
> Lorenzo Zambelli  
> MSc Thesis / Research Report  
> University of Groningen

---

# Overview

This repository contains the numerical proof-of-concept implementations developed during the research work on reconstruction techniques for elliptic problems solved on overlapping and disjoint computational domains.

The objective of the work is to investigate how the error introduced by artificial domain decomposition can be interpreted as a homogeneous solution of the governing elliptic operator and subsequently reconstructed.

The repository contains:

- One-dimensional overlap reconstruction;
- Two-dimensional overlap reconstruction;
- Four-domain overlap reconstruction;
- One-dimensional interface-flux reconstruction;
- Two-dimensional interface-flux reconstruction;
- Finite-element validation studies;
- Mesh-convergence studies;
- Modal-convergence studies;
- Conditioning studies;
- Overlap-width sensitivity studies.

The code was developed as a research code accompanying the thesis and associated numerical experiments rather than as a general-purpose software package.

---

# Scientific Background

The principal elliptic model considered throughout the repository is

```math
k u
-
\nabla \cdot
\left(
h \nabla u
\right)
=
\nabla \cdot
\left(
p \nabla f
\right).
```

In one dimension the problem becomes

```math
k u
-
\frac{d}{dx}
\left(
h \frac{du}{dx}
\right)
=
\frac{d}{dx}
\left(
p \frac{df}{dx}
\right).
```

The solution is viewed as

```math
u
=
u_p
+
u_h,
```

where

```math
u_p
```

is the particular solution and

```math
u_h
```

is the homogeneous contribution introduced by artificial interface conditions.

The reconstruction procedures implemented in this repository attempt to identify and remove this homogeneous component.

---

# Reconstruction Methodologies

Two families of reconstruction methods are implemented.

---

## 1. Overlap Reconstruction

The overlap reconstruction methodology exploits observations inside an overlap region shared by neighboring subdomains.

General workflow:

1. Compute a benchmark solution on the undecomposed domain.
2. Solve independent subdomain problems.
3. Extract observations inside the overlap region.
4. Assemble a reconstruction system.
5. Estimate homogeneous amplitudes.
6. Reconstruct the homogeneous correction.
7. Remove the correction from the local solutions.
8. Blend corrected subdomain solutions.
9. Compare against the benchmark solution.

Implemented in:

```text
overlapping1d.py
overlapping2d.py
overlapping2d_4dom.py
```

---

## 2. Flux Reconstruction

The flux reconstruction methodology does not require overlap observations.

Instead, reconstruction is performed directly from interface jumps.

For a given interface,

```math
J_u
=
u^{-}
-
u^{+}
```

and

```math
J_q
=
q^{-}
-
q^{+}
```

are computed, where

```math
q = h u_x.
```

These jumps define a small reconstruction system whose solution provides the amplitudes of the missing homogeneous modes.

Implemented in:

```text
flux1d.py
flux2d.py
```

---

# Repository Structure

```text
.
├── proofOfConcepts/
│
│   ├── overlapping1d.py
│   ├── overlapping2d.py
│   ├── overlapping2d_4dom.py
│   │
│   ├── flux1d.py
│   └── flux2d.py
│
├── src/
│
│   ├── problems/
│   │   ├── elliptic1d.py
│   │   └── elliptic2d.py
│   │
│   ├── solver/
│   │   ├── fem1d.py
│   │   └── fem2d.py
│   │
│   ├── reconstruction/
│   │   ├── overlap1d.py
│   │   ├── overlap2d.py
│   │   ├── flux.py
│   │   └── flux_2d.py
│   │
│   ├── plotlib/
│   │   ├── solution_plots.py
│   │   ├── overlap_plots.py
│   │   ├── validation_plots.py
│   │   ├── solution_plots_2d.py
│   │   ├── overlap_plots_2d.py
│   │   └── validation_plots_2d.py
│   │
│   └── utils/
│       ├── logger.py
│       └── utils.py
│
├── test/
│   ├── test_elliptic1d_validation.py
│   ├── test_elliptic2d_validation.py
│   ├── test_fem1d_validation.py
│   └── test_fem2d_validation.py
│
├── runs/
│
├── requirements.txt
│
└── README.md
```

---

# Main Numerical Studies

All numerical experiments discussed in the thesis are located in

```text
proofOfConcepts/
```

---

## overlapping1d.py

One-dimensional overlap reconstruction validation.

Features:

- benchmark solution generation;
- overlap observation construction;
- homogeneous-mode reconstruction;
- weighted overlap blending;
- convergence studies;
- overlap-width studies;
- conditioning studies.

Run:

```bash
python proofOfConcepts/overlapping1d.py
```

or

```bash
python -m proofOfConcepts.overlapping1d
```

---

## overlapping2d.py

Two-dimensional overlap reconstruction validation.

Features:

- two-dimensional overlap reconstruction;
- observation-strategy comparison;
- modal convergence;
- overlap-width studies;
- reconstruction diagnostics.

Run:

```bash
python proofOfConcepts/overlapping2d.py
```

or

```bash
python -m proofOfConcepts.overlapping2d
```

---

## overlapping2d_4dom.py

Four-domain overlap reconstruction study.

Features:

- multiple overlapping domains;
- multidirectional reconstruction;
- multi-interface coupling;
- complex overlap geometries.

Run:

```bash
python proofOfConcepts/overlapping2d_4dom.py
```

or

```bash
python -m proofOfConcepts.overlapping2d_4dom
```

---

## flux1d.py

One-dimensional interface-flux reconstruction.

Features:

- interface jump reconstruction;
- flux jump reconstruction;
- homogeneous correction recovery;
- conditioning studies;
- interface diagnostics.

Run:

```bash
python proofOfConcepts/flux1d.py
```

or

```bash
python -m proofOfConcepts.flux1d
```

---

## flux2d.py

Two-dimensional interface-flux reconstruction.

Features:

- modal flux reconstruction;
- interface-jump projection;
- analytical correction modes;
- interface-continuity reconstruction.

Run:

```bash
python proofOfConcepts/flux2d.py
```

or

```bash
python -m proofOfConcepts.flux2d
```

---

# Validation Suite

Validation scripts are located in

```text
test/
```

and verify the correctness of the model problems and finite-element discretizations.

---

## Elliptic Problem Validation

```bash
python test/test_elliptic1d_validation.py
```

```bash
python test/test_elliptic2d_validation.py
```

These tests verify:

- forcing functions;
- analytical derivatives;
- coefficient implementations;
- consistency of the governing equations.

---

## Finite Element Validation

```bash
python test/test_fem1d_validation.py
```

```bash
python test/test_fem2d_validation.py
```

These tests verify:

- finite-element implementation;
- convergence rates;
- residual consistency;
- benchmark solutions;
- interpolation accuracy.

---

# Requirements

The repository requires:

```text
Python 3.11+
NumPy
SciPy
Matplotlib
Pandas
scikit-fem
```

---

# Installation

Clone the repository:

```bash
git clone <repository-url>
```

Enter the repository:

```bash
cd overset_grid
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install numpy scipy matplotlib pandas scikit-fem
```

or

```bash
pip install -r requirements.txt
```

---

# Running a Study

Example:

```bash
python proofOfConcepts/overlapping1d.py
```

or

```bash
python proofOfConcepts/flux1d.py
```

The scripts automatically create a timestamped output directory and generate all associated diagnostics and plots.

---

# Output Structure

Each execution creates a dedicated run directory:

```text
runs/
└── overlap1d_YYYYMMDD_HHMMSS/
```

or

```text
runs/
└── flux1d_YYYYMMDD_HHMMSS/
```

depending on the experiment.

Typical contents include:

```text
run.log

results.csv

convergence.csv

overlap_study.csv

solution_with_error.png

overlap_detail.png

homogeneous_correction.png

reconstruction_residual.png

modal_observation_convergence.png

overlap_width_error.png

condition_vs_h.png
```

The exact set of figures depends on the script being executed.

---

# Main Components

## problems

Contains mathematical model definitions:

```text
elliptic1d.py
elliptic2d.py
```

---

## solver

Finite-element solvers:

```text
fem1d.py
fem2d.py
```

---

## reconstruction

Implementation of overlap and flux reconstruction methods:

```text
overlap1d.py
overlap2d.py

flux.py
flux_2d.py
```

---

## plotlib

Publication-quality visualization routines:

```text
solution_plots.py
overlap_plots.py
validation_plots.py

solution_plots_2d.py
overlap_plots_2d.py
validation_plots_2d.py
```

---

## utils

Utility routines:

```text
logger.py
utils.py
```

---

# Expected Results

A successful reconstruction should satisfy

```math
\|u_{\mathrm{rec}}-u_{\mathrm{exact}}\|
<
\|u_{\mathrm{unc}}-u_{\mathrm{exact}}\|
```

demonstrating that the interface-induced homogeneous error has been identified and removed.

Typical studies show substantial reductions in reconstruction error after application of the overlap or flux correction procedures.

---

# References

The methodology draws upon ideas from

- elliptic boundary-value problems;
- homogeneous solution reconstruction;
- domain decomposition methods;
- overset grids;
- finite-element methods;
- the Variational Boussinesq Model.

For detailed theoretical derivations, numerical analysis, and discussion, refer to the MSc thesis.

---

# Citation

If you use this repository, please cite:

```text
Zambelli, L.

Reconstruction Techniques for Elliptic Problems
on Overlapping and Disjoint Domains.

MSc Thesis / Research Report.

University of Groningen.

2026.
```

---

# Author

**Lorenzo Zambelli**

MSc Applied Mathematics

University of Groningen

Research topic:

```text
Overlap Reconstruction
Interface Error Reconstruction
Domain Decomposition
Overset Grids
Finite Elements
Variational Boussinesq Models
```