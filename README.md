# NPDAL-n: A New Primal-Dual Algorithm for General Convex-Concave Saddle-Point Problems

![MATLAB](https://img.shields.io/badge/MATLAB-R2023b-blue.svg)
![Python](https://img.shields.io/badge/Python-3.x-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

This repository contains the official MATLAB and Python implementations for the numerical experiments in the following paper:

> **A New Primal-Dual Algorithm with Two Convex Combinations and Linesearch for General Convex-Concave Saddle-Point Problems**  
> Zexian Liu, Shuning Liu, Jialong Li

The code is provided to reproduce the numerical results reported in the manuscript and to facilitate further research on primal-dual algorithms for general convex-concave saddle-point problems.

---

## 📖 1. Problem Formulation

We consider the following general convex-concave saddle-point problem:

$$
\min_{x\in\mathbb{R}^q}\max_{y\in\mathbb{R}^p}
\mathcal{L}(x,y)
=
g(x)+\Phi(x,y)-f^*(y).
$$

where \(g:\mathbb{R}^q\rightarrow(-\infty,+\infty]\) and
\(f^*:\mathbb{R}^p\rightarrow(-\infty,+\infty]\) are proper, closed,
and convex extended real-valued functions.

The coupling function

$$
\Phi:\operatorname{dom}(g)\times\operatorname{dom}(f^*)
\rightarrow\mathbb{R}
$$

is continuously differentiable, convex with respect to \(x\) for each
fixed \(y\), and concave with respect to \(y\) for each fixed \(x\).

The proposed **NPDAL-n** algorithm is designed to solve this class of
general convex-concave saddle-point problems by incorporating two convex
combinations and an adaptive linesearch strategy.

For strongly convex problems, an accelerated variant, **aNPDAL-n**, is
also developed.

---

## 🚀 2. Algorithms

This repository contains implementations of the proposed algorithms and
the baseline methods used in the numerical experiments.

### Proposed Methods

- **NPDAL-n**: The proposed new primal-dual algorithm with two convex
  combinations and an adaptive linesearch strategy.

- **aNPDAL-n**: An accelerated variant of NPDAL-n for strongly convex
  problems, achieving an \(\mathcal{O}(1/N^2)\) convergence rate in the
  corresponding setting.

- **NPGM**: A linesearch-free proximal gradient method obtained as a
  special case of NPDAL-n for composite convex optimization problems.

### Baseline Methods

The following existing algorithms are included for performance comparison:

- **PDAc-L**
- **PDB**
- **aPDAc-L**
- **aPDB**
- **aGRAAL**
- **aPGMc**

The specific algorithms used in each numerical experiment are described
in Section 6.

---

## 📂 3. Repository Structure

The repository currently provides the numerical experiment codes in two
compressed archives:

```text
NPDAL-n-for-General-Convex-Concave-Saddle-Point-Problems/
│
├── QCQP.rar       # MATLAB codes for QCQP experiments
├── SLR.rar        # Python codes for SLR experiments
├── LICENSE
└── README.md
```

### QCQP

`QCQP.rar` contains the MATLAB implementations and numerical experiment
codes for the **Quadratically Constrained Quadratic Programming (QCQP)**
experiments.

### SLR

`SLR.rar` contains the Python implementations and numerical experiment
codes for the **Sparse Logistic Regression (SLR)** experiments.

Please extract the corresponding `.rar` archive before running the
experiments.

---

## ⚙️ 4. Requirements

The numerical experiments are implemented in two different programming
environments.

### 4.1 QCQP Experiments — MATLAB

The QCQP experiments require:

- MATLAB R2023b or a compatible version.
- [CVX](http://cvxr.com/cvx/) for convex optimization modeling.
- [MOSEK](https://www.mosek.com/) for computing the reference optimal
  values.

The QCQP experiments reported in the paper use CVX with the MOSEK solver
to obtain the reference optimal values.

### 4.2 SLR Experiments — Python

The SLR experiments require:

- Python 3.x
- NumPy
- SciPy
- Matplotlib

The required packages can be installed using:

```bash
pip install numpy scipy matplotlib
```

---

## 💻 5. How to Run

### 5.1 QCQP Experiments

The QCQP experiments are implemented in MATLAB.

1. Download or clone this repository.
2. Extract `QCQP.rar`.
3. Open MATLAB.
4. Navigate to the extracted QCQP directory.
5. Add the directory and its subdirectories to the MATLAB path:

```matlab
addpath(genpath(pwd));
```

6. Run the corresponding MATLAB experiment scripts.

The parameter settings used in the experiments are provided in the
corresponding MATLAB source files.

---

### 5.2 SLR Experiments

The SLR experiments are implemented in Python.

1. Download or clone this repository.
2. Extract `SLR.rar`.
3. Open a terminal in the extracted SLR directory.
4. Install the required Python packages:

```bash
pip install numpy scipy matplotlib
```

5. Run the corresponding Python experiment scripts.

For example:

```bash
python <experiment_script>.py
```

Replace `<experiment_script>.py` with the corresponding experiment
script provided in the SLR directory.

---

## 📊 6. Numerical Experiments

The repository contains two groups of numerical experiments
corresponding to the experiments reported in the paper.

### 6.1 Quadratically Constrained Quadratic Programming (QCQP)

The QCQP experiments evaluate the performance of the proposed
primal-dual algorithms on constrained convex optimization problems.

For the general convex-concave setting, **NPDAL-n** is compared with
**PDAc-L** and **PDB**.

For the strongly convex setting, the accelerated algorithm **aNPDAL-n**
is further compared with **NPDAL-n**, **PDAc-L**, **aPDAc-L**, and
**aPDB**.

The numerical experiments report performance measures including:

- number of iterations;
- CPU time;
- number of linesearch steps;
- objective function error;
- constraint violation.

The QCQP experiments are implemented in MATLAB.

---

### 6.2 Sparse Logistic Regression (SLR)

The SLR experiments investigate the practical performance of the
proximal gradient method obtained from the proposed framework.

The proposed **NPGM** is compared with **aGRAAL** and **aPGMc** on
high-dimensional sparse logistic regression problems.

The experiments use the following LIBSVM datasets:

- **a9a**
- **rcv1**

The numerical results are evaluated primarily in terms of the
objective-function gap with respect to CPU time.

The SLR experiments are implemented in Python.

---

## 🔧 7. Parameter Settings

The algorithmic parameters used in the numerical experiments are
pre-configured in the corresponding MATLAB and Python source files.

For NPDAL-n, the parameters include

$$
a,\quad b,\quad \psi,\quad \phi,\quad
\xi,\quad \xi_1,\quad \mu,\quad \nu,\quad M,\quad \eta.
$$

The default parameter settings in the source codes correspond to the
numerical experiments reported in the paper.

For reproducibility, users are encouraged to use the default parameter
settings provided in the repository.

The complete parameter settings of the proposed and baseline algorithms
are specified in the corresponding experiment files.

---

## 🔁 8. Reproducibility

The codes in this repository are provided to facilitate the reproduction
of the numerical results reported in the paper.

To reproduce the experiments:

1. Download or clone this repository.
2. Extract `QCQP.rar` and/or `SLR.rar`.
3. For QCQP experiments, use MATLAB R2023b or a compatible version.
4. Install and configure CVX and MOSEK if required.
5. For SLR experiments, use Python 3.x with the required packages.
6. Run the corresponding experiment scripts.
7. Compare the generated numerical results with the figures and tables
   reported in the manuscript.

The default algorithmic parameters in the provided source codes are the
settings used for the reported numerical experiments.

---

## 📑 9. Paper and Code Correspondence

The relationship between the numerical experiments and the source code
is summarized below:

| Numerical Experiment | Problem | Programming Language |
|-----------------------|---------|----------------------|
| QCQP | Quadratically Constrained Quadratic Programming | MATLAB |
| SLR | Sparse Logistic Regression | Python |

### QCQP

The QCQP MATLAB codes implement the proposed and baseline primal-dual
algorithms used in the corresponding numerical experiments.

### SLR

The SLR Python codes implement NPGM and the corresponding baseline
methods used for comparison on the `a9a` and `rcv1` datasets.

---

## 📝 10. Citation

If you find this repository, the code, or the proposed algorithms useful
in your research, please consider citing the following paper:

```bibtex
@article{Liu2026NPDALn,
  title   = {A New Primal-Dual Algorithm with Two Convex Combinations and Linesearch for General Convex-Concave Saddle-Point Problems},
  author  = {Liu, Zexian and Liu, Shuning and Li, Jialong},
  year    = {2026}
}
```

The bibliographic information will be updated with the final publication
details once the paper is published.

---

## 🎓 11. Acknowledgements

This research is supported by the National Natural Science Foundation
of China (Nos. 12261019 and 12571329).

---

## 📜 12. License

This project is licensed under the MIT License.

See the [LICENSE](LICENSE) file for details.
