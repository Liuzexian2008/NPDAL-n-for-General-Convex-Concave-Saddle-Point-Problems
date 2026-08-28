# NPDAL-n: A New Primal-Dual Algorithm for General Convex-Concave Saddle-Point Problems

![MATLAB](https://img.shields.io/badge/MATLAB-R2023b-blue.svg)
![Python](https://img.shields.io/badge/Python-3.x-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

This repository contains the official MATLAB and Python implementations for the numerical experiments in the following paper:

> **A New Primal-Dual Algorithm with Two Convex Combinations and Linesearch for General Convex-Concave Saddle-Point Problems**  
> Zexian Liu, Shuning Liu, Jialong Li

The code is provided to reproduce the numerical results reported in the manuscript and to facilitate further research on primal-dual algorithms for general convex-concave saddle-point problems.

---

## 1. Problem Formulation

We consider the following general convex-concave saddle-point problem:
```math
$$
\min_{x\in\mathbb{R}^q}\max_{y\in\mathbb{R}^p}
\mathcal{L}(x,y)
=
g(x)+\Phi(x,y)-f^*(y).
$$
```
where $g:\mathbb{R}^q\rightarrow(-\infty,+\infty]$ and
$f^*:\mathbb{R}^p\rightarrow(-\infty,+\infty]$ are proper, closed,
and convex extended real-valued functions.

The coupling function
```math
$$
\Phi:dom(g)\times dom(f^*)
\rightarrow\mathbb{R}
$$
```
is continuously differentiable, convex with respect to $x$ for each
fixed $y$, and concave with respect to $y$ for each fixed $x$.

The proposed **NPDAL-n** algorithm is designed to solve this class of
general convex-concave saddle-point problems by incorporating two convex
combinations and an adaptive linesearch strategy.

For strongly convex problems, an accelerated variant, **aNPDAL-n**, is
also developed.

---

## 2. Algorithms

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

## 3. Numerical Experiments

The repository contains two groups of numerical experiments
corresponding to the experiments reported in the paper.

### 3.1 Quadratically Constrained Quadratic Programming (QCQP)

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

### 3.2 Sparse Logistic Regression (SLR)

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

## 4. Paper and Code Correspondence

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
