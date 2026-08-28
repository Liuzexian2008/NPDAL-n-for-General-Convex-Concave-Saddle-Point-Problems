import numpy as np
import scipy.linalg as LA
import scipy.sparse as spr
import scipy.sparse.linalg as spr_LA
from time import perf_counter
from sklearn import datasets
import matplotlib.pyplot as plt
import sys
sys.path.append(r'D:\Desktop\Code\13 sparse_logistic_regression')
from scipy import io

# ==========================================
# 1. Data loading and preprocessing
# ==========================================
# Make sure your project root directory has the 'data' folder with the corresponding dataset
filename = "data/a9a"
# filename = "data/rcv1_train.binary.bz2"

A, b = datasets.load_svmlight_file(filename)
m, n = A.shape

print("The dataset {}. The dimensions: m={}, n={}".format(filename[5:], m, n))

# Definitions for sparse logistic regression
gamma = 0.005 * LA.norm(A.T.dot(b), np.inf)
K = (A.T.multiply(-b)).T.tocsr()

# Find the norm of K^T K
L = spr_LA.svds(K, k=1, return_singular_vectors=False) ** 2

# Starting point
x0 = np.zeros(n)

# Stepsize
ss = 4 / L

g = lambda x: gamma * LA.norm(x, 1)
prox_g = lambda x, rho: x + np.clip(-x, -rho * gamma, rho * gamma)

f = lambda x: np.log(1. + np.exp(x)).sum()

def df(x):
    exp_x = np.exp(x)
    return exp_x / (1. + exp_x)

dh = lambda x, Kx: K.T.dot(df(Kx))

# Residual
res = lambda x: LA.norm(x - prox_g(x - dh(x, K.dot(x)), 1))

# Energy
J = lambda x, Kx: f(Kx) + g(x)

def prox_T(z, v, x, k):
    l = k ** 2
    z1 = z - np.max(np.dot(v, z - x) / l, 0) * v
    return z1


# ==========================================
# 2. Algorithm implementations
# ==========================================

def explicit_graal(x1, phi, numb_iter=100):
    begin = perf_counter()
    phi = 1.5
    x, x_ = x1.copy(), x1.copy()
    x0 = x + np.random.randn(x.shape[0]) * 1e-9
    Kx = K.dot(x)
    dhx = dh(x, Kx)
    la = phi / 2 * LA.norm(x - x0) / LA.norm(dhx - dh(x0, K.dot(x0)))
    rho = 1. / phi + 1. / phi ** 2
    values = [J(x, Kx)]
    step_list = [la]
    tt = [0]
    th = 1

    for i in range(numb_iter):
        x1 = prox_g(x_ - la * dhx, la)
        Kx1 = K.dot(x1)
        dhx1 = dh(x1, Kx1)

        n1 = LA.norm(x1 - x) ** 2
        n2 = LA.norm(dhx1 - dhx) ** 2

        n1_div_n2 = n1 / n2 if n2 != 0 else la * 10
        la1 = min(rho * la, 0.25 * phi * th / la * (n1_div_n2))
        x_ = ((phi - 1) * x1 + x_) / phi
        th = phi * la1 / la
        x, la, dhx = x1, la1, dhx1
        values.append(J(x1, Kx1))
        step_list.append(la1)
        tt.append(perf_counter() - begin)

    end = perf_counter()
    print("Time execution of EGRAAL:", end - begin)
    return values, x, step_list, tt


def explicit_pgm(x1, phi, rho, numb_iter=100):
    begin = perf_counter()
    x, x_ = x1.copy(), x1.copy()
    x0 = x + np.random.randn(x.shape[0]) * 1e-9
    Kx = K.dot(x)
    dhx = dh(x, Kx)
    la = phi / 2 * LA.norm(x - x0) / LA.norm(dhx - dh(x0, K.dot(x0)))
    xi = phi - phi ** 3 * rho / (2 * (1 + phi))
    print('aPGMc xi=', xi)
    values = [J(x, Kx)]
    step_list = [la]
    tt = [0]
    th = 1

    for i in range(numb_iter):
        x1 = prox_g(x_ - la * dhx, la)
        Kx1 = K.dot(x1)
        dhx1 = dh(x1, Kx1)

        n1 = LA.norm(x1 - x) ** 2
        n2 = LA.norm(dhx1 - dhx) ** 2

        n1_div_n2 = n1 / n2 if n2 != 0 else la * 10
        la1 = min(rho * la, xi ** 2 * th / la * (n1_div_n2))

        x_ = ((phi - 1) * x1 + x_) / phi
        th = la1 / la
        x, la, dhx = x1, la1, dhx1
        values.append(J(x1, Kx1))
        step_list.append(la1)
        tt.append(perf_counter() - begin)

    end = perf_counter()
    print(f"Time execution of aPGMc (rho={rho}):", end - begin)
    return values, x, step_list, tt


def explicit_Npgm(x1, phi, rho, numb_iter=100):
    begin = perf_counter()
    x, x_ = x1.copy(), x1.copy()
    x0 = x + np.random.randn(x.shape[0]) * 1e-9
    Kx = K.dot(x)
    dhx = dh(x, Kx)
    la = 10 * phi / 2 * LA.norm(x - x0) / LA.norm(dhx - dh(x0, K.dot(x0)))
    xi = phi - phi ** 3 * rho / (2 * (1 + phi))
    w = xi
    print('NPGM w=', w)
    values = [J(x, Kx)]
    step_list = [la]
    tt = [0]
    th = 1
    xi1 = 1 / rho
    print('NPGM xi1=', xi1)
    print('NPGM a_max=', w * xi1 / (w * xi1 + 1))
    a = 0.1 * w * xi1 / (w * xi1 + 1)
    # a = 1 / phi
    w1 = (1 - a) * w - a / xi1
    print('NPGM a=', a)
    print('NPGM w1=', w1)

    for i in range(numb_iter):
        x_ = ((phi - 1) * x + x_) / phi
        xmd = a * x + (1 - a) * x_
        x1 = prox_g(xmd - la * dhx, la)
        Kx1 = K.dot(x1)
        dhx1 = dh(x1, Kx1)

        n1 = LA.norm(x1 - x) ** 2
        n2 = LA.norm(dhx1 - dhx) ** 2
        n1_div_n2 = n1 / n2 if n2 != 0 else la * 10
        la1 = min(rho * la, (1 - a) * xi * w1 * th / la * (n1_div_n2))
        th = la1 / la
        x, la, dhx = x1, la1, dhx1
        values.append(J(x1, Kx1))
        step_list.append(la1)
        tt.append(perf_counter() - begin)

    end = perf_counter()
    print(f"Time execution of NPGM (rho={rho}):", end - begin)
    return values, x, step_list, tt


# ==========================================
# 3. Run algorithms and collect results
# ==========================================
N = 10000  # You may reduce this for faster testing if the dataset is large

print("Running explicit_graal...")
ans1 = explicit_graal(x0, phi=1.5, numb_iter=N)

print("Running explicit_pgm (rho=10/9)...")
ans2 = explicit_pgm(x0, phi=2, rho=10 / 9, numb_iter=N)

print("Running explicit_pgm (rho=6/5)...")
ans3 = explicit_pgm(x0, phi=2, rho=6 / 5, numb_iter=N)

print("Running explicit_Npgm (rho=6/5)...")
ans4 = explicit_Npgm(x0, phi=2.7, rho=4 / 3, numb_iter=N) # phi corresponds to psi, rho corresponds to varphi in the algorithm
# ans4 = explicit_Npgm(x0, phi=1.5, rho=4 / 3, numb_iter=N)

# ==========================================
# 4. Visualization (comparison of 4 algorithms)
# ==========================================

labels = [
    "aGRAAL",
    r"aPGMc, $\psi=2$, $\varphi=10/9$",
    r"aPGMc, $\psi=2$, $\varphi=6/5$",
    r"NPGM, $\psi=2.7$, $\varphi=4/3$"
]
linestyles = ["--", "-", "-.", ":"]
colors = ['#FFD700', 'b', 'c', 'm'] # Gold, blue, cyan, magenta

# --- Figure 1: Iterations vs objective function value ---
values = [ans1[0], ans2[0], ans3[0], ans4[0]]
v_min = min([min(v) for v in values])

plt.figure(figsize=(6, 4))
for i, v in enumerate(values):
    plt.plot(v - v_min, color=colors[i], label=labels[i], linestyle=linestyles[i])
plt.yscale('log')
plt.xlabel('Iterations')
plt.ylabel('$F(x^n)-F_{opt}$')
plt.legend()
plt.tight_layout()
plt.show()

# --- Figure 2: Time vs objective function value ---
tt = [ans1[-1], ans2[-1], ans3[-1], ans4[-1]]

plt.figure(figsize=(6, 4))
for i, v in enumerate(values):
    plt.plot(tt[i], v - v_min, color=colors[i], label=labels[i], linestyle=linestyles[i])
plt.yscale('log')
plt.xlabel('Time, seconds')
plt.ylabel('$F(x_n)-F_{opt}$')
plt.legend()
plt.tight_layout()
plt.show()

# --- Figure 3: Iterations vs stepsize ---
plt.figure(figsize=(6, 4))
plt.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')

plt.plot(ans1[-2], '.', color=colors[0], label=labels[0], markersize=3)
plt.plot(ans2[-2], '.', color=colors[1], label=labels[1], markersize=3)
plt.plot(ans3[-2], '.', color=colors[2], label=labels[2], markersize=3)
plt.plot(ans4[-2], '.', color=colors[3], label=labels[3], markersize=3)

plt.xlim(0, N)
plt.xlabel('Iterations')
plt.ylabel(r'Stepsize $\tau_n$')
plt.yscale('log')
# Depending on the dataset, you may need to adjust axis limits:
plt.axis([-10, N + 10, 5e-4, 1e+1])
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()