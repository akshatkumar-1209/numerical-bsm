# Part 3 — Tridiagonal Systems and the Thomas Algorithm

> **Program:** [`programs/thomas_solver.py`](../programs/thomas_solver.py)
> **Output:** [`results/04_thomas_solver.txt`](../results/04_thomas_solver.txt)

---

## 3.1 The problem

Every implicit time step of Parts 4, 5 and 8 requires the solution of a linear system

$$
A\,\mathbf{x} = \mathbf{d}, \qquad A \in \mathbb{R}^{m\times m},\ m = M-1, \qquad\qquad (3.1)
$$

in which row $i$ of $A$ contains at most three non-zero entries. Gaussian elimination on a dense $m\times m$ matrix costs $O(m^3)$ operations and $O(m^2)$ storage. The question of this part is:

> *How can the special structure of (3.1) be exploited to solve it in $O(m)$ operations, and under what conditions is that procedure safe?*

---

## 3.2 Why the matrix is tridiagonal

The discrete operator (2.8) couples node $j$ only to $j-1$ and $j+1$. For the θ-scheme (2.13) the unknowns at level $n+1$ satisfy, for $j=1,\dots,M-1$,

$$
\underbrace{-\theta\Delta\tau\,\alpha_j}_{a_j} V_{j-1}^{n+1} + \underbrace{\left(1 - \theta\Delta\tau\,\beta_j\right)}_{b_j} V_j^{n+1} + \underbrace{-\theta\Delta\tau\,\gamma_j}_{c_j} V_{j+1}^{n+1} = d_j, \qquad\qquad (3.2)
$$

where $d_j$ contains only known quantities (the old level and boundary values). Row $j$ has non-zeros only in columns $j-1, j, j+1$; the boundary values $V_0^{n+1}$ and $V_M^{n+1}$ are known and move to $d_1$ and $d_{M-1}$. Hence

$$
A = \begin{pmatrix}
b_1 & c_1 & & & \\
a_2 & b_2 & c_2 & & \\
 & \ddots & \ddots & \ddots & \\
 & & a_{m-1} & b_{m-1} & c_{m-1} \\
 & & & a_m & b_m
\end{pmatrix}, \qquad m = M-1. \qquad\qquad (3.3)
$$

---

## 3.3 Derivation of the algorithm

Write the system row by row (index $i = 1,\dots,m$, with $a_1 = c_m = 0$):

$$
a_i x_{i-1} + b_i x_i + c_i x_{i+1} = d_i. \qquad\qquad (3.4)
$$

**Forward elimination.** The aim is to turn (3.4) into the upper-bidiagonal form

$$
x_i + \tilde c_i\, x_{i+1} = \tilde d_i. \qquad\qquad (3.5)
$$

*Row 1.* Divide by $b_1$: $\ \tilde c_1 = c_1/b_1$, $\ \tilde d_1 = d_1/b_1$.

*Row $i \ge 2$.* Suppose row $i-1$ has already been reduced to $x_{i-1} = \tilde d_{i-1} - \tilde c_{i-1} x_i$. Substitute this into (3.4):

$$
a_i\left(\tilde d_{i-1} - \tilde c_{i-1} x_i\right) + b_i x_i + c_i x_{i+1} = d_i
\;\Longrightarrow\;
\underbrace{\left(b_i - a_i \tilde c_{i-1}\right)}_{q_i} x_i + c_i x_{i+1} = d_i - a_i \tilde d_{i-1}.
$$

Dividing by the pivot $q_i$ gives (3.5) with

$$
q_i = b_i - a_i\,\tilde c_{i-1}, \qquad \tilde c_i = \frac{c_i}{q_i}, \qquad \tilde d_i = \frac{d_i - a_i\,\tilde d_{i-1}}{q_i}, \qquad i = 2,\dots,m. \qquad\qquad (3.6)
$$

**Back substitution.** The last row has no $x_{m+1}$ ($c_m=0$), so $x_m = \tilde d_m$; then (3.5) is solved upwards:

$$
x_m = \tilde d_m, \qquad x_i = \tilde d_i - \tilde c_i\,x_{i+1}, \qquad i = m-1,\dots,1. \qquad\qquad (3.7)
$$

**Cost.** (3.6) needs about 6 operations per row and (3.7) about 2, i.e. roughly $8m$ operations and $O(m)$ storage, compared with $\tfrac23 m^3$ for dense elimination. For $M = 400$ this is ~3,200 operations instead of ~$4\times10^7$ per time step.

---

## 3.4 When is the algorithm safe? Diagonal dominance

The Thomas algorithm is Gaussian elimination **without pivoting**; it fails if a pivot $q_i$ vanishes and amplifies round-off if $\lvert\tilde c_i\rvert$ grows. A sufficient condition for safety is **strict diagonal dominance**:

$$
\lvert b_i\rvert > \lvert a_i\rvert + \lvert c_i\rvert, \qquad i = 1,\dots,m. \qquad\qquad (3.8)
$$

*Claim:* under (3.8), $\lvert\tilde c_i\rvert < 1$ and $q_i \neq 0$ for all $i$.
*Proof by induction:* $\lvert\tilde c_1\rvert = \lvert c_1\rvert/\lvert b_1\rvert < 1$. If $\lvert\tilde c_{i-1}\rvert < 1$, then $\lvert q_i\rvert \ge \lvert b_i\rvert - \lvert a_i\rvert\lvert\tilde c_{i-1}\rvert > \lvert b_i\rvert - \lvert a_i\rvert > \lvert c_i\rvert \ge 0$, so $q_i \ne 0$ and $\lvert\tilde c_i\rvert = \lvert c_i\rvert/\lvert q_i\rvert < 1$. $\square$

Consequently every error in $x_{i+1}$ is multiplied by a factor smaller than one in (3.7): back substitution is stable.

**The Black–Scholes matrices satisfy (3.8).** With $a_j = -\theta\Delta\tau\alpha_j$, $b_j = 1 + \theta\Delta\tau(\sigma^2 j^2 + r)$, $c_j = -\theta\Delta\tau\gamma_j$ and (2.9):

* If $\alpha_j \ge 0$ (all $j \ge r/\sigma^2$): $\lvert a_j\rvert + \lvert c_j\rvert = \theta\Delta\tau(\alpha_j + \gamma_j) = \theta\Delta\tau\sigma^2 j^2$, so
  $b_j - (\lvert a_j\rvert + \lvert c_j\rvert) = 1 + \theta\Delta\tau\, r > 0.$
* If $\alpha_j < 0$ (only $j=1$ at the benchmark): $\lvert a_j\rvert + \lvert c_j\rvert = \theta\Delta\tau(\gamma_j - \alpha_j) = \theta\Delta\tau\, r j$, so
  $b_j - (\lvert a_j\rvert + \lvert c_j\rvert) = 1 + \theta\Delta\tau(\sigma^2 j^2 + r - rj) > 0$ for $j=1$.

Hence, for the benchmark parameters, the fully implicit ($\theta=1$), Crank–Nicolson ($\theta=\tfrac12$) and half-step start-up matrices are strictly diagonally dominant for **every** $\Delta S$ and $\Delta\tau$, and the Thomas algorithm cannot break down.

*Remark.* The second case generalises to $\sigma^2 j^2 + r(1-j) > 0$. If a user chooses parameters with $r/\sigma^2 > 2$, several nodes near $S=0$ have $\alpha_j<0$ and this quantity can become negative for some $j$; dominance then holds only for small enough $\Delta\tau$. The code checks every pivot and stops with an error message instead of returning a wrong answer.

---

## 3.5 Algorithm

```
Input  : a (lower), b (main), c (upper), d (rhs), length m     [a[0], c[m-1] unused]
c~[0] = c[0]/b[0];  d~[0] = d[0]/b[0]
for i = 1 .. m-1:
    q     = b[i] - a[i]*c~[i-1]
    c~[i] = c[i]/q            (i < m-1)
    d~[i] = (d[i] - a[i]*d~[i-1])/q
x[m-1] = d~[m-1]
for i = m-2 .. 0:
    x[i] = d~[i] - c~[i]*x[i+1]
```

(Zero-based indices as in the code; the mathematics above uses $i=1,\dots,m$.)

---

## 3.6 Implementation and verification

`thomas_solve(a, b, c, d)` in `thomas_solver.py` implements (3.6)–(3.7) line by line and raises an error on a zero pivot. All solvers call this one function; no library solver is used inside the pricing code.

`thomas_solver.py` verifies it independently of any option pricing (Plan §6.4, §6.17):

| Test | $\max\lvert x - x_{\text{ref}}\rvert$ | $\max\lvert A x - d\rvert$ |
|---|---:|---:|
| 6×6 system with known solution $x = (1,\dots,6)$ | 8.9e-16 | 8.9e-16 |
| random diagonally dominant 200×200 vs `numpy.linalg.solve` | 1.1e-16 | 3.3e-16 |
| Black–Scholes fully implicit matrix, $M = 400$ vs NumPy | 1.7e-13 | 9.1e-13 |
| Black–Scholes Crank–Nicolson matrix, $M = 400$ vs NumPy | 1.7e-13 | 9.1e-13 |

The program also confirms strict diagonal dominance of both Black–Scholes matrices. In addition, `implicit_fdm.py` and `crank_nicolson.py` solve their first time step with both Thomas and `numpy.linalg.solve` and report a difference of about $3\times10^{-13}$, i.e. round-off level relative to option values of order $10^2$.
