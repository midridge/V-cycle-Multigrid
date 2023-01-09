import numpy as np
import matplotlib.pyplot as plt


def smoothing(phi, f, h):
    N = np.size(phi) - 1
    res = np.zeros(shape=(N + 1,))
    for j in range(1, N):
        res[j] = (phi[j + 1] + res[j - 1] - h ** 2 * f[j]) / 2
    return res


def residual(phi, f, h):
    N = np.size(phi) - 1
    res = np.zeros(shape=(N + 1,))
    res[1:N] = f[1:N] - (phi[0:N - 1] - 2 * phi[1:N] + phi[2:N + 1]) / h ** 2
    return res


def restriction(r):
    N = int((np.size(r) - 1) / 2)
    res = np.zeros(shape=(N + 1,))
    for j in range(2, N + 1):
        res[j - 1] = (r[2 * j - 3] + 2 * r[2 * j - 2] + r[2 * j - 1]) / 4
    return res


def prolongation(eps):
    N = (np.size(eps) - 1) * 2
    res = np.zeros(shape=(N + 1,))
    for j in range(2, N + 1, 2):
        res[j - 1] = (eps[int(j / 2 - 1)] + eps[int(j / 2)]) / 2
    for j in range(1, N + 2, 2):
        res[j - 1] = eps[int((j + 1) / 2 - 1)]
    return res

def V_Cycle(phi, f, h):
    phi = smoothing(phi, f, h)
    r = residual(phi, f, h)
    rhs = restriction(r)
    eps = np.zeros(np.size(rhs))

    if np.size(eps) - 1 == 2:
        eps = smoothing(eps, rhs, 2 * h)
    else:
        eps = V_Cycle(eps, rhs, 2 * h)
    
    phi = phi + prolongation(eps)
    phi = smoothing(phi, f, h)

    return phi


N = 64
L = 1
h = L / N

phi = np.zeros(shape=(N + 1,))
new = np.zeros(shape=(N + 1,))
f = np.array([np.sin(np.pi * i * h) / 2 + np.sin(16 * np.pi * i * h) / 2 for i in range(0, N + 1)])

resi = []
for cnt in range(0, 1000):
    phi = V_Cycle(phi, f, h)
    r = residual(phi, f, h)

    resi.append(np.max(np.abs(r)))

    if (resi[-1] < 0.001):
        break

plt.figure()
plt.plot(np.arange(len(resi))*10,resi,'+-')
plt.xlabel('Number of Iterations')
plt.ylabel('max(|r_j|)')
plt.title('Convergence Curve')
plt.show()
