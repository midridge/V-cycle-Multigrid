import numpy as np
import matplotlib.pyplot as plt


N = 64
L = 1
dx = L / N

phi = np.zeros(shape=(N + 1,))
new = np.zeros(shape=(N + 1,))
tmp = np.array([np.sin(np.pi * i * dx) / 2 + np.sin(16 * np.pi * i * dx) / 2 for i in range(1, N)])
r0 = np.zeros(shape=(N + 1,))
r10 = np.zeros(shape=(N + 1,))
r100 = np.zeros(shape=(N + 1,))
r0[1:N] = tmp

resi = [0]
resi[0] = np.max(np.abs(tmp))

for t in range(0, 10000):
    for j in range(1, N):
        new[j] = (phi[j + 1] + new[j - 1] - dx ** 2 * tmp[j - 1]) / 2
    new[0] = 0
    new[N] = 0
    
    r = tmp - (new[0:N - 1] - 2 * new[1:N] + new[2:N + 1]) / dx ** 2
    resi.append(np.max(np.abs(r)))
    phi = new

    if t == 10:
        r10[1:N] = r
    elif t == 100:
        r100[1:N] = r
    
    if(resi[-1] < 0.001):
        print("converge at {} iterations".format(t))
        break

plt.figure()
plt.plot(range(len(resi)), resi, '+-')
plt.xlabel("Number of iterations")
plt.ylabel("max(|r_j|)")
plt.title("Convergence Curve")

plt.figure()
x = np.linspace(0, 1, N + 1)
plt.plot(x, r0, '-', x, r10, '+-', x, r100, 'x-')
plt.legend(['0 iterations','10 iterations','100 iterations'])
plt.xlabel('x_j')
plt.ylabel('r_j')
plt.title('r_j against x_j')
plt.show()