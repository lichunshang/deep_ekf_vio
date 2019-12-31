import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-4, 4)
y_tanh = np.tanh(x)
y_tanh_derivative = 1-np.tanh(x)**2

plt.plot(x, y_tanh, label=r"$y=\tanh(x)$", linewidth=2)
plt.plot(x, y_tanh_derivative, label=r"$dy/dx$", linewidth=2)
plt.xlabel("x")
plt.ylabel("y")
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.grid()
ax = plt.axes()
ax.set_ylim(-1.5, 1.5)
plt.legend(loc=2)


plt.show()