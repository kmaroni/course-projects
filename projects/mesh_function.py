import numpy as np
from collections.abc import Callable
import matplotlib.pyplot as plt

def mesh_function(f: Callable[[float], float], t: float) -> np.ndarray:
    Nt = len(t)
    f_n = np.zeros(Nt)
    for n in range(Nt):
        f_n[n] = f(t[n])
    return f_n

def func(t: float) -> float:
    if t >= 0 and t <= 3:
        return np.exp(-t)
    elif t > 3 and t <= 4:
        return np.exp(-3*t)

def test_mesh_function():
    t = np.array([1, 2, 3, 4])
    f = np.array([np.exp(-1), np.exp(-2), np.exp(-3), np.exp(-12)])
    fun = mesh_function(func, t)
    assert np.allclose(fun, f)

if __name__ == "__main__":
    test_mesh_function()

    dt = 0.1
    t = np.arange(0,4,dt)
    fun = mesh_function(func, t)

    plt.plot(t,fun,marker='s',linestyle='-',markersize='3')
    plt.xlabel('t')
    plt.ylabel('f(t)')
    plt.show()
