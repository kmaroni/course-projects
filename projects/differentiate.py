import numpy as np


def differentiate(u: np.ndarray, dt: float) -> np.ndarray:
    Nt = len(u) - 1
    d = np.zeros(Nt + 1)
    d[0] = (u[1]-u[0])/dt
    d[-1] = (u[-1]-u[]-2])/dt
    for n in range(1,Nt):
        d[n] = (u[n+1]-u[n-1])/(2*dt)
    return d

def differentiate_vector(u: np.ndarray, dt: float) -> np.ndarray:
    raise NotImplementedError

def test_differentiate():
    t = np.linspace(0, 1, 10)
    dt = t[1] - t[0]
    u = t**2
    du1 = differentiate(u, dt)
    du2 = differentiate_vector(u, dt)
    assert np.allclose(du1, du2)

if __name__ == '__main__':
    test_differentiate()
