import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from sympy.utilities.lambdify import implemented_function
from poisson import Poisson

x, y = sp.symbols('x,y')

class Poisson2D:
    r"""Solve Poisson's equation in 2D::

        \nabla^2 u(x, y) = f(x, y), x, y in [0, Lx] x [0, Ly]

    with homogeneous Dirichlet boundary conditions.
    """

    def __init__(self, Lx, Ly, Nx, Ny):
        self.px = Poisson(Lx, Nx) # we can reuse some of the code from the 1D case
        self.py = Poisson(Ly, Ny)

    def create_mesh(self):
        """Return a 2D Cartesian mesh
        """
        return np.meshgrid(self.px.x, self.py.x, indexing='ij')

    def laplace(self):
        """Return a vectorized Laplace operator"""
        D2x = self.px.D2()
        D2y = self.py.D2()
        return (sparse.kron(D2x, sparse.eye(self.py.N+1)) +
                sparse.kron(sparse.eye(self.px.N+1), D2y))

    def assemble(self, f=None):
        """Return assembled coefficient matrix A and right hand side vector b"""
        self.xij, self.yij = self.create_mesh()
        F = sp.lambdify((x,y), f)(self.xij, self.yij)
        A = self.laplace()

        #Define homogeneous Dirichlet boundary condition
        B = np.ones((self.px.N+1,self.py.N+1), dtype=bool)
        B[1:-1,1:-1] = 0
        bnds = np.where(B.ravel() == 1)[0]
        A = A.tolil()
        for i in bnds:
            A[i] = 0
            A[i, i] = 1
        A = A.tocsr()
        b = F.ravel()
        b[bnds] = 0

        return A, b

    def l2_error(self, u, ue):
        """Return l2-error

        Parameters
        ----------
        u : array
            The numerical solution (mesh function)
        ue : Sympy function
            The analytical solution
        """
        uj = sp.lambdify([x, y], ue)(self.xij,self.yij)
        return np.sqrt(self.px.dx*self.py.dx*np.sum((u-uj)**2))

    def __call__(self, f=implemented_function('f', lambda x, y: 2)(x, y)):
        """Solve Poisson's equation with a given right hand side function

        Parameters
        ----------
        f : Sympy function
            The right hand side function f(x, y)

        Returns
        -------
        The solution as a Numpy array

        """
        A, b = self.assemble(f=f)
        return sparse.linalg.spsolve(A, b.ravel()).reshape((self.px.N+1, self.py.N+1))

    def plot(self,u):
        fig, ax = plt.subplots()
        ax.contourf(self.xij, self.yij, u)
        plt.show()

def test_poisson2d():
    tol = 1e-4

    #Define tests as dicts with exact solution and parameters
    test1 = {
    'ue': x*(1-x)*y*(1-y)*sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y)),
    'parameters': (1, 1, 100, 200)
    }

    test2 = {
    'ue': sp.sin(2*sp.pi*x)*sp.sin(2*sp.pi*y),
    'parameters': (1, 1, 100, 200)
    }

    test2 = {
    'ue': sp.sin(2*sp.pi*x)*sp.sin(2*sp.pi*y),
    'parameters': (1, 1, 100, 200)
    }

    test3 = {
    'ue': x*(2-x)*sp.sin(2*sp.pi*y)+sp.sin(sp.pi*x)*y*sp.cos(y),
    'parameters': (2, 1, 300, 200)
    }

    testsList = [test1,test2,test3]
    for test in testsList:
        ue = test['ue']
        Lx, Ly, Nx, Ny = test['parameters']
        sol = Poisson2D(Lx, Ly, Nx, Ny)
        u=sol(f=ue.diff(x, 2) + ue.diff(y, 2))

        err = sol.l2_error(u,ue)
        try:
            assert err < tol
        except AssertionError:
            print(f'ue = {ue} not within tolerance!')
            print(f'Error is {err:.4e}')


if __name__ == '__main__':
    test_poisson2d()
