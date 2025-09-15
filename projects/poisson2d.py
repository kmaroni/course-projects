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
        D2x = (1./self.px.dx**2)*self.px.D2()
        D2y = (1./self.py.dx**2)*self.py.D2()
        return (sparse.kron(D2x, sparse.eye(self.py.N+1)) +
                sparse.kron(sparse.eye(self.px.N+1), D2y))

    def assemble(self, bc=0, f=None):
        """Return assembled coefficient matrix A and right hand side vector b"""
        self.xij, self.yij = self.create_mesh()
        F = sp.lambdify((x,y), f)(self.xij, self.xij)
        A = self.laplace()

        #Define homogeneous Dirichlet boundary condition
        B = np.ones((self.px.N,self.py.N), dtype=bool)
        B[1:-1,1:-1] = 0
        bnds = np.where(B.ravel() == 1)[0]
        A = A.tolil()
        for i in bnds:
            A[i] = 0
            A[i, i] = 1
        A = A.tocsr()
        b = F.ravel()
        b[bnds] = bc #equal to constant on boundary

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
        uj = sp.lambdify([x, y], ue)(self.px.x,self.py.x)
        return np.sqrt(self.px.dx*self.py.dx*np.sum((uj-u)**2))

    def __call__(self, bc=0, f=implemented_function('f', lambda x, y: 2)(x, y)):
        """Solve Poisson's equation with a given right hand side function

        Parameters
        ----------
        f : Sympy function
            The right hand side function f(x, y)

        Returns
        -------
        The solution as a Numpy array

        """
        A, b = self.assemble(bc,f=f)
        return sparse.linalg.spsolve(A, b.ravel()).reshape((self.px.N+1, self.py.N+1))

def test_poisson2d():
    assert False

if __name__ == '__main__':
    Lx, Ly, Nx, Ny = 1, 1, 100, 100
    sol = Poisson2D(Lx, Ly, Nx, Ny)
    ue = x*(1-x)*y*(1-y)
    u=sol(bc=0,f=ue.diff(x, 2) + ue.diff(y, 2))
    err = sol.l2_error(u,ue)
    print(err)
