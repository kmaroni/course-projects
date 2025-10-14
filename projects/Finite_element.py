import numpy as np
import sympy as sp
from sympy import Mul
import matplotlib.pyplot as plt
from scipy.integrate import quad

x, h = sp.symbols('x, h')

def Lagrangebasis(xj, x=x, sympy=True):
    """Construct Lagrange basis function for points in xj

    Parameters
    ----------
    xj : array
        Interpolation points
    x : Sympy Symbol

    Returns
    -------
    Lagrange basis functions
    """
    from sympy import Mul
    n = len(xj)
    ell = []
    numert = Mul(*[x - xj[i] for i in range(n)])

    for i in range(n):
        numer = numert/(x - xj[i])
        denom = Mul(*[(xj[i] - xj[j]) for j in range(n) if i != j])
        ell.append(numer/denom)
    if sympy:
        return ell
    return [np.polynomial.Polynomial(l.as_poly().all_coeffs()[::-1]) for l in ell]

def get_element_boundaries(xj, e, d=1):
    return xj[d*e], xj[d*(e+1)]

def get_element_length(xj, e, d=1):
    xL, xR = get_element_boundaries(xj, e, d=d)
    return xR-xL

def local_to_global_map(e, r=None, d=1): # q(e, r)
    if r is None:
        return slice(d*e, d*e+d+1)
    return d*e+r

def map_true_domain(xj, e, d=1, x=x): # return x(X)
    xL, xR = get_element_boundaries(xj, e, d=d)
    hj = get_element_length(xj, e, d=d)
    return (xL+xR)/2+hj*x/2

def map_reference_domain(xj, e, d=1, x=x): # return X(x)
    xL, xR = get_element_boundaries(xj, e, d=d)
    hj = get_element_length(xj, e, d=d)
    return (2*x-(xL+xR))/hj

def map_u_true_domain(u, xj, e, d=1, x=x): # return u(x(X))
    return u.subs(x, map_true_domain(xj, e, d=d, x=x))

def assemble_b(u, xj, d=1):
    l = Lagrangebasis(np.linspace(-1, 1, d+1), sympy=False)
    N = len(xj)-1
    Ne = N//d
    b = np.zeros(N+1)
    for elem in range(Ne):
        hj = get_element_length(xj, elem, d=d)
        us = sp.lambdify(x, map_u_true_domain(u, xj, elem, d=d))
        integ = lambda xj, r: us(xj)*l[r](xj)
        for r in range(d+1):
            b[d*elem+r] += hj/2*quad(integ, -1, 1, args=(r,))[0]
    return b

def fe_evaluate_v(uh, pv, xj, d=1):
    l = Lagrangebasis(np.linspace(-1, 1, d+1), sympy=False)
    elem = (np.argmax((pv <= xj[::d, None]), axis=0)-1).clip(min=0) # find elements containing pv
    xL = xj[:-1:d] # All left element boundaries
    xR = xj[d::d]  # All right element boundaries
    xm = (xL+xR)/2 # middle of all elements
    hj = (xR-xL)   # length of all elements
    Xx = 2*(pv-xm[elem])/hj[elem] # map pv to reference space all elements
    dofs = np.array([uh[e*d+np.arange(d+1)] for e in elem], dtype=float)
    V = np.array([lr(Xx) for lr in l], dtype=float) # All basis functions evaluated for all points
    return np.sum(dofs * V.T, axis=1)

def L2_error(uh, ue, xj, d=1):
    yj = np.linspace(-1, 1, 4*len(xj))
    uhj = fe_evaluate_v(uh, yj, xj, d=d)
    uej = ue(yj)
    return np.sqrt(np.trapz((uhj-uej)**2, dx=yj[1]-yj[0]))

def Lagrange_points(d=1):
    """make array of d+1 uniformly spaced rational points in [-1,1]
    """
    xj = np.zeros(d+1)
    p = (d+1)//2
    end = np.arange(1,p+1)
    xj[-p:] = end
    xj[:p] = -end[::-1]
    return xj/p #make points contained in [-1,1]

def Ade(d=1):
    points = Lagrange_points(d)
    ld = Lagrangebasis(points)
    ade = lambda r, s: sp.integrate(ld[r]*ld[s], (x, -1, 1))
    Ade = np.zeros((d+1,d+1))
    for r in range(d+1):
        for s in range(r,d+1): #compue only above diagonal
            Ade[r,s] = ade(r,s)
    Ade +=  Ade.T - np.diag(Ade.diagonal()) #Make Ade symmetric
    return (h/2)*sp.Matrix(Ade)

def assemble_mass(xj, d=1):
    N = len(xj)-1
    Ne = N//d
    A = np.zeros((N+1, N+1))
    Ad = Ade(d)
    for elem in range(Ne):
        hj = get_element_length(xj, elem, d=d)
        s0 = local_to_global_map(elem, d=d)
        A[s0, s0] += np.array(Ad.subs(h, hj), dtype=float)
    return A

def assemble(u, N, domain=(-1, 1), d=1, xj=None):
    if xj is not None:
        mesh = xj
    else:
        mesh = np.linspace(domain[0], domain[1], N+1)

    A = assemble_mass(mesh, d=d)
    b = assemble_b(u, mesh, d=d)
    return A, b, mesh

if __name__ == '__main__':
    
    u = sp.exp(sp.cos(x))
    ue = sp.lambdify(x, u)
    err = []
    err2 = []
    err4 = []
    N_points = np.array([8,24,40,56])
    for N in N_points:
        xj = np.linspace(-1, 1, N+1)
        A, b, xj = assemble(u, N, (-1, 1), 1)
        uh = np.linalg.inv(A) @ b
        A2, b2, xj = assemble(u, N, (-1, 1), 2)
        uh2 = np.linalg.inv(A2) @ b2
        A4, b4, xj = assemble(u, N, (-1, 1), 4)
        uh4 = np.linalg.inv(A4) @ b4
        err.append(L2_error(uh, ue, xj, 1))
        err2.append(L2_error(uh2, ue, xj, 2))
        err4.append(L2_error(uh4, ue, xj, 4))

    plt.figure(figsize=(5, 3))
    plt.loglog(N_points, err, 'g',
               N_points, N_points.astype(float)**(-2), 'g--',
               N_points, err2, 'y',
               N_points, N_points.astype(float)**(-3), 'y--',
               N_points, err4, 'r',
               N_points, N_points.astype(float)**(-5), 'r--',)

    plt.legend(['FEM d=1', 'slope=2', 'FEM d=2', 'slope=3', 'FEM d=4', 'slope=5'])
    plt.show()

    funcs = {#function, interval=(a,b), N
    1:(abs(x),(-1,1),4),
    2:(sp.exp(sp.sin(x)),(0,2),8),
    3:(x**10, (0,1),12),
    4:(sp.exp(-(x-0.5)**2)-sp.exp(-0.25), (0,1),12),
    5:(sp.besselj(0,x),(0,100),56)
    }
    d = 4

    for n in funcs:
        u, domain, N = funcs[n]
        ue = sp.lambdify(x,u)
        A, b, xj = assemble(u, N, domain=domain, d=d)
        uh = np.linalg.inv(A) @ b
        yj = np.linspace(domain[0], domain[1], 200)
        plt.figure(figsize=(5, 3))
        plt.plot(xj, uh, 'b-o', yj, ue(yj), 'r--')
        plt.legend(['FEM', 'Exact'])
        plt.show()
