import numpy as np
import sympy as sp
from sympy import besselj
import scipy.sparse as sparse
from scipy.integrate import quad, dblquad
import matplotlib.pyplot as plt

x, y = sp.symbols('x, y')

class Galerkin:
    def __init__(self,domain=(-1,1)):
        self.a, self.b = domain

    def create_mesh(self,Nx=100):
        xj = np.linspace(self.a,self.b,Nx)
        Xj = -1+2/(self.b-self.a)*(xj-self.a)
        return xj, Xj

    def c(self,j):
        return 2 if j==0 else 1

    def T(self,k,x):
        return sp.cos(k * sp.acos(x))

    def innerw(self,v):
        us = sp.simplify(self.us.subs(x, sp.cos(x)), inverse=True)
        vs = sp.simplify(v.subs(x, sp.cos(x)), inverse=True)
        return sp.integrate(us*vs, (x, 0, sp.pi))

    def innerwn(self,v):
        us = sp.simplify(self.us.subs(x, sp.cos(x)), inverse=True)
        vs = sp.simplify(v.subs(x, sp.cos(x)), inverse=True)
        return quad(sp.lambdify(x, us*vs), 0, np.pi)[0]

    def uhat(self, j, Numeric=False):
        I = self.innerwn(self.T(j,x)) if Numeric else self.innerw(self.T(j,x))
        return 2/(self.c(j)*sp.pi)*I

    def __call__(self, u, N, Numeric=False):
        self.us = u.subs(x, self.a+(self.b-self.a)*(x+1)/2)
        uN = 0
        for j in range(N+1):
            uN += self.uhat(j,Numeric=Numeric)*self.T(j,x)

        return uN.subs(x,-1+2/(self.b-self.a)*(x-self.a))

class Galerkin2D(Galerkin):
    def __init__(self,domainx=(-1,1),domainy=(-1,1)):
        self.a, self.b = domainx
        self.c, self.d = domainy

    def uhat(self,i,j):
        us = sp.simplify(self.us.subs({x: sp.cos(x),y: sp.cos(y)}), inverse=True)
        Ti = sp.simplify(self.T(i,x).subs(x, sp.cos(x)), inverse=True)
        Tj = sp.simplify(self.T(j,y).subs(y, sp.cos(y)), inverse=True)
        I  = dblquad(sp.lambdify((x,y),us*Ti*Tj), 0, np.pi, 0, np.pi, epsabs=1e-12)[0]
        return I

    def __call__(self,u,N,return_coeff=False):
        self.us = u.subs({x: self.a+(self.b-self.a)*(x+1)/2,
                          y: self.c+(self.d-self.c)*(y+1)/2})
        uN = 0
        uij = np.zeros((N+1,N+1))
        c = np.ones(N+1)
        c[0] = 2
        A = sparse.diags([c/np.pi], [0], (N+1, N+1))
        for i in range(N+1):
            for j in range(N+1):
                uij[i,j] = self.uhat(i,j)
        uhat_ij = A @ uij @ A

        if return_coeff:
            return uhat_ij

        else:
            uN = 0
            for i in range(N+1):
                for j in range(N+1):
                    uN += uhat_ij[i,j]*self.T(i,x)*self.T(j,y)

            return uN.subs({x: -1+2/(self.b-self.a)*(x-self.a),
                            y: -1+2/(self.d-self.c)*(y-self.c)})

    def convergence(self,ue):
        error = []
        for N in range(4,12,2):
            uN = self(ue,N,return_coeff=True)
            xi = np.linspace(-1,1,N+1)
            V = np.cos(np.outer(xi,np.arange(N+1)))
            U = V @ uN @ V.T
            xij, yij = np.meshgrid(xi, xi, indexing='ij', sparse=True)
            ueij = sp.lambdify((x,y),ue)(xij,yij)
            h = xi[1]-xi[0]
            L2_error = np.sqrt(np.trapezoid(np.trapezoid((U-ueij)**2,dx=h), dx=h))
            error.append(L2_error)

        return error




if __name__ == '__main__':
    
    funcs = {#function, interval=(a,b), N
    1:(abs(x),(-1,1),[2,3,5]),
    2:(sp.exp(sp.sin(x)),(0,2),[2,3,5]),
    3:(x**10, (0,1),[2,5,10]),
    4:(sp.exp(-(x-0.5)**2)-sp.exp(-0.25), (0,1),[2,3,4]),
    5:(sp.besselj(0,x),(0,100),[10,20,50])
    }

    for n in funcs:
        Numeric = True
        u, domain, Nums = funcs[n]
        approx = Galerkin(domain)
        xj, Xj = approx.create_mesh(Nx=1000)

        fig, ax = plt.subplots(figsize=(10,5))
        ax.set_title(sp.sstr(u))
        ue = sp.lambdify(x,u)(xj)
        ax.plot(xj,ue,label='Exact')
        for N in Nums:
            uN = approx(u, N, Numeric=Numeric)
            uN = sp.lambdify(x,uN)(xj)
            ax.plot(xj,uN,':',label=f'u_{N}')
        plt.legend()
        plt.tight_layout()
        plt.show()


    ue = sp.exp(-(x**2+2*(y-0.5)**2))
    approx = Galerkin2D()

    fig, ax = plt.subplots(1,4,subplot_kw={"projection": "3d"})
    xj = np.linspace(-1,1,25)
    X, Y = np.meshgrid(xj,xj,indexing='ij')
    u = sp.lambdify((x,y),ue)(X,Y)
    ax[-1].plot_wireframe(X,Y,u)
    ax[-1].set_title('Exact')
    for n, N in enumerate([2,4,10]):
        uN = approx(ue, N)
        uN = sp.lambdify((x,y),uN)(X,Y)
        ax[n].plot_wireframe(X,Y,uN)
        ax[n].set_title(f'u_{N}')
    plt.show()

    error = approx.convergence(ue)
    fig, ax = plt.subplots()
    ax.semilogy(range(4,12,2),error)
    plt.show()
