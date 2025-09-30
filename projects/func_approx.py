import numpy as np
import numpy.polynomial.legendre as leg
import sympy as sp
from sympy import besselj
from scipy.integrate import quad
from scipy.interpolate import lagrange
import matplotlib.pyplot as plt
from tqdm import tqdm



x = sp.symbols('x')

class Galerkin:
    def __init__(self,u,domain=(-1,1)):
        self.u = u
        self.a, self.b = domain
        self.us = self.u.subs(x, self.a+(self.b-self.a)*(x+1)/2)

    def create_mesh(self,Nx=100):
        xj = np.linspace(self.a,self.b,Nx)
        Xj = -1+2/(self.b-self.a)*(xj-self.a)
        return xj, Xj

    def inner(self,u,v):
        return sp.integrate(u*v,(x,-1,1))

    def ninner(self,u,v):
        uv = sp.lambdify(x, u*v)
        return quad(uv,-1,1)[0]

    def uhat(self,j,Numeric=False):
        if Numeric:
            c = np.zeros(j+1)
            c[-1] = 1
            legj = leg.legval(x,c)
            return (2*j+1)*self.ninner(self.us,legj)/2

        else:
            return (2*j+1)*self.inner(self.us,sp.legendre(j,x))/2

    def __call__(self,N,Numeric=False):
        self.N = N
        uL = []
        uN = 0

        for i in range(N+1):
            ui = self.uhat(i,Numeric)
            #uL.append(ui)
            uN += ui*sp.legendre(i,x)

        return uN #,uL

    def plot(self,uN,ax):
        xj, Xj = self.create_mesh()
        u = sp.lambdify(x, self.u)(xj)
        uN = sp.lambdify(x, uN)(Xj)
        ax.plot(xj, uN,'--',label=f'u_{self.N}')
        ax.plot(xj, u,'b')
        ax.set_title(self.u)






class Collocation(Galerkin):

    def lin_mesh(self,N):
        self.xp = np.linspace(-1,1,N+1)
        return self.xp

    def cheb_mesh(self,N):
        self.xc = np.cos(np.arange(N+1)*np.pi/N)
        return self.xc

    def lagrange_basis(self, xj, x=x):
        ell = []
        numert = sp.Mul(*[x-xj[i] for i in range(self.N+1)])
        for i in range(self.N+1):
            numer = numert/(x - xj[i])
            denom = sp.Mul(*[(xj[i] - xj[j]) for j in range(self.N+1) if i != j])
            ell.append(numer/denom)
        return ell

    def lagrange_function(self,uj,basis):
        uN = 0
        for j, uj_ in enumerate(uj):
            uN += basis[j]*uj_
        return uN

    def __call__(self, N, method='l'):
        self.N = N
        self.method = method
        self.xj = self.cheb_mesh(N) if method=='c' else self.lin_mesh(N)
        self.uj = sp.lambdify(x,self.us)(self.xj)
        basis = self.lagrange_basis(self.xj)
        uN = self.lagrange_function(self.uj,basis)

        return uN

    def plot(self,uN,ax):
        yj, Yj = self.create_mesh()
        xj_ = self.a+(self.b-self.a)/2*(self.xj+1)
        u = sp.lambdify(x, self.u)(yj)
        uN = sp.lambdify(x, uN)(Yj)
        ax.plot(yj, uN,'--',label=f'u_{self.N}')
        ax.plot(yj, u,'b')
        ax.plot(xj_,self.uj, 'bo')
        ax.set_title(self.u)

if __name__ == '__main__':
    funcs = {#function, interval=(a,b), N
    1:(abs(x),(-1,1),[2,3,5]),
    2:(sp.exp(sp.sin(x)),(0,2),[2,3,5]),
    3:(x**10, (0,1),[2,5,10]),
    4:(sp.exp(-(x-0.5)**2)-sp.exp(-0.25), (0,1),[2,3,4]),
    5:(sp.besselj(0,x),(0,100),[10,20])
    }

    #Galerkin
    print('Start Galerkin method:')
    L = 5; str = ''; i = 0
    print('|',str+'□'*(L-i),f'[{i:2}/{L}]|')
    for n in funcs:
        Numeric = True if n==2 or n==5 else False
        u, domain, Nums = funcs[n]
        approx = Galerkin(u,domain)

        fig, ax = plt.subplots(figsize=(10,5))
        for N in Nums:
            uN = approx(N,Numeric=Numeric)
            approx.plot(uN,ax)

        #Progress bar:
        str += '■'; i += 1
        print('|',str+'□'*(L-i),f'[{i:2}/{L}]|')
        plt.legend()
        plt.tight_layout()
        plt.show()


    #Collocation
    print('Start collocation method:')
    L = 5; str = ''; i = 0
    print('|',str+'□'*(L-i),f'[{i:2}/{L}]|')
    for n in funcs:
        u, domain, Nums = funcs[n]
        approx = Collocation(u, domain)
        fig, ax = plt.subplots(1,2,figsize=(10,5),sharey=True)
        for N in Nums:
            uN = approx(N,method='l')
            uNc = approx(N,method='c')
            approx.plot(uN,ax[0])
            approx.plot(uNc,ax[1])

        #Progress bar:
        str += '■'; i += 1
        print('|',str+'□'*(L-i),f'[{i:2}/{L}]|')

        ax[0].set_title('Linear')
        ax[1].set_title('Chebyshev')
        plt.legend()
        plt.tight_layout()
        plt.show()
