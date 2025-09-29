import numpy as np
import sympy as sp
from sympy import besselj
import matplotlib.pyplot as plt


x = sp.symbols('x')

funcs = {#function, interval=(a,b)
1:(abs(x),(0,1)),
2:(sp.exp(sp.sin(x)),(0,2)),
3:(x**10, (0,1)),
4:(sp.exp(-(x-0.5)**2)-sp.exp(-0.25), (0,1)),
5:(sp.besselj(0,x),(0,100))
}
