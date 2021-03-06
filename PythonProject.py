# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 21:52:39 2019

@author: Sharlene
"""
#The goal is to use Runga Kutta to solve the Lane-Emden Equation by numerically
#integrating it and solving it for n = 1.5  and n = 3.5

import numpy as np
import matplotlib.pyplot as plt
#Defining the Lane-Emden Equationa and runga kutta
def LaneEmden(xi, Dn, z, n):
    return -1.0*np.power(Dn, n)-(2.0/xi)*z
def rk(func, h, x0, y0, z0, xf):
    yn=y0
    xn=x0
    zn=z0
    yvals=[]
    xvals=[]
    while xn < xf:
        k1=h*func(xn, yn, zn)
        l1=h*func(xn, yn, zn)
        k2=h*func(xn+0.5*h, yn+0.5*k1, zn+0.5*l1)
        l2=h*func(xn+0.5*h, yn+0.5*k1, zn+0.5*l1)
        k3=h*func(xn+0.5*h, yn+0.5*k2, zn+0.5*l2)
        l3=h*func(xn+0.5*h, yn+0.5*k2, zn+0.5*l2)
        k4=h*func(xn+h, yn+k3, zn+l3)
        l4=h*func(zn+h, yn+k3, zn+l3)
        zn += (k1+k2+k3+k4)/6.0
        yn += h*zn
        xn += h
        yvals.append(yn)
        xvals.append(xn)
    return(xvals,yvals)
if __name__ == "__main__":
#Using n = 1.5 and n = 3.5 for Lane Emden equations
    lane1 = lambda x, y, z: LaneEmden (x, y, z, 1.5)
    lane2 = lambda x, y, z: LaneEmden (x, y, z, 3.5)
#Calculating the points for the two solutions
(x1, y1) = rk(lane1, 0.01, 0.02, 0.999933333, -0.006666267, 3.65735)
(x2, y2) = rk(lane2, 0.01, 0.02, 0.99993305, -0.0066657, 9.536)
#Writing x1 and x2 as arrays, so that they can be plotted
x1 = np.asarray(x1)
x2 = np.asarray(x2)
p_15 = np.power(y1, 1.5)
p_35 = np.power(y2, 3.5)
plt.plot(x1 / max(x1), p_15, 'k-', label="n=1.5")
plt.plot(x2 / max(x2), p_35, 'k_', label="n=3.5")
plt.legend()
plt.ylabel("$D_n$")
plt.xlabel("$\\xi$")
plt.title("Solutions to the Lane-Emden Equation")
plt.ylim((0,1))
plt.savefig("Lane-Emden Equation Solutions")
plt.show()
# Plot for Temperature where T = y
plt.plot(x1 / max(x1), y1, 'k-', label="n=1.5")
plt.plot(x2 / max(x2), y2, 'k_', label="n=3.5")
plt.legend()
plt.ylabel("$Temperature(K)$")
plt.xlabel("$\\xi$")
plt.title("Temperature-Xi Plot for n = 1.5 and  n = 3.5")
plt.ylim(0,1)
plt.xlim(0,1)
plt.show()
# Plot for Pressure P which is proportional to y^n+1
P_15 = np.power(y1, 2.5)
P_35 = np.power(y2, 4.5)
plt.plot(x1 / max(x1), P_15, 'k-', label="n=1.5")
plt.plot(x2 / max(x2), P_35, 'k_', label="n=3.5")
plt.legend()
plt.ylabel("$Pressure(F/A)$")
plt.xlabel("$\\xi$")
plt.title("Pressure-Xi Plot for n = 1.5 and n = 3.5")
plt.ylim(0,1)
plt.xlim(0,1)
plt.show()