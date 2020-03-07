# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 21:35:07 2018

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 10:46:11 2017

@author: Admin
"""
import timeit
from PIL import Image
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from scipy.integrate import ode
import matplotlib
import matplotlib.pyplot as plt
from scipy import linalg as lin
from scipy.optimize import fsolve
from scipy.linalg import solve
from scipy.linalg import solve_banded
from scipy.special import jn_zeros #prvi parameter je order, drugi št. ničel
from scipy.special import jv #prvi order drugi argument
#from scipy.special import beta
import scipy.special as spec
import scipy.sparse
from scipy.optimize import root
from scipy.integrate import quad
from scipy.integrate import romb
from scipy.integrate import complex_ode
from scipy.integrate import simps
from scipy.optimize import linprog
import scipy.optimize as opt
from scipy.optimize import curve_fit
from scipy.stats import linregress
from scipy.linalg import svd
from matplotlib.patches import Ellipse
from matplotlib import gridspec
import time
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
plt.rc("text",usetex=True)
matplotlib.rcParams["text.latex.unicode"] = True
plt.close("all")
pi = np.pi

zacetnipribl = [0.1,0.1]

def k2(x,phi,Z,e): #Izracuna k kvadrat za dano schr. enacbo
    return 2*Z/x + 2*phi + e
def numerovkorak(y,x,h,phi,Z,e): #en korak numerova
    """y = [y_i-1,y_i], x = [x_i-1,x_i,x_i+1]"""
    faktor = 1/(1+h*h/12*k2(x[2],phi[2],Z,e))
    prvi = 2*(1-5*h*h/12*k2(x[1],phi[1],Z,e))
    drugi = -(1+h*h/12*k2(x[0],phi[0],Z,e))
    return faktor*(prvi*y[1]+drugi*y[0])
def numerov(Z,e,h,maxt,phi): #iz zacetnih pridemo do resitve shcr
    velikost= int(maxt/h)
    x = np.linspace(10**(-10),maxt,velikost)
    y = np.ones(velikost)
    y[0]=zacetnipribl[0]
    #print(zacetnipribl)
    y[1] = zacetnipribl[1] 
    for i in range(2,velikost):
        #print(i)
        y[i]= numerovkorak(y[i-2:i],x[i-2:i+1],h,phi[i-2:i+1],Z,e)
    return (x,y) #mogoce obiwan error?
def dobiPhi(x,R): #phi iz R
    vel = x.size
    phi = np.ones(vel)
    for i in range(vel):
        if i==0:
            phi[i] = - simps(R**2/x,x)
            continue
        elif i==vel-1:
            phi[i] = -1/x[i]*simps(R**2,x)
            continue
        phi[i] = -1/x[i]*simps(R[:i+1]**2,x[:i+1]) - simps(R[i:]**2/x[i:],x[i:])
    return phi
def zacetni(x,Z): #gen zacetni pogoj
    zz=Z-5/16
    return 2*np.sqrt(zz)*x*zz*np.exp(-zz*x)
def slabpribl(x): #neki zacetni pribl 
    if x<0.0001:
        return 0
    elif x<0.012:
        return 0.1
    if x%2 == 0:
        return 1
    else:
        return -1
def resiNumerov(Z,h,phi,maxt,eps=0.000000001): #bisekcija na e
    leva = -20
    desna = 0
    while (desna-leva) > eps:
        sredina = (desna+leva)/2
        #print(sredina)
        resitev = numerov(Z,sredina,h,maxt,phi)[1][-1]
        resitevleva = numerov(Z,leva,h,maxt,phi)[1][-1]
        if resitevleva*resitev < 0:
            desna = sredina
        else:
            leva = sredina
    pravaresitev = numerov(Z,sredina,h,maxt,phi)
    norm = np.sum(pravaresitev[1]**2 * h)
    return (pravaresitev[0],pravaresitev[1]/np.sqrt(norm))
def hartreeFock(Z,h,maxt,en=False): #HF iteracija
    x = np.linspace(10**(-10),maxt,int(maxt/h))
    global zacetnipribl
    #zacetna = np.sin(4*x)
    zacetna = np.vectorize(zacetni,excluded=["Z"])(x,Z)
    #zacetna = np.vectorize(slabpribl)(x)
    zacetnipribl = zacetna[:2]
    prejsnja = np.ones(x.size)
    energije = []
    potenciali = []
    rji = []
    i = 0
    while np.sum(np.abs(prejsnja-zacetna))>0.1:
        prejsnja = zacetna
        potencial = dobiPhi(x,zacetna)
        #rji.append(zacetna)
        #potenciali.append(potencial)
        #energije.append(zracunajE((x,zacetna,potencial,h,Z)))
        temp = resiNumerov(Z,h,potencial,maxt)
        x = temp[0]
        zacetna = temp[1]
        i+=1
    #return energije
    #return(x,rji,potenciali)
    if en:
        return zracunajE((x,zacetna,potencial,h,Z))
        #return (x,zacetna,potencial,h,Z)
    return (x,zacetna,potencial)
def zracunajE(resitev): #dejanski E
    odvod = np.gradient(resitev[1],resitev[3])
    e0 = 13.6058
    #print(odvod)
    return 2*e0*(simps(odvod**2,resitev[0]) -2*resitev[4]*simps(resitev[1]**2/resitev[0],resitev[0]) - simps(resitev[1]**2 * resitev[2],resitev[0]))

#resitev = hartreeFock(3,0.01,5,True)
#print(zracunajE(resitev))
#plt.plot(resitev[0],resitev[1],label=r"$R(x)$")
#plt.plot(resitev[0],resitev[1]/(np.sqrt(4*pi)*resitev[0]),label=r"$\varphi_{1s}(x)$")
#plt.plot(resitev[0],resitev[2],label=r"$\Phi(x)$")
#plt.xlabel("x")
#plt.title("Valovna funkcija in potencial, Z=3")
#plt.legend(loc="best")
#plt.savefig("prva/funkcijelitij.pdf")

zji = np.linspace(1.1,3,30)
energije = []
for z in zji:
    if z>=2.2:
        energije.append(hartreeFock(z,0.01,5,True))
    elif z>=1.7:
        energije.append(hartreeFock(z,0.01,7,True))
    elif z>=1.5:
        energije.append(hartreeFock(z,0.01,10,True))
    elif z>=1.3:
        energije.append(hartreeFock(z,0.01,12,True))
    else:
        energije.append(hartreeFock(z,0.01,14,True))
     
             
koef = np.polyfit(zji,energije,4)
z = np.linspace(0.5,5,100)
y = koef[0]*z**4 + koef[1]*z**3 + koef[2]*z**2 + koef[3]*z+koef[4]
plt.plot(zji,energije,".",color="k")
plt.plot(z,y,color="b")
#plt.legend(loc="best")
en = koef[0] + koef[1] + koef[2]+ koef[3]+koef[4]
plt.axvline(x=1,color="r")
plt.text(1,en,str(round(en,2)))
plt.xlabel("x")
plt.title(r"Vezavna energija")
plt.ylabel("E")
plt.savefig("prva/energijevodik.pdf")   
"""
resitev = hartreeFock(1,0.01,17,True)
#print(len(resitev))
plt.plot(range(len(resitev)),resitev,color="blue")
plt.text(0,resitev[-1],str(round(resitev[-1],2)))
plt.xlim(0)
plt.title(r"Konvergenca E, $H^{-}$")
#plt.savefig("prva/energijevodik.pdf")
"""
"""
cmap1 = plt.get_cmap("cool")
parameters = np.linspace(0,12,13)
colors = cmap1(np.linspace(0,1,parameters.size))

x = np.linspace(0,10,1000)
for i,j in zip(range(parameters.size),colors):
    if i==0:
        plt.plot(x,x*parameters[i],color=j,label=r"$\Phi^0$")
    elif i==parameters.size-1:
        plt.plot(x,x*parameters[i],color=j,label=r"$\Phi$")
    else:
        plt.plot(x,x*parameters[i],color=j)
plt.legend(loc="best")  
plt.xlabel("x")
plt.title("Potencial, $R^0 = 1-e^{-x}$")
plt.savefig("prva/funkcije3U.pdf")    
"""

"""
energije = hartreeFock(2,0.01,7)
cmap1 = plt.get_cmap("cool")
cmap2 = plt.get_cmap("winter")
barve = cmap2(np.linspace(0.05,1,len(energije[2][:6])))
x = energije[0]
for i,j in zip(range(len(energije[2][:6])),barve):
    if i==0:
        #plt.plot(x,energije[2][i],color=j,label=r"$\Phi^0$")
        plt.plot(x,energije[1][i],color=j,label=r"$R^0$")
    elif i==len(energije[2][:5])-1:
       # plt.plot(x,energije[2][i],color=j,label=r"$\Phi$")
        plt.plot(x,energije[1][i],color=j,label=r"$R$")        
    else:
        plt.plot(x,energije[1][i],color=j)
        #plt.plot(x,energije[2][i],color=j)
plt.legend(loc="best")  
plt.xlabel("x")
plt.title("R(x), $R^0 = \sin 5x$")
plt.savefig("prva/funkcijeR5.pdf")        
"""  
    
    
