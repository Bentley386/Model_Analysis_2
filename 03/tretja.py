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
from scipy.integrate import complex_ode
from scipy.optimize import linprog
import scipy.optimize as opt
from scipy.optimize import curve_fit
from scipy.stats import linregress
from scipy.linalg import svd
from matplotlib.patches import Ellipse
from matplotlib import gridspec
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
plt.rc("text",usetex=True)
matplotlib.rcParams["text.latex.unicode"] = True
plt.close("all")
pi = np.pi

def k2(x,l,e):
    return 2/x - l*(l+1)/(x*x) + e
def schrDE(x,R,l,e):
    return [R[1],(-2/x + l*(l+1)/(x*x) - e)*R[0]]
"""
def prestejnicle(y):
     indeksi=[0]
     predznak = 1
     for i in range(1,y.size):
         if y[i]*predznak <0 :
             predznak *=-1
             indeksi.append(i)    
     return len(indeksi)
"""
def prestejnicle(y):
    stevilo=0
    for i in range(y.size-1):
        if np.sign(y[i]) != np.sign(y[i+1]):
            stevilo+=1
    return stevilo    
def RK(kot0,l,e,h,maxt):
     r = ode(schrDE).set_integrator("dopri5")
     r.set_initial_value([0,5*pi/6], 10**(-10)).set_f_params(l,e)
     casi = np.linspace(0,maxt,int(maxt/h)+1)
     y = np.ones(casi.size)
     y[0]=0
     i=1
     for i in range(1,casi.size):
         y[i] = r.integrate(r.t+h)[0]
         i+=1
     return (casi,y)
def resitevRK4(n,l,h,maxt):
    velikost = int(maxt/h)+1
    x = np.linspace(10**(-10),maxt,velikost+1)
    y = np.ones(velikost)
    y[0]=0
    leva = -2
    desna = 0
    energija = 100
    while (desna-leva) > 0.000001:
        sredina = (desna+leva)/2
        resitev = prestejnicle(RK(pi/4,l,sredina,h,maxt)[1])
        #resitevleva = prestejnicle(RK(pi/4,l,leva,h,maxt)[1])
        #resitevdesna = prestejnicle(RK(pi/4,l,desna,h,maxt)[1])
        """
        if resitev==(n-l):
            energija = sredina
            break
        elif resitevleva==(n-l):
            energija = leva
            break
        elif resitevdesna == (n-l):
            energija = desna
            break
        """
        if resitev > (n-l):
            desna = sredina
        else:
            leva = sredina
    energija = sredina
    pravaresitev = np.array(RK(pi/4,l,energija,h,maxt)[1])
    norm = np.sum(pravaresitev**2 * h)
    return (x[:-1],pravaresitev/np.sqrt(norm))
def priblizek(l,e,h,st):
    koef = [((3*e*e+10*e-2)/360,(2*e-1)/18,(2-e)/6,-1,1,0),((7*e-2)/180,(1-e)/10,-0.5,1,0,0),((2/3-e)/14,-1/3,1,0,0,0)]
    return np.polyval(koef[l][st:],h)
def numerovkorak(y,x,h,l,e):
    """y = [y_i-1,y_i], x = [x_i-1,x_i,x_i+1]"""
    faktor = 1/(1+h*h/12*k2(x[2],l,e))
    prvi = 2*(1-5*h*h/12*k2(x[1],l,e))
    drugi = -(1+h*h/12*k2(x[0],l,e))
    return faktor*(prvi*y[1]+drugi*y[0])
def numerov(l,e,h,maxt,st):
    velikost= int(maxt/h)+1
    x = np.linspace(10**(-10),maxt,velikost+1)
    y = np.ones(velikost)
    y[0]=0
    y[1] = priblizek(l,e,h,st)
    for i in range(2,velikost):
        y[i]= numerovkorak(y[i-2:i],x[i-2:i+1],h,l,e)
    return (x[:-1],y)
def resitevnumerov(n,l,h,maxt,st=0,vrni=False,eps=0.0000001):
    velikost = int(maxt/h)+1
    x = np.linspace(10**(-10),maxt,velikost+1)
    leva = -2
    desna = 0
    energija = 100
    while (desna-leva) > eps:
        sredina = (desna+leva)/2
        #print(sredina)
        resitev = prestejnicle(numerov(l,sredina,h,maxt,st)[1])
        #resitevleva = prestejnicle(numerov(l,leva,h,maxt,st)[1])
        #resitevdesna = prestejnicle(numerov(l,desna,h,maxt,st)[1])
        """
        if resitev==(n-l):
            energija = sredina
            break
        elif resitevleva==(n-l):
            energija = leva
            break
        elif resitevdesna == (n-l):
            energija = desna
            break
        """ 
        if resitev > (n-l):
            desna = sredina
        else:
            leva = sredina
    energija = sredina
    pravaresitev = np.array(numerov(l,energija,h,maxt,st)[1])
    norm = np.sum(pravaresitev**2 * h)
    if vrni:
        return (x[:-1],pravaresitev/np.sqrt(norm),energija) 
    if np.sqrt(norm)==0:
        print(st)
    return (x[:-1],pravaresitev/np.sqrt(norm))


def preveriasimptotiko(y):
    for i in range(100,y.size):
        if sum(np.abs(y[i:i+100])) < 100:
            return i
def R10(x):
    return 2*x*np.exp(-x)
def R20(x):
    return 1/np.sqrt(2)*x*(1-0.5*x)*np.exp(-x/2)
def R21(x):
    return 1/np.sqrt(24)*x*x*np.exp(-x/2)            
    
def n(x):
    if x<1:
        return 2-0.5*x*x
    else:
        return 1
def k22(x,k,lamb):
    nn = n(x)
    return 1/(4*x*x) + nn*nn*k*k - lamb*lamb
def numerovkorak2(y,x,h,k,lamb):
    """y = [y_i-1,y_i], x = [x_i-1,x_i,x_i+1]"""
    faktor = 1/(1+h*h/12*k22(x[2],k,lamb))
    prvi = 2*(1-5*h*h/12*k22(x[1],k,lamb))
    drugi = -(1+h*h/12*k22(x[0],k,lamb))
    return faktor*(prvi*y[1]+drugi*y[0])

def druga(k,lamb,h,maxt,y1=0.001):
    velikost= int(maxt/h)+1
    x = np.linspace(10**(-10),maxt,velikost+1)
    y = np.ones(velikost)
    y[0]=0
    y[1] = 0.001
    for i in range(2,velikost):
        y[i]= numerovkorak2(y[i-2:i],x[i-2:i+1],h,k,lamb)
    return (x[:-1],y)    
def pomoznaDruga(k,h,maxt):
    resitve = []
    for i in np.linspace(0,20,10000):
        res = druga(k,i,h,maxt)
        resitve.append(res[1][-1])
    return (np.linspace(0,20,10000),resitve)
def bisekcijaDruga(lleva,ddesna,k,h,maxt,eps=0.000001):
    leva = lleva
    desna = ddesna
    while (desna-leva) > eps:
        sredina = (desna+leva)/2
        #print(sredina)
        resitev = druga(k,sredina,h,maxt)[1][-1]
        resitevleva = druga(k,leva,h,maxt)[1][-1]
        resitevdesna = druga(k,desna,h,maxt)[1][-1]
        if resitevleva*resitev < 0:
            desna = sredina
        else:
            leva = sredina
        """
        if resitev==(n-l):
            energija = sredina
            break
        elif resitevleva==(n-l):
            energija = leva
            break
        elif resitevdesna == (n-l):
            energija = desna
            break
        """ 
    print(sredina)
    pravaresitev = druga(k,sredina,h,maxt)
    norm = np.sum(pravaresitev[1]**2 * h)
    return (pravaresitev[0],pravaresitev[1]/np.sqrt(norm))
osnovne = [0.822756,0.9635126,1.124537,1.48729019,1.87890906,2.28354635,2.6911835,3.102257728,
           4.12076091,3.14400577,4.2194736,5.328394889,6.419220924,
           5.1342058,6.28101,7.45965,8.6111,7.10421,8.2869577,
           9.5186605,10.7230577,9.0672,10.284749,11.559233]
prve = [5.137991905,6.1480512,7.15455913,8.15977954,7.48861536026,8.5355749,9.5719118,
        10.59869,9.732350349,10.82772731,11.90233707,12.96058368,11.905053,13.045945,14.16118]
druge = [9.16242122,10.164101600,11.1651735,12.165858,11.62167,12.63662624,13.64804,14.656941,14.01065,15.0463247,16.074431]
tretje = [13.16784,14.1680955,15.168238,16.16830158,17.16830438,18.1708040,19.170725822]
plt.plot([0.8,0.9,1,1.2,1.4,1.6,1.8,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10],osnovne)
plt.plot([3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10],prve)
plt.plot([5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10],druge)
plt.plot([7,7.5,8,8.5,9,9.5,10],tretje)
plt.title(r"$\lambda(k)$")
plt.savefig("druga/disperzija.pdf")
if 0:
    #narisem prvih 10
    #l=0
    fig, ax = plt.subplots(3)
    colormap = plt.get_cmap("rainbow")
    barve = np.linspace(0,1,8)
    resitev = druga(3,prve[0],0.01,1.2)
    plt.suptitle("Prva vzbujena stanja")
    ax[0].plot(resitev[0],resitev[1],label="k=3",color=colormap(barve[0]))
    #ax[0].plot(np.linspace(10,40,20),np.ones(20)*resitev[1][-1],"--",color=colormap(barve[0]))
    resitev = druga(4,prve[1],0.01,1.2)
    ax[0].plot(resitev[0],resitev[1],label="k=4",color=colormap(barve[1]))
    #ax[0].plot(np.linspace(20,40,20),np.ones(20)*resitev[1][-1],"--",color=colormap(barve[1]))
    resitev = druga(5,prve[2],0.01,1.2)
    ax[0].plot(resitev[0],resitev[1],label="k=5",color=colormap(barve[2]))
    #ax[0].plot(np.linspace(0,1,100),np.vectorize(n)(np.linspace(0,1,100)),"--",color="k")
    ax[0].legend(loc="upper right")
    resitev = druga(6,prve[3],0.01,2.5)
    ax[1].plot(resitev[0],resitev[1],label="k=6",color=colormap(barve[3]))
    #ax[1].plot(np.linspace(60,120,20),np.ones(20)*resitev[1][-1],"--",color=colormap(barve[3]))
    resitev = druga(7,prve[4],0.01,2.5)
    ax[1].plot(resitev[0],resitev[1],label="k=7",color=colormap(barve[4]))
    #ax[1].plot(np.linspace(90,120,20),np.ones(20)*resitev[1][-1],"--",color=colormap(barve[4]))
    resitev = druga(8,prve[5],0.01,1.2)
    ax[1].plot(resitev[0],resitev[1],label="k=8",color=colormap(barve[5]))
    #ax[1].plot(np.linspace(0,1,100),np.vectorize(n)(np.linspace(0,1,100)),"--",color="k")
    ax[1].legend(loc="best")   
    resitev = druga(9,prve[6],0.01,1.2)
    ax[2].plot(resitev[0],resitev[1],label="k=9",color=colormap(barve[6]))
    #ax[2].plot(np.linspace(170,270,20),np.ones(20)*resitev[1][-1],"--",color=colormap(barve[6]))
    resitev = druga(10,prve[7],0.01,1.2)
    ax[2].plot(resitev[0],resitev[1],label="k=10",color=colormap(barve[7]))
    ax[2].legend(loc="upper right")
    plt.savefig("druga/prva.pdf")
if 0:
    #narisem prvih 10
    #l=0
    fig, ax = plt.subplots(3)
    colormap = plt.get_cmap("rainbow")
    barve = np.linspace(0,1,10)
    resitev = druga(1,osnovne[2],0.01,5)
    plt.suptitle("Osnovna stanja")
    ax[0].plot(resitev[0],resitev[1],label="k=1",color=colormap(barve[0]))
    #ax[0].plot(np.linspace(10,40,20),np.ones(20)*resitev[1][-1],"--",color=colormap(barve[0]))
    resitev = druga(2,osnovne[7],0.01,2.5)
    ax[0].plot(resitev[0],resitev[1],label="k=2",color=colormap(barve[1]))
    #ax[0].plot(np.linspace(20,40,20),np.ones(20)*resitev[1][-1],"--",color=colormap(barve[1]))
    resitev = druga(3,osnovne[9],0.01,2.5)
    ax[0].plot(resitev[0],resitev[1],label="k=3",color=colormap(barve[2]))
    #ax[0].plot(np.linspace(0,1,100),np.vectorize(n)(np.linspace(0,1,100)),"--",color="k")
    ax[0].legend(loc="upper right")
    resitev = druga(4,osnovne[11],0.01,1.5)
    ax[1].plot(resitev[0],resitev[1],label="k=4",color=colormap(barve[3]))
    #ax[1].plot(np.linspace(60,120,20),np.ones(20)*resitev[1][-1],"--",color=colormap(barve[3]))
    resitev = druga(5,osnovne[13],0.01,1.5)
    ax[1].plot(resitev[0],resitev[1],label="k=5",color=colormap(barve[4]))
    #ax[1].plot(np.linspace(90,120,20),np.ones(20)*resitev[1][-1],"--",color=colormap(barve[4]))
    resitev = druga(6,osnovne[15],0.01,1.5)
    ax[1].plot(resitev[0],resitev[1],label="k=6",color=colormap(barve[5]))
    #ax[1].plot(np.linspace(0,1,100),np.vectorize(n)(np.linspace(0,1,100)),"--",color="k")
    ax[1].legend(loc="best")   
    resitev = druga(7,osnovne[17],0.01,2.5)
    ax[2].plot(resitev[0],resitev[1],label="k=7",color=colormap(barve[6]))
    #ax[2].plot(np.linspace(170,270,20),np.ones(20)*resitev[1][-1],"--",color=colormap(barve[6]))
    resitev = druga(8,osnovne[19],0.01,1.3)
    ax[2].plot(resitev[0],resitev[1],label="k=8",color=colormap(barve[7]))
    #ax[2].plot(np.linspace(170,270,20),np.ones(20)*resitev[1][-1],"--",color=colormap(barve[7]))
    resitev = druga(9,osnovne[21],0.01,2.5)
    ax[2].plot(resitev[0],resitev[1],label="k=9",color=colormap(barve[8]))
    #ax[2].plot(np.linspace(230,270,20),np.ones(20)*resitev[1][-1],"--",color=colormap(barve[8]))
    resitev = druga(10,osnovne[23],0.01,1.3)
    ax[2].plot(resitev[0],resitev[1],label="k=10",color=colormap(barve[9]))
    #ax[2].plot(np.linspace(0,1,100),np.vectorize(n)(np.linspace(0,1,100)),"--",color="k")
    ax[2].legend(loc="upper right")
    plt.savefig("druga/osnovna.pdf")
if 0:
    res= pomoznaDruga(10,0.01,2.5)
    plt.plot(res[0],np.abs(res[1]))
    plt.yscale("log")   

if 0:
    k = 10
    lamb = bisekcijaDruga(11,12,k,0.01,2.5)
    plt.plot(lamb[0],lamb[1])    
    lamb = bisekcijaDruga(14,15,k,0.01,2.5)
    plt.plot(lamb[0],lamb[1])    
    lamb = bisekcijaDruga(16,17,k,0.01,2)
    plt.plot(lamb[0],lamb[1])    
    lamb = bisekcijaDruga(17,18,k,0.01,2)
    plt.plot(lamb[0],lamb[1]) 
    lamb = bisekcijaDruga(19,20,k,0.01,1.5)
    plt.plot(lamb[0],lamb[1]) 
    
    
    
if 0:
    hji = np.array([0,1,2,3,4,5])
    napake1 = []
    maxodmik1 = []
    napake2 = []
    maxodmik2 = []
    napake3 = []
    maxodmik3 = []
    for i in hji:
        rezultat = resitevnumerov(1,0,0.01,10,st=i)
        prava = np.vectorize(R10)(rezultat[0])
        napake1.append(np.sum(np.abs(rezultat[1]-prava))/prava.size)
        maxodmik1.append(np.amax(np.abs(rezultat[1]-prava)))
        rezultat = resitevnumerov(2,0,0.01,25,st=i)
        prava = np.vectorize(R20)(rezultat[0])
        napake2.append(np.sum(np.abs(rezultat[1]-prava))/prava.size)
        maxodmik2.append(np.amax(np.abs(rezultat[1]-prava)))
        rezultat = resitevnumerov(2,1,0.01,25,st=i)
        prava = np.vectorize(R21)(rezultat[0])
        napake3.append(np.sum(np.abs(rezultat[1]-prava))/prava.size)
        maxodmik3.append(np.amax(np.abs(rezultat[1]-prava)))
    plt.title("Povprečna in maksimalna napaka metode Numerova")
    plt.plot(hji,napake1,label=r"Povp. napaka $R_{10}$",color="red")
    plt.plot(hji,maxodmik1,"--",label=r"Max. napaka $R_{10}$",color="red")
    plt.plot(hji,napake2,label=r"Povp. napaka $R_{20}$",color="blue")
    plt.plot(hji,maxodmik2,"--",label=r"Max. napaka $R_{20}$",color="blue")
    plt.plot(hji,napake3,label=r"Povp. napaka $R_{21}$",color="green")
    plt.plot(hji,maxodmik3,"--",label=r"Max. napaka $R_{21}$",color="green")
    plt.yscale("log")
    #plt.xscale("log")
    plt.legend(loc="best")
    plt.xlabel(r"st")
    plt.savefig("prva/odst.pdf")
if 0:
    hji = np.array([0.0000001,0.000001,0.0000025,0.000005,0.0000075,0.00001,0.000025,0.00005,0.000075,0.0001,0.001,0.0025,0.005,0.0075,0.01,0.02])
    napake1 = []
    maxodmik1 = []
    napake2 = []
    maxodmik2 = []
    napake3 = []
    maxodmik3 = []
    for i in hji:
        rezultat = resitevnumerov(1,0,0.01,10,eps=i)
        prava = np.vectorize(R10)(rezultat[0])
        napake1.append(np.sum(np.abs(rezultat[1]-prava))/prava.size)
        maxodmik1.append(np.amax(np.abs(rezultat[1]-prava)))
        rezultat = resitevnumerov(2,0,0.01,25,eps=i)
        prava = np.vectorize(R20)(rezultat[0])
        napake2.append(np.sum(np.abs(rezultat[1]-prava))/prava.size)
        maxodmik2.append(np.amax(np.abs(rezultat[1]-prava)))
        rezultat = resitevnumerov(2,1,0.01,25,eps=i)
        prava = np.vectorize(R21)(rezultat[0])
        napake3.append(np.sum(np.abs(rezultat[1]-prava))/prava.size)
        maxodmik3.append(np.amax(np.abs(rezultat[1]-prava)))
    plt.title("Povprečna in maksimalna napaka metode Numerova")
    plt.plot(hji,napake1,label=r"Povp. napaka $R_{10}$",color="red")
    plt.plot(hji,maxodmik1,"--",label=r"Max. napaka $R_{10}$",color="red")
    plt.plot(hji,napake2,label=r"Povp. napaka $R_{20}$",color="blue")
    plt.plot(hji,maxodmik2,"--",label=r"Max. napaka $R_{20}$",color="blue")
    plt.plot(hji,napake3,label=r"Povp. napaka $R_{21}$",color="green")
    plt.plot(hji,maxodmik3,"--",label=r"Max. napaka $R_{21}$",color="green")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend(loc="best")
    plt.xlabel(r"$\epsilon$")
    plt.savefig("prva/odeps.pdf")
    
    
    
    
    
if 0:
    hji = np.array([0.0001,0.001,0.0025,0.005,0.0075,0.01,0.025,0.05,0.075,0.1,0.25,0.5,0.75,1])
    napake1 = []
    maxodmik1 = []
    napake2 = []
    maxodmik2 = []
    napake3 = []
    maxodmik3 = []
    for i in hji:
        rezultat = resitevnumerov(1,0,i,10)
        prava = np.vectorize(R10)(rezultat[0])
        napake1.append(np.sum(np.abs(rezultat[1]-prava))/prava.size)
        maxodmik1.append(np.amax(np.abs(rezultat[1]-prava)))
        rezultat = resitevnumerov(2,0,i,25)
        prava = np.vectorize(R20)(rezultat[0])
        napake2.append(np.sum(np.abs(rezultat[1]-prava))/prava.size)
        maxodmik2.append(np.amax(np.abs(rezultat[1]-prava)))
        rezultat = resitevnumerov(2,1,i,25)
        prava = np.vectorize(R21)(rezultat[0])
        napake3.append(np.sum(np.abs(rezultat[1]-prava))/prava.size)
        maxodmik3.append(np.amax(np.abs(rezultat[1]-prava)))
    plt.title("Povprečna in maksimalna napaka metode Numerova")
    plt.plot(hji,napake1,label=r"Povp. napaka $R_{10}$",color="red")
    plt.plot(hji,maxodmik1,"--",label=r"Max. napaka $R_{10}$",color="red")
    plt.plot(hji,napake2,label=r"Povp. napaka $R_{20}$",color="blue")
    plt.plot(hji,maxodmik2,"--",label=r"Max. napaka $R_{20}$",color="blue")
    plt.plot(hji,napake3,label=r"Povp. napaka $R_{21}$",color="green")
    plt.plot(hji,maxodmik3,"--",label=r"Max. napaka $R_{21}$",color="green")
    plt.yscale("log")
    plt.legend(loc="best")
    plt.xlabel("h")
    plt.savefig("prva/odh.pdf")
if 0:
    runge20 = resitevRK4(2,0,0.01,25)            
    numerov20 = resitevnumerov(2,0,0.01,25)
    prava = np.vectorize(R20)(runge20[0])
    runge21 = resitevRK4(2,1,0.01,25)            
    numerov21 = resitevnumerov(2,1,0.01,25)
    prava2 = np.vectorize(R21)(runge21[0])
    plt.plot(runge20[0],np.abs(runge20[1]-prava),label=r"RK4, $R_{20}$",color="red")            
    plt.plot(runge21[0],np.abs(runge21[1]-prava2),label=r"RK4, $R_{21}$",color="blue")            
    plt.plot(runge20[0],np.abs(numerov20[1]-prava),label=r"Numerov, $R_{20}$",color="magenta")                        
    plt.plot(runge21[0],np.abs(numerov21[1]-prava2),label=r"Numerov, $R_{21}$",color="cyan")
    plt.xlim(0.01,25)
    plt.ylim(10**(-7),10**(-3))
    plt.legend(loc="best")
    plt.yscale("log")
    plt.title("Primerjava napak metod")
    plt.savefig("prva/primerjava.pdf")
if 0:
    #primerjamo energije            
    limits = [10,25,40,60,90,120,170,170,230,270]
    for i in range(len(limits)):
        resitev = resitevnumerov(i+1,0,0.01,limits[i],vrni=True)[2]
        plt.plot([i+1],[resitev],"ko")
        if i>1:
            resitev = resitevnumerov(i+1,2,0.01,limits[i],vrni=True)[2]
            plt.plot([i+1],[resitev],"ro")
    x = np.linspace(1,10,100)
    plt.title("e(n)")
    plt.ylabel("e")
    plt.xlabel("n")
    plt.plot(x,-1/(x**2),"b")
    plt.savefig("prva/energije.pdf")
if 0:
    #narisem prvih 10
    #l=2
    fig, ax = plt.subplots(3)
    colormap = plt.get_cmap("rainbow")
    barve = np.linspace(0,1,8)
    plt.suptitle("l=2")
    resitev = resitevnumerov(3,2,0.01,40)
    ax[0].plot(resitev[0],resitev[1],label="n=3",color=colormap(barve[0]))
    ax[0].plot(np.linspace(40,90,20),np.ones(20)*resitev[1][-1],"--",color=colormap(barve[0]))
    resitev = resitevnumerov(4,2,0.01,60)    
    ax[0].plot(resitev[0],resitev[1],label="n=4",color=colormap(barve[1]))
    ax[0].plot(np.linspace(60,90,20),np.ones(20)*resitev[1][-1],"--",color=colormap(barve[1]))
    resitev = resitevnumerov(5,2,0.01,90)
    ax[0].plot(resitev[0],resitev[1],label="n=5",color=colormap(barve[2]))
    ax[0].legend(loc="upper right")
    resitev = resitevnumerov(6,2,0.01,120)
    ax[1].plot(resitev[0],resitev[1],label="n=6",color=colormap(barve[3]))
    ax[1].plot(np.linspace(120,170,20),np.ones(20)*resitev[1][-1],"--",color=colormap(barve[3]))
    resitev = resitevnumerov(7,2,0.01,170)    
    ax[1].plot(resitev[0],resitev[1],label="n=7",color=colormap(barve[4]))
    resitev = resitevnumerov(8,2,0.01,170)
    ax[1].plot(resitev[0],resitev[1],label="n=8",color=colormap(barve[5]))
    ax[1].legend(loc="best")   
    resitev = resitevnumerov(9,2,0.01,230)
    ax[2].plot(resitev[0],resitev[1],label="n=9",color=colormap(barve[6]))
    ax[2].plot(np.linspace(230,270,20),np.ones(20)*resitev[1][-1],"--",color=colormap(barve[6]))
    resitev = resitevnumerov(10,2,0.01,270)
    ax[2].plot(resitev[0],resitev[1],label="n=10",color=colormap(barve[7]))
    ax[2].legend(loc="best")
    plt.savefig("prva/l2.pdf") 

            
if 0:
    #narisem prvih 10
    #l=1
    fig, ax = plt.subplots(3)
    colormap = plt.get_cmap("rainbow")
    barve = np.linspace(0,1,9)
    plt.suptitle("l=1")
    resitev = resitevnumerov(2,1,0.01,25)
    ax[0].plot(resitev[0],resitev[1],label="n=2",color=colormap(barve[0]))
    ax[0].plot(np.linspace(20,40,20),np.ones(20)*resitev[1][-1],"--",color=colormap(barve[0]))
    resitev = resitevnumerov(3,1,0.01,40)
    ax[0].plot(resitev[0],resitev[1],label="n=3",color=colormap(barve[1]))
    ax[0].legend(loc="upper right")
    resitev = resitevnumerov(4,1,0.01,60)    
    ax[1].plot(resitev[0],resitev[1],label="n=4",color=colormap(barve[2]))
    ax[1].plot(np.linspace(60,120,20),np.ones(20)*resitev[1][-1],"--",color=colormap(barve[2]))
    resitev = resitevnumerov(5,1,0.01,90)
    ax[1].plot(resitev[0],resitev[1],label="n=5",color=colormap(barve[3]))
    ax[1].plot(np.linspace(90,120,20),np.ones(20)*resitev[1][-1],"--",color=colormap(barve[3]))
    resitev = resitevnumerov(6,1,0.01,120)
    ax[1].plot(resitev[0],resitev[1],label="n=6",color=colormap(barve[4]))
    ax[1].legend(loc="best")   
    resitev = resitevnumerov(7,1,0.01,170)    
    ax[2].plot(resitev[0],resitev[1],label="n=7",color=colormap(barve[5]))
    ax[2].plot(np.linspace(170,270,20),np.ones(20)*resitev[1][-1],"--",color=colormap(barve[5]))
    resitev = resitevnumerov(8,1,0.01,170)
    ax[2].plot(resitev[0],resitev[1],label="n=8",color=colormap(barve[6]))
    ax[2].plot(np.linspace(170,270,20),np.ones(20)*resitev[1][-1],"--",color=colormap(barve[6]))
    resitev = resitevnumerov(9,1,0.01,230)
    ax[2].plot(resitev[0],resitev[1],label="n=9",color=colormap(barve[7]))
    ax[2].plot(np.linspace(230,270,20),np.ones(20)*resitev[1][-1],"--",color=colormap(barve[7]))
    resitev = resitevnumerov(10,1,0.01,270)
    ax[2].plot(resitev[0],resitev[1],label="n=10",color=colormap(barve[8]))
    ax[2].legend(loc="best")
    plt.savefig("prva/l1.pdf")            
            
            
            
            
            
if 0:
    #narisem prvih 10
    #l=0
    fig, ax = plt.subplots(3)
    colormap = plt.get_cmap("rainbow")
    barve = np.linspace(0,1,10)
    resitev = resitevnumerov(1,0,0.1,10)
    plt.suptitle("l=0")
    ax[0].plot(resitev[0],resitev[1],label="n=1",color=colormap(barve[0]))
    ax[0].plot(np.linspace(10,40,20),np.ones(20)*resitev[1][-1],"--",color=colormap(barve[0]))
    resitev = resitevnumerov(2,0,0.01,25)
    ax[0].plot(resitev[0],resitev[1],label="n=2",color=colormap(barve[1]))
    ax[0].plot(np.linspace(20,40,20),np.ones(20)*resitev[1][-1],"--",color=colormap(barve[1]))
    resitev = resitevnumerov(3,0,0.01,40)
    ax[0].plot(resitev[0],resitev[1],label="n=3",color=colormap(barve[2]))
    ax[0].legend(loc="upper right")
    resitev = resitevnumerov(4,0,0.01,60)    
    ax[1].plot(resitev[0],resitev[1],label="n=4",color=colormap(barve[3]))
    ax[1].plot(np.linspace(60,120,20),np.ones(20)*resitev[1][-1],"--",color=colormap(barve[3]))
    resitev = resitevnumerov(5,0,0.01,90)
    ax[1].plot(resitev[0],resitev[1],label="n=5",color=colormap(barve[4]))
    ax[1].plot(np.linspace(90,120,20),np.ones(20)*resitev[1][-1],"--",color=colormap(barve[4]))
    resitev = resitevnumerov(6,0,0.01,120)
    ax[1].plot(resitev[0],resitev[1],label="n=6",color=colormap(barve[5]))
    ax[1].legend(loc="best")   
    resitev = resitevnumerov(7,0,0.01,170)    
    ax[2].plot(resitev[0],resitev[1],label="n=7",color=colormap(barve[6]))
    ax[2].plot(np.linspace(170,270,20),np.ones(20)*resitev[1][-1],"--",color=colormap(barve[6]))
    resitev = resitevnumerov(8,0,0.01,170)
    ax[2].plot(resitev[0],resitev[1],label="n=8",color=colormap(barve[7]))
    ax[2].plot(np.linspace(170,270,20),np.ones(20)*resitev[1][-1],"--",color=colormap(barve[7]))
    resitev = resitevnumerov(9,0,0.01,230)
    ax[2].plot(resitev[0],resitev[1],label="n=9",color=colormap(barve[8]))
    ax[2].plot(np.linspace(230,270,20),np.ones(20)*resitev[1][-1],"--",color=colormap(barve[8]))
    resitev = resitevnumerov(10,0,0.01,270)
    ax[2].plot(resitev[0],resitev[1],label="n=10",color=colormap(barve[9]))
    ax[2].legend(loc="best")
    plt.savefig("prva/l0.pdf")