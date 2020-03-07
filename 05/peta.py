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
from numba import jit
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
plt.rc("text",usetex=True)
matplotlib.rcParams["text.latex.unicode"] = True
plt.close("all")
pi = np.pi

def temperaturniprofil(h,tempplasc,tok,omega):
    """vektor T"""
    iksi = np.linspace(0,1,int(1/h)) #rji
    ipsiloni = np.linspace(0,1,int(1/h)) #zji
    zacetna = np.asarray([[1.0 for iks in iksi] for ips in ipsiloni]) #vsaka vrstica svoj z
    zacetna[0] = np.ones(iksi.size)*0.0
    #zacetna[-2]-zacetna[-4]/2h = K
    zacetna[-2] = tok*2*h + np.ones(iksi.size)
    zacetna[-1]=np.ones(iksi.size)*0.0
    zacetna[:,-1] = np.ones(iksi.size)*tempplasc
    def gaussTemp(u):
        velikost = u[0].size
        uu = np.copy(u)
        def pomozna(i,j): #i je z, j je r
            if i==0 or i==velikost-1:
                return 0
            elif j==0 and i>=2 and i<=velikost-3:
                return uu[i][1]
            elif j==0:
                return 0
            elif j==velikost-1 and i>=2 and i<= velikost-3:
                return tempplasc
            elif j==velikost-1:
                return 0
            elif i==3:
                return uu[1][j]
            elif i==velikost-4:
                return tok*2*h + uu[velikost-2][j]
            return 2/5*((1+1/((j+1)))*uu[i][j+1]+(1-1/((j+1)))*uu[i][j-1] + 1/4*uu[i+1][j]+1/4*uu[i-1][j])
        for i in range(velikost):
            for j in range(velikost):
                uu[i][j] = u[i][j]+omega*(pomozna(i,j)-u[i][j])
        return uu
    temp = 0
    for i in range(100):
        if i==99:
            temp = np.copy(zacetna)
        zacetna = gaussTemp(zacetna)
    return np.linalg.norm(zacetna-temp)
    #cs =plt.imshow(zacetna,cmap=plt.get_cmap("hot"),aspect=1)
    #plt.title(r"Temp. plasca = 2,1, Tok = 1")
    #plt.savefig("valj4.pdf")
    #plt.colorbar(cs)
def pogojimacka(u,velikost):
    """predp da je x.shape (sodo,sodo)"""
    x = u[0]+1
    y = u[1]+1
    cetrtina = int(velikost/4)
    if x> velikost/2:
        x = -x+2*int(velikost/2) #s tem lahko gledam samo levo stran
    if y<= cetrtina-x:
        return 0
    elif y>= 3*cetrtina+x:
        return 0
    elif y<= -cetrtina + x:
        return 0
    elif x>=cetrtina and y<= 2*cetrtina and y>= x:
        return 0
    
    """
    if y<=cetrtina-x:
        return 0
    elif y>=3*cetrtina+x:
        return 0
    elif y>=5*cetrtina-x:
        return 0
    elif x>=cetrtina and y>=2*cetrtina and y<=velikost-x:
        return 0
    """
    return 1.0
def pogojipolkrog(u,velikost):
    x = u[0]+1
    y = u[1]+1
    """
    if y <= velikost/2-np.sqrt(velikost*velikost/4-(x-velikost/2)*(x-velikost/2)) or y>=velikost/2+np.sqrt(velikost*velikost/4-(x-velikost/2)*(x-velikost/2)):
        return 0.0
    """
    if y <= velikost/2-np.sqrt(velikost*velikost/4-(x-velikost/2)*(x-velikost/2)) or y>=velikost/2:
        return 0.0    
    return 1.0
def jacobi(u,h,q,macka=False):
    velikost = u[0].size
    star = np.copy(u)
    def pomozna(i,j):
        if macka=="polkrog":
            if pogojipolkrog((i,j),velikost)==0:
                return 0
        elif macka:
            if pogojimacka((i,j),velikost) == 0:
                return 0
        if i==0 or i==velikost-1 or j==0 or j==velikost-1:
            return 0.0
        return 1/4*(star[int(i+1)][int(j)]+star[int(i-1)][int(j)]+star[int(i)][int(j+1)]+star[int(i)][int(j-1)] - h*h*q[int(i)][int(j)])
    #return np.fromfunction(np.vectorize(pomozna),u.shape)
    for i in range(velikost):
        for j in range(velikost):
            u[i][j] = pomozna(i,j)
    return u
def jacobiChebyshev(u,h,q,omega,ro):
    velikost = u[0].size
    star = np.copy(u)
    aux = np.copy(u)
    def pomozna(i,j):
        if i==0 or i==velikost-1 or j==0 or j==velikost-1:
            return 0
        return 1/4*(star[int(i+1)][int(j)]+star[int(i-1)][int(j)]+star[int(i)][int(j+1)]+star[int(i)][int(j-1)] - h*h*q[int(i)][int(j)])
    for i in range(velikost):
        for j in range(velikost):
            if (i+j)%2!=0:
                continue
            aux[i][j] = pomozna(i,j)
    star = star+omega[0]*(aux-star)
    star2 = np.copy(star)
    for i in range(velikost):
        for j in range(velikost):
            if (i+j)%2==0:
                continue
            star2[i][j] = pomozna(i,j)    
    omega1 = 1/(1-ro*omega[-1]/4)
    omega2 = 1/(1-ro*omega1/4)
    return (star+omega[1]*(star2-star),(omega1,omega2))

def gaussSeidler(u,h,q,macka=False):
    velikost = u[0].size
    uu = np.copy(u)
    def pomozna(i,j):
        if macka=="polkrog":
            if pogojipolkrog((i,j),velikost)==0:
                return 0.0
        elif macka:
            if pogojimacka((i,j),velikost) == 0:
                return 0.0
        if i==0 or i==velikost-1 or j==0 or j==velikost-1:
            return 0.0
        return 0.25*(uu[i-1][j]+uu[i+1][j] + uu[i][j-1]+uu[i][j+1] - h*h*q[i][j])
    for i in range(velikost):
        for j in range(velikost):
            uu[i][j] = pomozna(i,j)
    return uu
    #return np.fromfunction(np.vectorize(pomozna),u.shape)
def jacobiSOR(u,h,q,omega):
    return u+omega*(jacobi(u,h,q)-u)
def gaussSeidlerSOR(u,h,q,omega,macka=False):
    velikost = u[0].size
    uu = np.copy(u)
    def pomozna(i,j):
        if macka=="polkrog":
            if pogojipolkrog((i,j),velikost)==0:
                return 0.0
        elif macka:
            if pogojimacka((i,j),velikost) == 0:
                return 0.0
        if i==0 or i==velikost-1 or j==0 or j==velikost-1:
            return 0.0
        return 0.25*(uu[i-1][j]+uu[i+1][j] + uu[i][j-1]+uu[i][j+1] - h*h*q[i][j])
    for i in range(velikost):
        for j in range(velikost):
            uu[i][j] = u[i][j]+omega*(pomozna(i,j)-u[i][j])
    return uu
def spektralniRadij(alfa,J):
    #ro^2 =1-alfa^2*pi^2/J^2
    #return np.cos(pi/20)
    return 1-alfa*alfa*pi*pi/(J*J)
def integriraj(u):
    x = np.linspace(0,1,u[0].size)
    y = np.linspace(0,1,u[:,0].size)
    I = np.zeros(u[:,0].size)
    for i in range(I.size):
        I[i] = np.trapz(u[i],x)
    return np.trapz(I,y)
    

def izracunaj(hx,hy,metoda,vrnicase=False,stiter=3000,macka=False,vrnicas=False,omegaa=0):
    iksi = np.linspace(0,1,int(1/hx))
    ipsiloni = np.linspace(0,1,int(1/hy))
    qji = np.array([[-1.0 for iks in iksi] for ips in ipsiloni])
    if macka=="polkrog":
        zacetna = np.asarray([[pogojipolkrog((iks,ips),ipsiloni.size) for iks in range(iksi.size)] for ips in range(ipsiloni.size)])
        ploscina = integriraj(zacetna)
    elif macka:
        zacetna = np.asarray([[pogojimacka((iks,ips),iksi.size) for iks in range(iksi.size)] for ips in range(ipsiloni.size)])
        ploscina = integriraj(zacetna)        
    else:
        zacetna = np.asarray([[1.0 for iks in iksi] for ips in ipsiloni])
        zacetna[0] = np.zeros(iksi.size)
        zacetna[-1]=np.zeros(iksi.size)
        zacetna[:,0] = np.zeros(iksi.size)
        zacetna[:,-1] = np.zeros(iksi.size)
    if vrnicase:
        start = timeit.default_timer()
        casi = []
        vrednosti = []
    if metoda=="jacobi":
        if vrnicase:
            for i in range(stiter):
                casi.append(timeit.default_timer()-start)
                vrednosti.append(8*pi*integriraj(zacetna))
                zacetna = jacobi(zacetna,hx,qji,macka)
            return (casi,vrednosti)                 
        for i in range(stiter):
            zacetna = jacobi(zacetna,hx,qji,macka)
    elif metoda=="gauss":
        if vrnicase:
            for i in range(stiter):
                casi.append(timeit.default_timer()-start)
                vrednosti.append(8*pi*integriraj(zacetna))
                zacetna = gaussSeidler(zacetna,hx,qji,macka)                
            return (casi,vrednosti)
        for i in range(stiter):
            zacetna = gaussSeidler(zacetna,hx,qji,macka)
    elif metoda=="gausssor":
        if omegaa!=0:
            omega = omegaa
        else:
            omega = 2/(1+pi/int(1/hx))            
        #if vrnicas:
            #start = timeit.default_timer()
        if vrnicase:
            for i in range(stiter):
                casi.append(timeit.default_timer()-start)
                vrednosti.append(8*pi*integriraj(zacetna))
                zacetna = gaussSeidlerSOR(zacetna,hx,qji,omega,macka)              
            return (casi,vrednosti)
        for i in range(stiter):
            zacetna = gaussSeidlerSOR(zacetna,hx,qji,omega,macka)
            #if vrnicas:
                #if np.abs(8*pi*integriraj(zacetna)-0.883271434893398)<0.1:
                    #return i
                    #return timeit.default_timer()-start
        if vrnicas:
            return np.abs(8*pi*integriraj(zacetna)/(ploscina**2)-0.34585015156658505)
            raise NameError("nisem prisel do take natancnosti")
    elif metoda=="chebyshev":
        ro = spektralniRadij(1,int(1/hx))
        omega = (1,1/(1-ro/2))
        if vrnicase:
            for i in range(stiter):
                casi.append(timeit.default_timer()-start)
                vrednosti.append(8*pi*integriraj(zacetna))
                temp = jacobiChebyshev(zacetna,hx,qji,omega,ro)
                zacetna = temp[0]
                omega=temp[1]                
            return (casi,vrednosti)
        for i in range(stiter):
            temp = jacobiChebyshev(zacetna,hx,qji,omega,ro)
            zacetna = temp[0]
            omega=temp[1]
    if macka:
        print("ja")
        print(8*pi*integriraj(zacetna)/(ploscina**2))
    else:
        print(8*pi*integriraj(zacetna))
    #fig, ax = plt.subplots()
    #cs = ax.contourf(iksi,iksi,zacetna,cmap=plt.get_cmap("hot"))
    #ax.set_aspect(1)
    velikost = zacetna[:,0].size
    cs = plt.imshow(np.matrix(zacetna).T,cmap=plt.get_cmap("hot"),aspect=1)
    plt.colorbar(cs)
    #plt.title("Pretok, 100x100")
    #plt.savefig("prva/profilmacka.pdf")
#izracunaj(0.01,0.01,"gausssor",macka=True)    
#praviC = 0.8832714348933981
#izracunaj(0.025,0.025,"gausssor",omegaa=1.85,stiter=100)
#izracunaj(0.01,0.01,"chebyshev")
#izracunaj(0.025,0.025,"chebyshev",False,3000)
if 0:
    h = 0.01
    rezult = np.array(izracunaj(h,h,"gausssor",True,3000))
    plt.plot(rezult[0],np.abs(praviC-rezult[1]),label="Gauss-Seidler, SOR")
    rezult = np.array(izracunaj(h,h,"gauss",True,3000))
    plt.plot(rezult[0],np.abs(praviC-rezult[1]),label="Gauss-Seidler")
    rezult = np.array(izracunaj(h,h,"jacobi",True,3000))
    plt.plot(rezult[0],np.abs(praviC-rezult[1]),label="Jacobi")
    rezult = np.array(izracunaj(h,h,"chebyshev",True,3000))
    plt.plot(rezult[0],np.abs(praviC-rezult[1]),label="Chebyshev")
    plt.title("Napaka C za kvadratno cev, 100x100")
    plt.ylabel("Abs. napaka")
    plt.xlabel("čas[s]")
    plt.legend(loc="best")
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig("prva/hitrosti100.pdf")
#izracunaj(0.05,0.05,"gausssor",macka="polkrog")
if 1:
    omege = np.linspace(1,1.9,100)
    casi = []
    for omega in omege:
        casi.append(np.abs(temperaturniprofil(0.025,1,1,omega)))
    minimum = omege[np.argmin(casi)]
    alfa = (2/minimum - 1)*40/pi #20 je J
    plt.axvline(minimum,color="red")
    plt.text(minimum,0.1,r"$\omega = {0}, \alpha = {1}$".format(str(round(minimum,2)),str(round(alfa,2))))
    plt.plot(omege,casi)
    plt.xlabel(r"$\omega$")
    plt.ylabel("Napaka")
    plt.yscale("log")
    plt.title(r"Napaka po 100 iteracijah, 100x100")
    plt.savefig("omege.pdf")

    
    
    
    
    
"""
suma = 0
for n in range(1,10000):
    suma+= np.tanh((2*n-1)*pi/2)/((2*n-1)**5)
    
print(2*pi*(1/3-64/(pi**5)*suma))
"""