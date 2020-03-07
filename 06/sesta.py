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

def gostote(i,j,N,gost): #gost je gostota sencenega x desno y dol
    x=i+1
    y=j+1
    pet=N/5
    #y = 3pet -x
    if y<= 3*pet - x:
        return gost
    elif y<=2*pet and y>= pet and x>=3*pet and x<=4*pet:
        return gost
    elif x<= 3*pet and y<= 2*pet and y>=4*pet-x: #y = 4pet-x
        return gost
    elif x<=pet and y>=3*pet+x and y<= 4*pet + x:
        return gost
    elif x>=pet and x<= 2*pet and y>=N-x and y<= 6*pet-x:
        return gost  
    elif x>=2*pet and x<=3*pet and y>=pet+x and y<= 2*pet+x:
        return gost
    elif x>=3*pet and x<=4*pet and y>=7*pet - x and y<= 8*pet - x:
        return gost
    return 1
def narediA2(h,N,fgt=0):
    """baza ij je (00)(10)(20)...(N0)(01)(02)..."""
    if fgt!=0:
        srednja = np.tile(np.array([-2*(1/(h*h)+1/((h*h*pi*(i+0.1))**2))-fgt for i in range(N)]),N)
    else:
        srednja = np.tile(np.array([-2*(1/(h*h)+1/((h*h*pi*(i+0.1))**2)) for i in range(N)]),N)
    temp = np.array([1/(h*h)+1/(2*(i+0.1)*h*h) for i in range(N-1)])
    temp1 = np.tile(np.append(temp,0),N-1)
    tempp = np.array([1/(h*h)-1/(2*(i+1.1)*h*h) for i in range(N-1)])
    temp2 = np.tile(np.append(tempp,0),N-1)    
    ob1 = np.concatenate((temp1,temp1[:-1]))
    ob2 = np.concatenate((temp2,temp2[:-1]))
    obob1 = np.tile([1/(((i+0.1)*h*h*pi)**2) for i in range(N)],N*N-N)
    obob2 = np.tile([1/(((i+0.1)*h*h*pi)**2) for i in range(N)],N*N-N)
    diagonals= [obob1,ob2,srednja,ob1,obob2]
    return scipy.sparse.diags(diagonals,[-(N),-1,0,1,(N)],shape=(N*N,N*N))    
def narediA(h,N,sparse=False,diagonalc=0):
    """Laplac matrika baza je (i,j) = (0,0),(0,1)...(0,N),(1,0),(1,1)..."""
    faktor = 1/(h*h)
    if sparse:
        if diagonalc!=0:
            srednja = np.ones(N*N)*(faktor*4-float(diagonalc))
        else:
            srednja = np.ones(N*N)*faktor*4
        temp = np.tile(np.append((np.ones(N-1)*(-faktor)),0),N-1)
        ob = np.concatenate((temp,np.ones(N-1)*(-faktor)))
        obob = np.ones(N*N-N)*(-faktor)
        diagonals= [obob,ob,srednja,ob,obob]
        return scipy.sparse.diags(diagonals,[-(N),-1,0,1,(N)],shape=(N*N,N*N))
    def pomozna(i,j):
        if i==j:
            return 4*faktor-diagonalc
        elif np.abs(i-j)==1:
            if i%N==0 and j%N!=0 and i>j:
                return 0
            elif j%N==0 and i%N != 0 and j>i:
                return 0
            else:                  
                return -1*faktor
        elif np.abs(i-j)==N: #N je st. tock
            return -1*faktor
        else:
            return 0
    return np.fromfunction(np.vectorize(pomozna),(N*N,N*N))
def inverzna2(h,N):
    if 1:
        vektor = np.matrix(np.ones(N*N)).T
        AA = narediA2(h,N)
        A = scipy.sparse.linalg.inv(narediA2(h,N,0).tocsc())
        for i in range(20):
            vektor = A*vektor
            vektor = vektor/lin.norm(vektor)
            #vrednosti.append((AA*vektor)[0]/vektor[0])
        vrednost = (AA*vektor)[0]/vektor[0]
        print(vrednost)
        return (vrednost,vektor)
#matrika = narediA(1/50,50,True)
#lastne = scipy.sparse.linalg.eigs(matrika,which="SM")[1][:,0]
#x = np.linspace(0,1,50)
#y = np.linspace(0,1,50)
#x,y = np.meshgrid(x,y)
if 1:
    vektor = inverzna2(1/70,70)
    rji = np.linspace(0,1,70)
    fiji = np.linspace(0,pi,70)
    xx,yy = np.meshgrid(rji,fiji)
    x, y = xx*np.cos(yy),xx*np.sin(yy)
    plt.title(r"$\omega^2 = 70.22$")
    #plt.contourf(x,y,np.reshape(lastne,(50,50)),cmap=plt.get_cmap("hot"))
    plt.contourf(x,y,np.real(np.reshape(vektor[1].flatten(),(70,70))))
    #plt.savefig("3.pdf")
if 0:
    #rji = np.linspace(0,1,100)
    #fiji = np.linspace(0,pi,100)
    #xx,yy = np.meshgrid(rji,fiji)
    #x, y = xx*np.cos(yy),xx*np.sin(yy)
    matrika = narediA2(1/60,60)
    #print("tukaj")
    lastne = scipy.sparse.linalg.eigs(matrika.tocsc(),which="SM")
    #vrednost = lastne[0][0]
    vektor = lastne[1][:,0]
    #plt.contourf(x,y,np.reshape(lastne,(50,50)),cmap=plt.get_cmap("hot"))
    #plt.contourf(x,y,np.real(np.reshape(vektor,(100,100))))
    plt.imshow(np.real(np.reshape(vektor,(60,60))))
def inverznaIteracija(h,N,sparse=False,stiter=10):
    start = timeit.default_timer()
    """inverzna potencna + rayleigh"""
    if sparse:
        vrednosti = []
        vektor = np.matrix(np.ones(N*N)).T
        A = narediA(h,N,sparse)
        vrednost = 1
        for i in range(stiter):
            matrika = scipy.sparse.linalg.inv(narediA(h,N,sparse,vrednost).tocsc())
            vektor = matrika*vektor
            vektor = vektor/lin.norm(vektor)
            #vrednost = vektor.T * A * vektor
            #vrednosti.append(vrednost)
        vrednost = (A*vektor)[0]/vektor[0]
        return timeit.default_timer()-start
        return vrednosti
        return (vrednost,vektor)
    else:
        vrednosti = []
        vektor = np.matrix(np.ones(N*N)).T
        A = narediA(h,N,sparse)
        vrednost = 1
        for i in range(stiter):
            matrika = lin.inv(narediA(h,N,sparse,vrednost))
            vektor = matrika*vektor
            vektor = vektor/lin.norm(vektor)
            #vrednost = vektor.T * A * vektor
            #vrednosti.append(vrednost)
        vrednost = (A*vektor)[0]/vektor[0]
        return timeit.default_timer()-start
        return vrednosti
        return (vrednost,vektor)        
def inverznaIteracijaBrez(h,N,sparse=False,stiter=10):
    """inverzna potencna"""
    start = timeit.default_timer()
    if sparse:       
        vrednosti = []
        vektor = np.matrix(np.ones(N*N)).T
        AA = narediA(h,N,sparse)
        A = scipy.sparse.linalg.inv(narediA(h,N,sparse,1).tocsc())
        for i in range(stiter):
            vektor = A*vektor
            vektor = vektor/lin.norm(vektor)
            #vrednosti.append((AA*vektor)[0]/vektor[0])
        vrednost = (AA*vektor)[0]/vektor[0]
        return timeit.default_timer()-start
        return vrednosti
        vrednost = (AA*vektor)[0]/vektor[0]
        return (vrednost,vektor)
    else:
        vrednosti = []
        vektor = np.matrix(np.ones(N*N)).T
        AA = narediA(h,N,sparse)
        A = lin.inv(narediA(h,N,sparse,1))
        for i in range(stiter):
            vektor = A*vektor
            vektor = vektor/lin.norm(vektor)
            #vrednosti.append((AA*vektor)[0]/vektor[0])
        vrednost = (AA*vektor)[0]/vektor[0]
        return timeit.default_timer()-start
        return vrednosti
        vrednost = (AA*vektor)[0]/vektor[0]
        return (vrednost,vektor)
"""
gostotke = np.linspace(0.1,10,100)
frekve = []
for g in gostotke:
    MM = np.array([[gostote(i,j,80,g) for i in range(80)] for j in range(80)])
    M =scipy.sparse.diags(MM.flatten(),0)
    matrika = scipy.sparse.linalg.eigs(narediA(1/80,80,True).tocsc(),M=M.tocsc(),which="SM",k=10)
    frekve.append(matrika[0][0])
plt.plot(gostotke,frekve)
plt.title("Odvisnost najnižje frekvence od gostote belega dela")
plt.xlabel(r"$\rho$")
plt.ylabel(r"$\omega^2$")
plt.savefig("prva/nehomoodvisnost.pdf")
"""
if 0:
    if 0:
        #lambda od N za diag
        lambde = np.ones((3,20))
        nji = list(range(10,201,10))
        for i in range(20):
            lambde[0][i] = scipy.sparse.linalg.eigsh(narediA(1/nji[i],nji[i],True),which="SM")[0][0]
            lambde[1][i] = scipy.sparse.linalg.eigsh(narediA(1/nji[i],nji[i],True),which="SM")[0][1]
            lambde[2][i] = scipy.sparse.linalg.eigsh(narediA(1/nji[i],nji[i],True),which="SM")[0][3]
        plt.title("Lastne vrednosti pridobljene z direktno diagonalizacijo")
        plt.xlabel("N")
        plt.ylabel(r"$\omega^2$")
        plt.plot(nji,lambde[0],"r")
        plt.axhline(2*pi**2,color="r",ls="--")
        plt.plot(nji,lambde[1],"b")
        plt.axhline(5*pi**2,color="b",ls="--")
        plt.plot(nji,lambde[2],"g")
        plt.axhline(8*pi**2,color="g",ls="--")
        plt.savefig("prva/diag.pdf")        
    #hitrost v odv od N
    if 0:
        pravavrednost = 2*pi*pi
        plt.title("Napaka najmanjse frekvence pri inverzni potencni metodi, N=50")
        plt.xlabel("St iteracij")
        plt.ylabel("Abs. napaka")
        vrednosti = inverznaIteracijaBrez(1/50,50,True)
        plt.plot(range(1,len(vrednosti)+1),np.abs(np.array(vrednosti).flatten()-pravavrednost),label="Inverzna iteracija")
        vrednosti = inverznaIteracija(1/50,50,True)
        plt.plot(range(1,len(vrednosti)+1),np.abs(np.array(vrednosti).flatten()-pravavrednost),label="Rayleighova iteracija")
        plt.yscale("log")
        plt.legend(loc="best")
        plt.savefig("prva/potencna1.pdf")    
        #Odvisnost lambde od st iteracij pri potencni metodi pri fiksnem N
    if 0:
        plt.title("Hitrost razlicnih potencnih metod (5 iteracij)")
        plt.xlabel("N")
        plt.ylabel("t[s]")
        nji = [5,10,15,20,25,30]
        casi = []
        for n in nji:
            casi.append(inverznaIteracijaBrez(1/n,n,False,5))
        print("skozi")
        plt.plot(nji,casi,"-.",label="Navadna potencna")
        casi = []
        for n in nji:
            casi.append(inverznaIteracijaBrez(1/n,n,True,5))
        plt.plot(nji,casi,"-.",label="Sparse potencna")
        print("skozi")
        casi = []
        for n in nji:
            casi.append(inverznaIteracija(1/n,n,False,5))
        plt.plot(nji,casi,"-.",label="Navaden Rayleigh")
        casi = []
        print("skozi")
        for n in nji:
            casi.append(inverznaIteracija(1/n,n,True,5))
        print("skozi")

        plt.plot(nji,casi,"-.",label="Sparse Rayleigh")
        plt.legend(loc="best")
        plt.savefig("prva/potencna2.pdf")

    if 0:
        nji = list(range(5,76,5))
        print(len(nji))
        casi = np.ones((2,15))
        for n in range(len(nji)):
            start = timeit.default_timer()
            inverznaIteracija(1/nji[n],nji[n],True,stiter=2)
            casi[0][n] = timeit.default_timer()-start
            start = timeit.default_timer()
            inverznaIteracijaBrez(1/nji[n],nji[n],True,stiter=6)
            casi[1][n] = timeit.default_timer()-start
        plt.xlabel("N")
        plt.ylabel("t[s]")
        plt.title("Primerjava rayleighove in inverzne iteracije")
        plt.plot(nji,casi[0],label="Inverzna")
        plt.plot(nji,casi[1],label="Rayleigh")
        plt.legend(loc="best")
        #plt.yscale("log")
        plt.savefig("prva/potencna3.pdf")
    if 0:
        nji = list(range(5,51,5))
        print(len(nji))
        casi = np.ones((4,10))
        for n in range(len(nji)):
            start = timeit.default_timer()
            lin.eig(narediA(1/nji[n],nji[n]))
            casi[0][n] = timeit.default_timer()-start
            start = timeit.default_timer()
            lin.eigh(narediA(1/nji[n],nji[n]))
            casi[1][n] = timeit.default_timer()-start
            start = timeit.default_timer()
            scipy.sparse.linalg.eigs(narediA(1/nji[n],nji[n],True))
            casi[2][n] = timeit.default_timer()-start
            start = timeit.default_timer()
            scipy.sparse.linalg.eigsh(narediA(1/nji[n],nji[n],True))
            casi[3][n] = timeit.default_timer()-start
        plt.xlabel("N")
        plt.ylabel("t[s]")
        plt.title("Čas za diagonalizacijo matrik")
        plt.plot(nji,casi[0],label="Navadni eig")
        plt.plot(nji,casi[1],label="Navadni eigh")
        plt.plot(nji,casi[2],label="Sparse eig")
        plt.plot(nji,casi[3],label="Sparse eigh")
        plt.legend(loc="best")
        plt.yscale("log")
        plt.savefig("prva/diag2.pdf")
    #hitro od N za diag
if 0:      
    rezultat = inverznaIteracija(1/50,50,True)
    print(rezultat[0])
    plt.imshow(np.reshape(rezultat[1],(50,50)))
if 0:
    matrika = narediA(1/100,100,True)
    eigens = scipy.sparse.linalg.eigsh(matrika,which="SM")
    print(eigens[1])
    print(2*pi**2)
    plt.imshow(np.reshape(eigens[1][:,1],(100,100)),cmap=plt.get_cmap("hot"))
    
    



