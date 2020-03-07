# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 21:35:07 2018

@author: Admin
"""
import time
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
from scipy import fftpack
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
plt.rc("text",usetex=True)
matplotlib.rcParams["text.latex.unicode"] = True
plt.close("all")
pi = np.pi
Whole = 2*pi
def getForces(Phi0,Phi1,DeltaS,DeltaT,F=[0]):
    #return getForcesSOR(Phi0,Phi1,DeltaS,DeltaT,F)
    N = Phi0.size
    try:
        Diagonal = -2-(np.roll(Phi1,-1) - np.roll(Phi1,1))**2/4
        Diagonal[0] = 1
        Diagonal[-1] = 1
        UpDiagonal = np.ones(N-1)
        UpDiagonal[0] = -1
        LowDiagonal = np.ones(N-1)
        LowDiagonal[-1] = 0
        RightSide = -DeltaS*DeltaS/(DeltaT*DeltaT) *(Phi1-Phi0)**2
        RightSide[0] = DeltaS * np.sin(Phi1[0])
        RightSide[-1] = 0
    except:
        print(Diagonal)
        print(UpDiagonal)
        print(LowDiagonal)
        print(RightSide)
        print(1/0)
    #return Thomas(LowDiagonal,Diagonal,UpDiagonal,RightSide)
    TheMatrix = scipy.sparse.diags([Diagonal,UpDiagonal,LowDiagonal],[0,1,-1],(N,N),"csc")
    #return scipy.sparse.linalg.lsqr(TheMatrix,RightSide)[0]
    return scipy.sparse.linalg.spsolve(TheMatrix,RightSide)
    

def Thomas(LowDiagonal,Diagonal,UpDiagonal,RightSide):
    n = Diagonal.size
    c = np.ones(n-1)*1.0
    if 1:
        c[0] = UpDiagonal[0]/Diagonal[0]
        for i in range(1,n-1):
            c[i] = UpDiagonal[i]/(Diagonal[i]-LowDiagonal[i-1]*c[i-1])
            d = np.ones(n)*1.0
        d[0] = RightSide[0]/Diagonal[0]
        for i in range(1,n):
            d[i] = (RightSide[i] - LowDiagonal[i-1] * d[i-1])/(Diagonal[i]-LowDiagonal[i-1]*c[i-1])
            Solution = np.ones(n)*1.0
        Solution[-1] = d[-1]
        for i in range(n-2,-1,-1):
            Solution[i] = d[i]-c[i]*Solution[i+1]

    return Solution
def getForcesSOR(Phi0,Phi1,DeltaS,DeltaT,F):
    if len(F)==1:
        F = np.ones(Phi0.size)
    u = np.copy(F)
    stara = np.zeros(u.size)
    counter=0
    while np.sum(np.abs(stara-u))>0.01 and counter<5000:
        counter+=1
        stara = u
        u = getForcesSORAux(Phi0,Phi1,DeltaS,DeltaT,u)
    return u
    
def getForcesSORAux(Phi0,Phi1,DeltaS,DeltaT,u):
    velikost = u.size
    uu = np.copy(u)
    def pomozna(i):
        if i==0:
            return uu[1]+DeltaS*np.sin(Phi1[0])
        elif i==velikost-1:
            return 0
        ei = -2-(Phi1[i+1]-Phi1[i-1])**2/4
        fi = - DeltaS*DeltaS/(DeltaT*DeltaT)*(Phi1[i]-Phi0[i])**2 
        return 1/ei*(fi-uu[i-1]-uu[i+1])
    for i in range(velikost):
            uu[i] = u[i]+1.5*(pomozna(i)-u[i])    
    return uu
def getAngles(Phi0,Phi1,Forces,DeltaT,DeltaS):
    Aux1 = 2*Phi1 - Phi0
    Aux2 = (np.roll(Forces,-1)-np.roll(Forces,1))*(np.roll(Phi1,-1)-np.roll(Phi1,1))/2 + Forces*(np.roll(Phi1,-1)-2*Phi1 + np.roll(Phi1,1))
    Result = Aux1 + (DeltaT*DeltaT)/(DeltaS*DeltaS) * Aux2
    Result[0] = Result[2] + 2*DeltaS/Forces[1] * np.cos(Result[1])
    Result[-1] = 2*Result[-2] - Result[-3]
    return Result
    
def solver(Phi0,EndTime,DeltaT):
    N = Phi0.size
    M = int(EndTime//DeltaT)
    DeltaS = 1/N
    Time=np.linspace(0,EndTime,M)
    Forces = np.ones((M,N))
    Angles = np.ones((M,N))
    for i in range(M):
        if i==0:
            Angles[i] = Phi0
            Forces[i] = getForces(Phi0,Phi0,DeltaS,DeltaT)
            continue
        if i==1:
            Forces[i] = Forces[0]
            Angles[i] = getAngles(Phi0,Phi0,Forces[0],DeltaT,DeltaS)
            continue
        #Angles[i-1] = Angles[i-1] % (2*pi)
        Forces[i] = getForces(Angles[i-2],Angles[i-1],DeltaS,DeltaT)
        Angles[i] = getAngles(Angles[i-2],Angles[i-1],Forces[i],DeltaT,DeltaS)
    return(Time,Angles,Forces)

def toXY(Angles):
    N = Angles.size
    DeltaS = 1/N
    Xs = np.ones(N)
    Ys = np.ones(N)
    Xs[0]=0
    Ys[0]=0
    for i in range(1,N):
        try:
            Xs[i] = DeltaS*np.sum(np.cos(Angles[:i])) + 0.5*DeltaS*np.cos(Angles[i])
            Ys[i] = DeltaS*np.sum(np.sin(Angles[:i])) + 0.5*DeltaS*np.sin(Angles[i])
        except:
            print(Angles[:i])
            print(Angles[i])
            print(1/0)
    return (Xs,-Ys)
def energies(Angles0,Angles1,DeltaT,hitrosti=False):
    if hitrosti:
        X1, Y1 = toXY(Angles1)
        X0, Y0 = toXY(Angles0)       
        return ((X1-X0)**2+(Y1-Y0)**2)/(DeltaT*DeltaT)
    N = Angles0.size
    DeltaS=1/N
    X1, Y1 = toXY(Angles1)
    X0, Y0 = toXY(Angles0)
    Kinetic = 0.5*DeltaS/(DeltaT*DeltaT)*((X1-X0)**2+(Y1-Y0)**2) + 1/24*DeltaS*DeltaS/(DeltaT*DeltaT)*(Angles1-Angles0)**2
    Potential = Y1*DeltaS 
    return (np.sum(Kinetic+Potential),np.sum(Kinetic),np.sum(Potential))

if 0:
    N = 100
    zacetni = np.ones(100)*pi/3
    #zacetni = np.linspace(20,90,100)*pi/180
    #zacetni = np.linspace(20,160,100)*pi/180
    #X,Y = toXY(zacetni)    #plt.plot(X,Y)
    #zacetni = np.linspace(20,160,100)*pi/180
    rezultat = solver(zacetni,1.5,0.001)
    j = np.linspace(0,1,int(len(rezultat[1])//100)+1)
    counter = 0
    barve = plt.get_cmap("Wistia")
    
    for i in range(0,len(rezultat[1]),100):
        X,Y = toXY(rezultat[1][i])
        if i==0:
            plt.plot(X,Y,color=barve(j[counter]),label=r"$t=0$")
            counter+=1
            continue            
        elif i>len(rezultat[1])-150:
            plt.plot(X,Y,color=barve(j[counter]),label=r"$t=2.5$")                        
            counter+=1
            continue
        plt.plot(X,Y,color=barve(j[counter]))
        counter+=1
    plt.title(r"$N=100, \Delta t = 0.001, \Delta s = 0.01, \delta t = 0.1$")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axhline(y=0,color="k")
    plt.legend(loc="best")
    #plt.savefig("nihanje3.pdf")
    
if 0:
    N = 100
    zacetni = np.ones(100)*pi/3
    #zacetni = np.linspace(20,90,100)*pi/180
    #zacetni = np.linspace(20,160,100)*pi/180
    #X,Y = toXY(zacetni)
    #plt.plot(X,Y)
    #zacetni = np.linspace(20,160,100)*pi/180
    rezultat = solver(zacetni,2.5,0.001)
    plt.title(r"$N=100, \Delta t = 0.001, \Delta s = 0.01, \delta t = 0.1$")
    plt.xlabel("t")
    plt.ylabel("Energija")
    Energije = np.array([energies(rezultat[1][i-1],rezultat[1][i],0.001) for i in range(1,len(rezultat[1]))])
    plt.plot(rezultat[0][2:]-0.0005,Energije[:,0][1:],label=r"$E_k + E_p$")
    plt.plot(rezultat[0][2:]-0.0005,Energije[:,1][1:],label=r"$E_k$")
    plt.plot(rezultat[0][2:]-0.0005,Energije[:,2][1:],label=r"$E_p$")

    #plt.axhline(y=0,color="k")
    plt.legend(loc="best")
    plt.savefig("energija1.pdf")
if 0:
    N = 1000
    #zacetni = np.ones(N)*pi/3
    zacetni = np.linspace(20,90,N)*pi/180
    #zacetni = np.linspace(20,160,N)*pi/180
    rezultat = solver(zacetni,1.5,0.0005)
    referenca = energies(rezultat[1][-2],rezultat[1][-1],0.0005)[0]
    energije = []
    for n in np.linspace(30,500,60):
        print(n)
        zacetni = np.linspace(20,90,int(n))*pi/180
        #zacetni = np.linspace(20,160,int(n))*pi/180
        #zacetni = np.ones(int(n))*pi/3
        rezultat = solver(zacetni,1.5,0.0005)
        energije.append(energies(rezultat[1][-2],rezultat[1][-1],0.0005)[0])
    energije = np.array(energije)
    plt.plot(np.linspace(30,500,60),energije-referenca)
    plt.xlim(30,500)
    plt.xticks([30,100,200,300,400,500])
    plt.xlabel("N")
    plt.ylabel(r"$E(N) - E_{ref}$")
    plt.title(r"$\Delta t = 0.0005, \Delta s = 1/N, N(E_{ref})=1000$")
    plt.savefig("napake22.pdf")
if 0:
    fig = plt.figure(figsize=(20,20))
    ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=2) #glavni Ax
    ax2 = plt.subplot2grid((2, 2), (0, 1)) 
    ax3 = plt.subplot2grid((2, 2), (1, 1))
    #fig, (ax1,ax2,ax3)=plt.subplots(3,figsize=(20,20))
    N = 100
    zacetni = np.ones(100)*pi/3
    print("tukaj")
    r = solver(zacetni,15,0.0005)
    rezultat = r[1][::100]
    sile = r[2][::100]
    hitrosti = [energies(rezultat[t-1],rezultat[t],0.0005,True) for t in range(1,len(rezultat))]
    print("tukaj2")    
    def animiraj5(t):
        global ax1
        global ax2
        global ax3
        print(t)
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax1.set_xlim(-0.8,0.8)
        ax1.set_ylim(-1,0)
        ax1.axhline(y=0,color="k")
        X, Y = toXY(rezultat[t])
        ax1.plot(X,Y)
        #ax1.plot(np.linspace(-0.8,0.8,100),np.linspace(-0.8,0.8,100)*t)
        plt.suptitle(r"$t={}$".format(str(round(t*0.05,2))))
        ax2.set_title("F(s)")
        ax2.plot(np.linspace(0,1,100),sile[t])
        ax3.set_title(r"$v^2(s)$")
        if t>0:
            ax3.plot(np.linspace(0,1,100),hitrosti[t-1])
    ani = animation.FuncAnimation(fig,animiraj5,frames=300,interval=100,repeat=False)   
    #plt.show()
    ani.save("vecinfo.mp4")
if 0:
    fig, ax1 = plt.subplots()
    N = 100
    zacetni3 = np.ones(100)*pi/10
    zacetni2 = np.linspace(20,90,100)*pi/180
    zacetni = np.linspace(20,160,100)*pi/180
    print("tukaj")
    #zacetni = np.linspace(20,160,100)*pi/180
    rezultat = solver(zacetni,15,0.0005)[1][::100]
    print(len(rezultat))
    rezultat2 = solver(zacetni2,15,0.0005)[1][::100]
    rezultat3 = solver(zacetni3,15,0.0005)[1][::100]
    indeks1 = 0
    indeks2 = 0
    indeks3 = 0
    print("tukaj2")    
    def animiraj3(t):
        global indeks1
        global indeks2
        global indeks3
        ax1.clear()
        ax1.set_xlim(-0.8,0.8)
        ax1.set_ylim(-1,0)
        ax1.axhline(y=0,color="k")
        if t<60:
            X, Y = toXY(rezultat[indeks1])
            ax1.plot(X,Y,color="b")
            ax1.set_title(r"$t={}$".format(str(round(indeks1*0.05,2))))
            indeks1 +=1
        elif t<=200:
            X, Y = toXY(rezultat2[indeks2])
            plt.plot(X,Y,color="r")
            ax1.set_title(r"$t={}$".format(str(round(indeks2*0.05,2))))
            indeks2 +=1
        elif t<=500:
            X, Y = toXY(rezultat3[indeks3])
            plt.plot(X,Y,color="g")
            ax1.set_title(r"$t={}$".format(str(round(indeks3*0.05,2))))
            indeks3 +=1
    ani = animation.FuncAnimation(fig,animiraj3,range(0,501),interval=50)   
    #plt.show()
    ani.save("zjebane.mp4")
    

if 1:
    fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3,sharey="row",sharex="col",figsize=(30,30))
    aksi = [ax1,ax2,ax3,ax4,ax5,ax6]
    zacetni = [np.linspace(n,180-n,20)*pi/180 for n in [20,30,40,50,60,70]]
    #zacetni = [np.ones(20)*pi/n for n in [3,4,5,6,7,8]]
    print("tukaj")
    #zacetni = np.linspace(20,160,100)*pi/180
    #rezultat = solver(zacetni,25,0.0005)[1][::100]
    rezultati = [solver(zacetni[i],10,0.0005)[1][::100] for i in range(6)]
    print("tukaj2")    
    def animiraj2(t):
        print(t)
        global zjebano
        for i in range(6):
            aksi[i].clear()
            aksi[i].set_xlim(-0.8,0.8)
            aksi[i].set_ylim(-1,0)
            aksi[i].axhline(y=0,color="k")
            X, Y = toXY(rezultati[i][t])
            aksi[i].plot(X,Y)
        plt.suptitle(r"$t={}$".format(str(round(t*0.05,2))))

    ani = animation.FuncAnimation(fig,animiraj2,range(0,200),interval=50)   
    #plt.show()
    ani.save("ukrivljenstart.mp4")