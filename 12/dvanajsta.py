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
BoundVel = np.array([1,0,0,0])
def solverSOR(N,Initial,Q,Omega = 1, Eps=0.01,MaxIter=5000):
    h = 1/N
    def aux(u): #Next approximation
        uu = np.copy(u)
        def auxAux(i,j): 
            if i==0:
                return 0.0
            if i==0 or j==0 or i==N-1 or j==N-1:
                return 0.0
            return 0.25*(uu[i-1][j]+uu[i+1][j] + uu[i][j-1]+uu[i][j+1] - h*h*Q[i][j])
        for i in range(N):
            for j in range(N):
                uu[i][j] = u[i][j]+Omega*(auxAux(i,j)-u[i][j])    
        return uu
    #u = np.zeros((N,N))
    u = np.copy(Initial)
    Old = np.ones((N,N))*100
    counter=0
    while np.sum(np.abs(Old-u))>Eps and counter<MaxIter:
        counter+=1
        Old = u
        u = aux(u)
    return u
def solverSparse(N,Q):
    h = 1/N
    a = h*h
    Diagonal = np.ones(N*N)*(-4.0)
    ND = np.ones(N*N)
    ND2 = np.ones(N*N)
    RightSide = Q.flatten()*a
    Matrix = scipy.sparse.spdiags([Diagonal,ND,ND,ND2,ND2],[0,1,-1,N,-N],N*N,N*N)
    return np.reshape(scipy.sparse.linalg.spsolve(Matrix,RightSide),(N,N))
def timeEvolution(Psi, Zeta, U,V,DeltaT,DeltaX,Re,BoundVel):
    New =  Zeta-0.5*DeltaT/DeltaX*(np.roll(U,-1,0)*np.roll(Zeta,-1,0) - np.roll(U,1,0)*np.roll(Zeta,1,0) - np.roll(V,1,1)*np.roll(Zeta,1,1) + np.roll(V,-1,1)*np.roll(Zeta,-1,1))
    New = New + DeltaT/(Re*DeltaX*DeltaX)*(np.roll(Zeta,1,1)+np.roll(Zeta,-1,1)+np.roll(Zeta,1,0)+np.roll(Zeta,-1,0) - 4*Zeta)
    New[0] = -2/(DeltaX*DeltaX)*(Psi[1]-BoundVel[2]*DeltaX) #spodaj -> levo
    New[-1] = -2/(DeltaX*DeltaX)*(Psi[-2]-BoundVel[3]*DeltaX) #zgoraj -> desno
    New[:,0] = -2/(DeltaX*DeltaX)*(Psi[:,1] - BoundVel[0]*DeltaX) #levo  ->zgoraj
    New[:,-1] = -2/(DeltaX*DeltaX)*(Psi[:,-2] - BoundVel[1]*DeltaX) #desno -> dol
    return New
def getForce(u0,u1,h,re):
    Derivative = (u1-u0)/h
    N = len(Derivative)
    return 1/re*scipy.integrate.trapz(np.linspace(0,1,N),Derivative)        
def solver(N,Re,DeltaT,EndTime,Konc=0.1,Frek=1):
    global BoundVel
    BoundVel=np.array([1.0,0.0,0.0,0.0])
    DeltaX = 1/N
    Psi = np.zeros((N,N))
    #Psi[1] = 2*DeltaX
    U = np.zeros((N,N))
    U[0] = np.ones(N)*BoundVel[2]
    U[-1] = np.ones(N)*BoundVel[3]
    V = np.zeros((N,N))
    V[:,0] = np.ones(N)*BoundVel[0]
    V[:,-1] = np.ones(N)*BoundVel[1]
    #Zeta = xiupdate(np.zeros((N,N)),DeltaX,Psi,U,V,DeltaT,Re)
    Zeta = timeEvolution(Psi,np.zeros((N,N)),U,V,DeltaT,DeltaX,Re,BoundVel)
    #Meja = 0
    #Zeta = 1/(2*DeltaX)*(np.roll(Us,-1,0) - np.roll(Us,1,0)) 
    #Zeta = np.zeros((N,N))
    #Zeta[0] = -BoundVel[0]/(DeltaX)
    #Zeta[-1] = 0
    #Zeta[:,0] = 0
    #Zeta[:,-1] = 0
    #ZetaOld = np.ones((N,N))*100
    #while np.sum(np.abs(Zeta-ZetaOld))>1:
    Time = 0
    Psiji = [Psi]
    Uji = [U]
    Vji = [V]
    Zete = [Zeta]
    casi = [0]
    Eps = 0.01
    counter=0
    Hitrosti = [BoundVel[0]]
    #NotStopped = True
    while Time<EndTime:
        if Time>Konc:
            #NotStopped=False
            #Meja = np.abs(np.sum(Zeta)*0.01)
            BoundVel[0]=np.cos(Frek*(Time-Konc))**2
        #Psi = psiupdate(Psi,Zeta,1/N)
        #Psi = solverSparse(N,-Zeta)
        Psi = solverSOR(N,Psi,-Zeta,Eps=Eps)
        #Psi = solverSparse(N,Zeta)
        U = (np.roll(Psi,-1,1) - np.roll(Psi,1,1))/(2*DeltaX)
        U[0] = np.ones(N)*BoundVel[2]
        U[-1] = np.ones(N)*BoundVel[3]
        U[:,0] = np.zeros(N)
        U[:,-1] = np.zeros(N)
        V = -(np.roll(Psi,-1,0)-np.roll(Psi,1,0))/(2*DeltaX)
        V[0] = np.zeros(N)
        V[-1] = np.zeros(N)
        V[:,0] = np.ones(N)*BoundVel[0]
        V[:,-1] = np.ones(N)*BoundVel[1]
        if counter==50:
            Uji.append(U)
            Vji.append(V)
            Psiji.append(Psi)
            Zete.append(Zeta)
            Hitrosti.append(BoundVel[0])
            casi.append(Time)
            counter = 0
            print(Time)
        #Uji.append(U)
        #Vji.append(V)
        #Psiji.append(Psi)
        #ZetaOld = Zeta
        #Zeta = xiupdate(Zeta,DeltaX,Psi,U,V,DeltaT,Re)
        Zeta = timeEvolution(Psi,Zeta,U,V,DeltaT,DeltaX,Re,BoundVel)
        #if np.abs(np.sum(Zeta))<Meja:
            #return Time
        #if np.sum(np.abs(ZetaOld-Zeta))<1:
            #return Time
            #return getForce(V[:,0],V[:,1],DeltaX,Re)
        Time+=DeltaT
        #print(Time)
        counter+=1
    print("kva")
    return (Psiji,Uji,Vji,Zete,Hitrosti,casi)
if 0:
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    N = 50
    rez = solver(N,50000,0.001,10,0.1,2)
    #rez = solver(N,50,0.001,10,0.1,2)
    casi = np.array(rez[-1])
    ax1.plot(casi,[np.abs(np.sum(i)) for i in rez[3]],"r",label="Vrtinčnost")
    ax2.plot(casi,np.cos(2*(casi-0.1))**2,"b",label="Hitrost roba")
    fig.legend(loc=(0.7,0.8))
    plt.title(r"$Re=50000, \omega = 2$")
    plt.savefig("osc7.pdf")
    
    ax1.clear()
    ax2.clear()
    rez = solver(N,5000,0.001,10,0.1,2)
    #rez = solver(N,50,0.001,10,0.1,2)
    casi = np.array(rez[-1])
    ax1.plot(casi,[np.abs(np.sum(i)) for i in rez[3]],"r",label="Vrtinčnost")
    ax2.plot(casi,np.cos(2*(casi-0.1))**2,"b",label="Hitrost roba")
    fig.legend(loc=(0.7,0.8))
    plt.title(r"$Re=5000, \omega = 2$")
    plt.savefig("osc6.pdf")
    #plt.yscale("log")
    
if 0:
    fig, ax = plt.subplots()
    N=50
    Re = np.linspace(50,5000,30)
    casi = [[],[],[]]
    counter = 0
    for r in Re:
        counter+=1
        print(counter)
        Rez = solver(N,r,0.001,100,0.1)
        casi[0].append(Rez)
        Rez = solver(N,r,0.001,100,0.2)
        casi[1].append(Rez)
        Rez = solver(N,r,0.001,100,0.5)
        casi[2].append(Rez)
    ax.set_title("Relaksacijski časi")
    ax.set_xlabel("Re")
    ax.set_ylabel("t")
    ax.plot(Re,casi[0],label=r"$t_0=0.1$")
    ax.plot(Re,casi[1],label=r"$t_0=0.2$")
    ax.plot(Re,casi[2],label=r"$t_0=0.5$")
    plt.legend(loc="best")
    plt.savefig("RelakCasi.pdf")
    ax.clear()
    ax.set_title("Relaksacijski časi - log skala")
    ax.set_xlabel("Re")
    ax.set_ylabel("t")                    
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(Re,casi[0],label=r"$t_0=0.1$")
    ax.plot(Re,casi[1],label=r"$t_0=0.2$")
    ax.plot(Re,casi[2],label=r"$t_0=0.5$")
    plt.legend(loc="best")
    plt.savefig("RelakCasiLog.pdf")
if 0:
    fig, ax = plt.subplots()
    N=50
    Re = np.linspace(50,5000,100)
    casi = []
    Rez = [np.sum(i) for i in solver(N,5000,0.001,25)[3]]
    casi = np.linspace(0,25,len(Rez))
    if 0:
        counter = 0
        for r in Re:
            counter+=1
            print(counter)
            Rez = solver(N,r,0.001,100)
            casi.append(Rez)
    ax.set_title(r"Re = 5000, relaksacija po 0.1s vlecenja")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\sum \zeta$")
    ax.plot(casi,np.abs(Rez))
    #plt.savefig("Rel.pdf")
    plt.show()
    ax.clear()
    ax.set_title(r"Re = 5000, relaksacija po 0.1s vlecenja")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\sum \zeta$")                    
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(casi,np.abs(Rez))
    plt.show()
    #plt.savefig("RelLog.pdf")
if 0:
    fig, ax = plt.subplots()
    N=50
    Re = np.linspace(50,5000,100)
    casi = []
    counter = 0
    for r in Re:
        counter+=1
        print(counter)
        Rez = solver(N,r,0.001,100)
        casi.append(Rez)
    ax.set_title(r"Strižna sila na spodnji rob")
    ax.set_xlabel("Re")
    ax.set_ylabel("F")
    ax.plot(Re,casi)
    plt.savefig("Sile.pdf")
    ax.clear()
    ax.set_title(r"Strižna sila na spodnji rob - log skala")
    ax.set_xlabel("Re")
    ax.set_ylabel("F")                    
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(Re,np.abs(casi))
    plt.savefig("Silelog.pdf")
if 0: #levo desno dol zgoraj
    print("h")
    N= 50
    Rez = solver(N,50000,0.001,5) #dol gor levo desn
    print("a")
    X , Y = np.meshgrid(np.linspace(0,1,N),np.linspace(0,1,N))
    plt.contourf(X,Y,Rez[0][-1].transpose(),cmap="hot")
    #plt.quiver(X,Y,Rez[1][-1].transpose(),Rez[2][-1].transpose())      
    r = np.linspace(0,1,20)
    plt.title("Re = 50000, t=5")
    plt.ylim(0,1)
    startpoints=np.array([(i,j) for i in r for j in r])
    plt.streamplot(X,Y,Rez[1][-1].transpose(),Rez[2][-1].transpose(),start_points=startpoints,color="k")
    plt.savefig("50000Re.pdf")    
if 0:
    BoundVel = np.array([1,0,0,0])
    N=50
    r = np.linspace(0,1,20)
    startpoints=np.array([(i,j) for i in r for j in r])
    print(startpoints.shape)
    Rez = solver(N,50000,0.001,25)
    Psiji = Rez[0]
    Uji = Rez[1]
    Vji = Rez[2]
    Zete = Rez[3]
    X , Y = np.meshgrid(np.linspace(0,1,N),np.linspace(0,1,N))
    Maks = np.amax(Psiji[-1])
    Minim = np.amin(Psiji[-1])
    NFrames = len(Psiji)
    print(NFrames)
    levels = np.linspace(Minim,Maks,50)
    fig, ax = plt.subplots()    
    def animiraj(t):
        print(t)
        ax.clear()
        ax.contourf(X, Y ,Psiji[t].transpose(),levels=levels,cmap="plasma")
        ax.streamplot(X,Y,Uji[t].transpose(),Vji[t].transpose(),start_points=startpoints,color="k")
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        Cas = round(0.001*t*50,2)
        summa = round(np.sum(Zete[t]),2)
        plt.suptitle(r"$Re = 50000$, relaksacija po 0.1s sunku")
        ax.set_title(r"$t={}, \sum \zeta = {}$".format(str(Cas),str(summa)))
        #ax.set_title(r"$t={}$".format(str(round(0.001*t*50,2))))
    ani = animation.FuncAnimation(fig,animiraj,range(NFrames),interval=50)   
    #plt.show()
    ani.save("Relaksacija2.mp4")
if 1:
    BoundVel = np.array([1,0,0,0])
    N=50
    r = np.linspace(0,1,20)
    startpoints=np.array([(i,j) for i in r for j in r])
    print(startpoints.shape)
    Rez = solver(N,50000,0.001,25,0.1,2)
    #Rez = solver(N,50000,0.001,25)
    Psiji = Rez[0]
    Uji = Rez[1]
    Vji = Rez[2]
    Zete = Rez[3]
    Sume = [np.abs(round(np.sum(i),2)) for i in Zete]
    Hitrosti = Rez[4]
    X , Y = np.meshgrid(np.linspace(0,1,N),np.linspace(0,1,N))
    Maks = np.amax(Psiji)
    Minim = np.amin(Psiji)
    NFrames = len(Psiji)
    print(NFrames)
    levels = np.linspace(Minim,Maks,50)
    fig, (ax1,ax2) = plt.subplots(2,1,gridspec_kw={"height_ratios":[2,1]},figsize=(15,15))
    ax3 = ax2.twinx()    
    def animiraj(t):
        print(t)
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax1.contourf(X, Y ,Psiji[t].transpose(),levels=levels,cmap="plasma")
        ax1.streamplot(X,Y,Uji[t].transpose(),Vji[t].transpose(),start_points=startpoints,color="k")
        ax1.set_xlim(0,1)
        ax1.set_ylim(0,1)
        Cas = round(0.001*t*50,2)
        #suma = round(np.sum(Zete[t]),2)
        plt.suptitle(r"$Re = 50000, \omega = 2$")
        ax1.set_title(r"$t={}$".format(str(Cas)))
        ax2.plot(np.linspace(0,Cas,t+1),Hitrosti[:t+1],"r",label="Hitrost roba")
        ax3.plot(np.linspace(0,Cas,t+1),Sume[:t+1],"b",label="Vrtincnost")
        #ax.set_title(r"$t={}$".format(str(round(0.001*t*50,2))))
        fig.legend(loc=(0.7,0.2))
    ani = animation.FuncAnimation(fig,animiraj,range(NFrames),interval=50)   
    #plt.show()
    ani.save("Oscilacija3.mp4")