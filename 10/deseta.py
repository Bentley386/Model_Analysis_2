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

def gostote(i,j,N,gost): #gost je gostota sencenega, x desno y dol
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
    

def gaussSeidlerSOR(u,h,q,omega): #resi laplac u = q
    velikost = u[0].size
    uu = np.copy(u)
    def pomozna(i,j):
        if i==0 or i==velikost-1 or j==0 or j==velikost-1:
            return 0.0
        return 0.25*(uu[i-1][j]+uu[i+1][j] + uu[i][j-1]+uu[i][j+1] + h*h*q[i][j])
    for i in range(velikost):
        for j in range(velikost):
            uu[i][j] = u[i][j]+omega*(pomozna(i,j)-u[i][j])
    return uu
def gaussTemp(u,h,T1,T2,omega): #resi laplac u = 
    velikost = u[0].size
    uu = np.copy(u)
    def pomozna(i,j):
        if i==0 or i==velikost-1:
            return 0.0
        if j==0:
            #return 0.0
            return uu[i][1]
        if j==velikost-1:
            return T2-T1
        return 1/(2*np.cos(pi*(i)/velikost)-4)*(-(1-1/(2*(j)))*uu[i][j-1]-(1+1/(2*(j)))*uu[i][j+1])
    for i in range(velikost):
        for j in range(velikost):
            uu[i][j] = u[i][j]+omega*(pomozna(i,j)-u[i][j])
    return uu        
def sineTransform(a):
    N = len(a)
    pomozniarej = [0] + [np.sin(i*pi/N)*(a[i]+a[N-i]) + 0.5*(a[i]-a[N-i]) for i in range(1,N)]
    transformiran = np.fft.rfft(pomozniarej)
    imaginarni = np.imag(transformiran)
    realni = np.real(transformiran)
    transformiranka = np.ones(N)
    for i in range(N):
        if i==0:
            transformiranka[i]=imaginarni[0]
            continue
        if i==1:
            transformiranka[i]=0.5*realni[0]
            continue
        if i%2==0:
            transformiranka[i]=imaginarni[i//2]
        else:
            transformiranka[i]=transformiranka[i-2]+realni[(i-1)//2]
    return transformiranka        
def cosineTransform(a):
    N = len(a)-1
    pomozniarej = [0.5*(a[i]+a[N-i]) - np.sin(pi*i/N)*(a[i]-a[N-i]) for i in range(N)]
    transformiran = np.fft.rfft(pomozniarej)
    imaginarni = np.imag(transformiran)
    realni = np.real(transformiran)
    transformiranka = np.ones(N+1)
    for i in range(N+1):
        if i==0:
            transformiranka[i]=realni[0]
            continue
        if i==1:
            transformiranka[i]=0.5*(a[0]-a[N])+ np.sum(a[1:N]*np.array([np.cos(j*pi/N) for j in range(1,N)]))
            continue
        if i%2==0:
            transformiranka[i]=realni[i//2]
        else:
            transformiranka[i]=transformiranka[i-2]+imaginarni[(i-1)//2]
    return transformiranka       
def dvojniFourier(u,h,q):
    velikost = u[0].size
    mji = list(range(velikost))
    mji[0] = 0.00001
    mji = np.array(mji)
    M, N = np.meshgrid(mji,mji)
    G = np.transpose(np.array([sineTransform(q[i]) for i in range(velikost)]))
    G = np.transpose(np.array([sineTransform(G[i]) for i in range(velikost)]))
    #G = fftpack.dstn(q,type=1)
    uu = 0.5*h*h/(np.cos(pi*N/(velikost)) + np.cos(pi*M/(velikost)) - 2) * G
    """
    uu = np.ones((velikost,velikost))
    for i in range(velikost):
        for j in range(velikost):
            uu[i][j] = 0.5*h*h/(np.cos(pi*(i+1)/(velikost)) + np.cos(pi*(j+1)/(velikost))-2)* G[i][j]
    """
    uuu = 2/velikost * np.transpose(np.array([sineTransform(uu[i]) for i in range(velikost)]))
    return 2/velikost * np.transpose(np.array([sineTransform(uuu[i]) for i in range(velikost)]))
    #return 1/(2*(velikost+1))*fftpack.idst(uu,type=1)
    
    
def enojniFourier(u,h,q):
    velikost = u[0].size
    G = np.array([sineTransform(q[i]) for i in range(velikost)])
    #G = fftpack.dst(q,type=1,axis=1) #four transf po iksu
    if 0:
        diag = np.repeat(np.array([-4+2*np.cos(m*pi/velikost) for m in range(velikost)]),velikost) #vektor u je oblike 00 01 02 03... 10 11 12.... xx = ml
        obdiag = np.ones(velikost*velikost)
        matrika = scipy.sparse.diags([diag,obdiag,obdiag],[0,-1,1],(velikost*velikost,velikost*velikost),"csc")
        #matrika = matrika.tocsc()
        #resitev = TDMAsolver(np.concatenate((np.array([0]),diag[1:])),diag,np.concatenate((obdiag[:-1],np.array([0]))),vektor)
        #U = np.reshape(resitev,(N,N))
        U = np.reshape(scipy.sparse.linalg.spsolve(matrika,vektor),(velikost,velikost))
    def resi(m):
        diag = np.ones(velikost)*(2*np.cos(pi*(m-1)/velikost)-4)
        obdiag = np.ones(velikost)
        matrika = scipy.sparse.diags([diag,obdiag,obdiag],[0,-1,1],(velikost,velikost),"csc")
        resitev = scipy.sparse.linalg.spsolve(matrika,h*h*G[:,m-1])
        return resitev
    U = np.transpose(np.array([resi(m) for m in range(1,N+1)]))
    return 2/velikost*np.array([sineTransform(U[i])for i in range(velikost)])
    #return fftpack.idst(U,type=1,axis=1)
def temperaturni(u,h,T1,T2):
    velikost = u[0].size
    #G = np.transpose(np.array([sineTransform(u[:,i]) for i in range(velikost)])) #FT po z
    #G = fftpack.dst(q,type=1,axis=1) #four transf po iksu
    #eps = 1
    q = np.array([[0 if i!=velikost-1 else T2-T1*(1-j/velikost) for i in range(velikost)] for j in range(velikost)])
    Q = np.transpose(np.array([cosineTransform(q[:,i]) for i in range(velikost)]))
    #star = np.zeros((velikost,velikost))
    #i=0
    """
    while eps>0.5 and i<100:
        i+=1
        print(i)
        u = gaussTemp(u,h,T1,T2,1)
        eps = np.sum(np.abs(star-u))
        star = np.copy(u)
    """
    
    def resi(m):
        diag = np.ones(velikost)*(2*np.cos(pi*(m)/velikost)-4)
        diag[-1] = 1
        diag[0]=1
        obdiaggor = np.array([-1]+[1+1/(2*(k)) for k in range(1,velikost-1)])
        obdiagdol = np.array([1-1/(2*(k))for k in range(1,velikost-1)]+[0])
        matrika = scipy.sparse.diags([diag,obdiagdol,obdiaggor],[0,-1,1],(velikost,velikost),"csc")
        resitev = scipy.sparse.linalg.spsolve(matrika,Q[m])
        return resitev
    U = np.array([resi(m) for m in range(velikost)])
    return 2/velikost*(np.transpose(np.array([cosineTransform(U[:,i])for i in range(velikost)])))+T1* np.transpose(np.array([[T1*(1-j)/velikost for j in range(velikost)]for i in range(velikost)]))
    #return fftpack.idst(U,type=1,axis=1
if 0:
    N = 1000
    u = np.ones((N,N))
    u = temperaturni(u,1/N,0.1,1)
    x = np.linspace(0,1,N)
    X,Y = np.meshgrid(x,x)
    #plt.imshow(u)
    fig,ax = plt.subplots()
    print(u)
    cs = ax.contourf(X,Y,np.abs(u),levels=np.linspace(np.amin(u),np.amax(u),50),cmap="hot")
    #cs = ax.contourf(X,Y,np.transpose(np.fliplr(np.transpose(u))),levels=np.linspace(np.amin(u),np.amax(u),30))
    plt.colorbar(cs)
    ax.set_aspect(1)
    plt.title(r"Temperaturni profil, $j_1 = 0.1, T_2 = 1$")
    plt.savefig("toplotni3.png")    
if 0:    
    N = 1000
    u = np.ones((N,N))
    q = np.array([[gostote(i,j,N,2) for i in range(N)]for j in range(N)])
    u = np.abs(dvojniFourier(u,1/N,q))
    x = np.linspace(0,1,N)
    X,Y = np.meshgrid(x,x)
    #plt.imshow(u)
    fig,ax = plt.subplots()
    cs = ax.contourf(X,Y,np.transpose(np.fliplr(np.transpose(u))),levels=np.linspace(np.amin(u),np.amax(u),30))
    plt.colorbar(cs)
    ax.set_aspect(1)
    plt.title("2D Fourier, 1000x1000, obtezena opna, poves na sredini = {}".format(str(round(u[N//2][N//2],5))))
    plt.savefig("obtezena/2dfour1000.png")
    
if 0:
    N = 100
    u = np.ones((N,N))
    #q = np.ones((N,N))
    q = np.array([[gostote(i,j,N,2) for i in range(N)]for j in range(N)])
    eps = 1
    i = 0
    star = np.zeros((N,N))
    while eps>0.0001:
        i+=1
        u = gaussSeidlerSOR(u,1/N,q,1.95)
        eps = np.sum(np.abs(star-u))
        star = np.copy(u)
    x = np.linspace(0,1,N)
    X,Y = np.meshgrid(x,x)
    #plt.imshow(u)
    fig,ax = plt.subplots()
    u = np.abs(u)
    cs = ax.contourf(X,Y,np.fliplr(u),levels=np.linspace(np.amin(u),np.amax(u),30))
    plt.colorbar(cs)
    ax.set_aspect(1)
    plt.title("SOR Gauss, 100x100, obtezena opna, poves na sredini = {}".format(str(round(u[N//2][N//2],5))))
    plt.savefig("obtezena/SOR100.png")
if 0:
    #kalibracija SORa
    N = 100
    u = np.ones((N,N))
    #q = np.copy(u)
    q = np.array([[gostote(i,j,N,2) for i in range(N)]for j in range(N)])
    if 0:
        for i in range(10000):
            u = gaussSeidlerSOR(u,1/N,q,1)
        prava= u[N//2][N//2]
        print(prava)
        print(1/0)
    omege = np.linspace(1,2,20)
    vrednosti = []
    for omega in omege:
        u = np.ones((N,N))
        for i in range(50):
            u = gaussSeidlerSOR(u,1/N,q,omega)
        vrednosti.append(u[N//2][N//2])
    razlike = np.abs(np.abs(np.array(vrednosti))-0.09617055253675486)
    plt.plot(omege,razlike)
    minimalna = omege[np.argmin(razlike)]
    plt.axvline(minimalna,color="r")
    plt.text(minimalna,0.8,r"$\omega = {}$".format(str(round(minimalna,2))))
    plt.xlabel(r"$\omega$")
    plt.title("Kalibracija SOR metode")
    plt.ylabel("Napaka")
    plt.savefig("kalibracija2.pdf")
    #ax.axvline()
#plt.plot(np.linspace(0,1,50),np.ones(50))
#plt.plot(np.linspace(0,1,50),2*np.ones(50))
#plt.plot(np.linspace(0,1,50),3*np.ones(50))

if 0:
    #Nji = [100,200,300,400,500,600,700,800,900,1000]
    Four1 = []
    Four2 = []
    SOR = []
    Nji = [60,80,100,120,140,160,180,200]
    #Nji=[60]
    for N in Nji:
        q = np.ones((N,N))
        u = np.ones((N,N))
        start = timeit.default_timer()
        star = np.zeros((N,N))
        eps = 1
        i=0
        while eps>0.1:
            i+=1
            u = gaussSeidlerSOR(u,1/N,q,1.95)
            eps = np.sum(np.abs(star-u))
            star = np.copy(u)
        SOR.append(timeit.default_timer()-start)
        start = timeit.default_timer()
        u = dvojniFourier(q,1/N,q)
        Four2.append(timeit.default_timer()-start)
        start = timeit.default_timer()
        u = enojniFourier(q,1/N,q)
        Four1.append(timeit.default_timer()-start)
    plt.title("Primerjava hitrosti metod")
    plt.xlabel("N")
    #print(Four1)
    #print(Four2)
    #print(SOR)
    plt.ylabel("Hitrost[s]")
    plt.yscale("log")
    plt.xscale("log")
    plt.plot(Nji,SOR,label="SOR") 
    plt.plot(Nji,Four1,label="1D Fourier") 
    plt.plot(Nji,Four2,label="2D Fourier")    
    plt.legend(loc="best")     
    plt.savefig("hitrosti.pdf")





    
