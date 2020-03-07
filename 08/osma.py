# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 21:35:07 2018

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

def ploscina(i,j,k):
    #plosc = 0.5*(i[0]*(j[1]-k[1]) +k[1]*(i[0]-j[1])+j[1]*(k[1]-i[0]))
    plosc= 0.5*np.abs((j[0]-i[0])*(k[1]-i[1])-(j[1]-i[1])*(k[0]-i[0]))
    return plosc
    plosc= i[0]*j[1]+j[0]*k[1]+k[0]*i[1]-i[0]*k[1]-j[0]*i[1]-k[0]*j[1]
    if plosc==0:
        print(i[0],i[1])
        print(j[0],j[1])
        print(k[0],k[1])
        print(1/0)
    return np.abs(plosc)
#print(ploscina((1,0),(2,0),(1,1)))
def Atmn(t,m,n,tocke):
    temp = 1/(4*ploscina(tocke[t[0]],tocke[t[1]],tocke[t[2]]))*((tocke[t[(m+1)%3]][1]-tocke[t[(m+2)%3]][1])*(tocke[t[(n+1)%3]][1]-tocke[t[(n+2)%3]][1]) + (tocke[t[(m+2)%3]][0]-tocke[t[(m+1)%3]][0])*(tocke[t[(n+2)%3]][0]-tocke[t[(n+1)%3]][0]))
    return temp
def btm(t,m,tocke):
    return 1/6*ploscina(tocke[t[0]],tocke[t[1]],tocke[t[2]])
    return 1/12*((tocke[t[(m+1)%3]][0]-tocke[t[m]][0])*(tocke[t[(m+2)%3]][1]-tocke[t[m]][1])-(tocke[t[(m+1)%3]][1]-tocke[t[m]][1])*(tocke[t[(m+2)%3]][0]-tocke[t[m]][0]))
def btmn(t,m,n,tocke):
    return 1/12*ploscina(tocke[t[0]],tocke[t[1]],tocke[t[2]])
def sestaviMatriki(tocke,trikotniki,strob):
    aktiv=tocke[strob:].size//2
    A = np.zeros((aktiv,aktiv))
    B = np.zeros((aktiv,aktiv))
    for trikot in trikotniki:
        for m in range(3):
            if trikot[m]<strob:
                continue
            for n in range(3):
                if trikot[n]<strob:
                    continue
                A[trikot[m]-strob][trikot[n]-strob] += Atmn(trikot,m,n,tocke)
                if trikot[m]==trikot[n]:
                    B[trikot[m]-strob][trikot[m]-strob] += btm(trikot,m,tocke)
                else:
                    B[trikot[m]-strob][trikot[n]-strob] += btmn(trikot,m,n,tocke)                    
    return (A,B)
def randomTriag(stevilo=10,strob=10):
    rob1 = np.reshape(np.dstack((np.linspace(-1,1,strob),np.zeros(strob))),(strob,2))
    rob2 = np.reshape(np.dstack((np.vectorize(np.cos)(np.linspace(0,pi,2*strob)),np.vectorize(np.sin)(np.linspace(0,pi,2*strob)))),(2*strob,2))     
    rob = np.concatenate((rob1,rob2))
    fiji = np.random.rand(stevilo)*pi
    ri = np.sqrt(np.random.rand(stevilo))
    tocke = np.reshape(np.dstack((ri*np.cos(fiji),ri*np.sin(fiji))),(stevilo,2)).tolist()
    tocke = np.array(sorted(sorted(tocke,key=lambda x : x[0]),key=lambda x : x[1]))
    return np.concatenate((rob,tocke))
def enakomerenTriag(stevilo=10,strob=10):
    rob1 = np.reshape(np.dstack((np.linspace(-1,1,strob),np.zeros(strob))),(strob,2))
    rob2 = np.reshape(np.dstack((np.vectorize(np.cos)(np.linspace(0,pi,2*strob)),np.vectorize(np.sin)(np.linspace(0,pi,2*strob)))),(2*strob,2))     
    rob = np.concatenate((rob1,rob2))
    tocke = [(-1+j*2/stevilo,i/stevilo) for j in range(stevilo+1) for i in range(stevilo+1)] + [(-1+1/stevilo+j*2/stevilo,i/stevilo+1/(2*stevilo)) for j in range(stevilo+1) for i in range(stevilo+1)]
    tocke = [i for i in tocke if i[0]**2+i[1]**2 < 1 and i[1]>0]
    tocke = np.array(sorted(sorted(tocke,key=lambda x : x[0]),key=lambda x : x[1]))
    return np.concatenate((rob,tocke))    
 
def SOR(A,b,w):
    stara = np.ones(b.size)
    nova = np.zeros(b.size)
    x=np.ones(b.size)
    #i = 0
    #print(w)
    #while lin.norm(stara-nova)>0.0000001:
    for i in range(100):
        stara = np.copy(nova)
        for i in range(b.size):
            x[i] = (1-w)*x[i]+w/A[i][i]*(b[i]-np.sum(A[i][:i]*x[:i]) - np.sum(A[i][i+1:]*x[i+1:]))
        nova = np.copy(x)
        #i+=1
    return(lin.norm(stara-nova))
    return(i)
    return x
def minkot(trikotniki,tocke):
    def zracunajkot(x):
        prvi = tocke[x[0]]
        drugi = tocke[x[1]]
        tretji = tocke[x[2]]
        vek1 = prvi-drugi
        vek2 = tretji-drugi
        return np.arccos(np.sum(vek1*vek2)/(lin.norm(vek1)*lin.norm(vek2)))        
    minimalen = pi
    for k in trikotniki:
        kot = [zracunajkot(k),zracunajkot(np.roll(k,1)),zracunajkot(np.roll(k,2))]
        if min(kot)<minimalen:
            minimalen = min(kot)
    return minimalen/pi*180

def resiGeneralised(A,B):
    L = lin.cholesky(B,overwrite_a=True)
    U,s,Vt = lin.svd(L) #koren Bja
    #S = lin.diagsvd(s,U[0].size,U[0].size)
    InvS = lin.diagsvd(1/s,U[0].size,U[0].size)
    InvKoren = np.matrix(Vt).H * InvS * np.matrix(U).H
    C = InvKoren * np.matrix(A) * InvKoren
    print(lin.eig(C)[0])
    # C = B^-1/2 * A * B^-1/2    
def galerkinA(m,k,l):
    return pi/2*((k*l)/(2*m+k+l) - (2*k*l+k+l)/(2*m+k+l+1) + (k+1)*(l+1)/(2*m+k+l+2))
    return pi/2*((1+2*m)*(l+2*m)+k*(1+2*l+2*m))/((k+l+2*m)*(1+k+l+2*m)*(2+k+l+2*m))
def galerkinB(m,k,l):
    return pi/2*(1/(2*m+k+l+2) - 2/(2*m+k+l+3) + 1/(2*m+k+l+4))    
def trial(r,m,k):
    if r<=k/10:
        return 10*r/k
    else:
        return (1-r)/(1-k/10)
def gradient(r,m,k):
    if r<=k/10:
        return 10/k
    else:
        return -1/(1-k/10)
def narediGalerkinMatriki(mcifra,kcifra):
    """baza je 10,11,12...1,k,20,21,22,....,2k"""
    """
    def integralA(m,k,n,l):
        def pomozna1(r):
            return pi*k*np.cos(k*pi*r)*pi*l*np.cos(l*pi*r)*r
        def pomozna2(r):
            return np.sin(k*pi*r)*np.sin(l*pi*r)/r
        return pi/2*scipy.integrate.quad(pomozna1,0,1)[0] + pi/2*m*m *scipy.integrate.quad(pomozna2,0,1)[0]
    def integralB(m,k,n,l):
        def pomozna(r):
            return np.sin(k*pi*r)*np.sin(l*pi*r)*r
        return pi/2*scipy.integrate.quad(pomozna,0,1)[0]
    """
    def integralA(m,k,n,l):
        def pomozna1(r):
            return r*gradient(r,m,k)*gradient(r,n,l)
        def pomozna2(r):
            return 1/r*trial(r,m,k)*trial(r,n,l)
        return pi/2*(scipy.integrate.quad(pomozna1,0,1)[0] + scipy.integrate.quad(pomozna2,0,1)[0])
    def integralB(m,k,n,l):
        def pomozna(r):
            return r*trial(r,m,k)*trial(r,n,l)
        return pi/2* scipy.integrate.quad(pomozna,0,1)[0]
    #vel = mcifra*(kcifra+1)
    vel = mcifra*kcifra
    A = np.zeros((vel,vel))
    B = np.zeros((vel,vel))
    for i in range(vel):
        for j in range(vel):
            m = i//(kcifra) + 1
            k = i%(kcifra)+1
            if k>9:
                print(i)
                print("stop")
                print(1/0)
            n = j//(kcifra) + 1
            l = j%(kcifra) + 1
            if m!=n:
                continue
            #A[i][j] = galerkinA(m,k,l)
            A[i][j] = integralA(m,k,n,l)
            B[i][j] = integralB(m,k,n,l)
    return(A,B)
def vrednostGalerkin(r,fi,mcifra,kcifra,koeficienti):
    vrednost = 0
    for i in range(mcifra*(kcifra+1)):
        m = i//(kcifra+1)+1
        k = i%(kcifra+1)
        vrednost += koeficienti[i]*r**(m+k)*(1-r)*np.sin(m*fi)
    return vrednost
def vrednostGalerkin2(r,fi,mcifra,kcifra,koeficienti):
    vrednost = 0
    for i in range(mcifra*(kcifra)):
        m = i//(kcifra+1)+1
        k = i%(kcifra+1) 
        vrednost += koeficienti[i]*np.sin(k*pi*r)*np.sin(m*fi)
    return vrednost
def vrednostGalerkin3(r,fi,mcifra,kcifra,koeficienti):
    vrednost = 0
    for i in range(mcifra*(kcifra)):
        m = i//(kcifra)+1
        k = i%(kcifra) + 1
        vrednost += koeficienti[i]*trial(r,m,k)*np.sin(m*fi)
    return vrednost    
if 1:
    #GALERKIN
    if 1:
        A,B = narediGalerkinMatriki(9,9)
        x = np.sort(lin.eig(A,B)[0])[:20]
        nicle = np.sort(np.concatenate((scipy.special.jn_zeros(1,20),scipy.special.jn_zeros(2,20),scipy.special.jn_zeros(3,20),scipy.special.jn_zeros(4,20),scipy.special.jn_zeros(5,20),scipy.special.jn_zeros(6,20),scipy.special.jn_zeros(7,20),scipy.special.jn_zeros(8,20),scipy.special.jn_zeros(9,20))))
        plt.title("Relativna napaka lastnih frekvenc, m=k=9")
        plt.plot(range(1,x.size+1),np.abs(nicle[:x.size]**2-x)/nicle[:x.size]**2,"o")
        #plt.plot(range(x.size),nicle[:x.size]**2,label="Kvadrati ničel Bessela",color="r")
        #plt.plot(range(x.size),x,label="Izračunane frekvence")
        #plt.legend(loc="best")
        #plt.yscale("log")
        plt.xticks([2,4,6,8,10,12,14,16,18])
        plt.ylabel("Napaka")
        plt.xlabel("Zaporedna številka lastne vrednosti")
        plt.savefig("druga/idk2.pdf")
    if 0:
        #mji = [1,2,3,4,5,6,7,8,9,10]
        #kji = [1,2,3,4,5,6,7,8,9,10]
        nicle = np.sort(np.concatenate((scipy.special.jn_zeros(1,20),scipy.special.jn_zeros(2,20),scipy.special.jn_zeros(3,20),scipy.special.jn_zeros(4,20),scipy.special.jn_zeros(5,20),scipy.special.jn_zeros(6,20),scipy.special.jn_zeros(7,20),scipy.special.jn_zeros(8,20),scipy.special.jn_zeros(9,20))))
        lastna = nicle[0]**2
        for k in range(10):
            vrednosti = []
            for m in range(10):
                A, B = narediGalerkinMatriki(mji[m],kji[k])
                vrednosti.append(np.abs(lin.eigh(A,B)[0][0]-lastna)/lastna)
            plt.plot(range(1,11),vrednosti,label="k={}".format(k))
        plt.legend(loc="best")
        plt.xlabel("m")
        plt.ylabel("Rel. napaka")
        plt.yscale("log")
        plt.title("Napake pri prvi lastni vrednosti")
        plt.savefig("druga/sinprimerjava.pdf")
        #plt.colormap(cs)
    if 0:
        A, B  = narediGalerkinMatriki(9,9)
        vektor = lin.eig(A,B)
        print(vektor[0])
        """
        katera = 0
        vrednost = vektor[0][katera]
        vektor = vektor[1][:,katera]
        """
        katera = np.argmin(np.abs(vektor[0]))
        vrednost = np.real(vektor[0][katera])
        vektor = np.real(vektor[1][:,katera])
        rji = np.linspace(0,1,100)
        koti = np.linspace(0,pi,100)
        R,Fi = np.meshgrid(rji,koti)
        X, Y = R*np.cos(Fi), R*np.sin(Fi)
        Z = np.zeros((100,100))
        for i in range(100):
            for j in range(100):
                Z[i][j] = vrednostGalerkin3(rji[j],koti[i],9,9,vektor)
        cs = plt.contourf(X,Y,Z,levels=np.linspace(min(Z.flatten()),max(Z.flatten()),50))
        plt.title(r"$\omega = {}, m=k=9$".format(round(np.sqrt(vrednost),2)))
        plt.colorbar(cs)
        plt.savefig("druga/idk.png")
if 0:
    #FEM
    kr=300 #koliko jih je na robu zdaj bo kr*14
    tocke=enakomerenTriag(stevilo=50,strob=kr)
    tri = scipy.spatial.Delaunay(tocke)
    Points = tri.points
    print(Points.size)
    Triangles = tri.simplices
    A, B = sestaviMatriki(Points,Triangles,3*kr)
    #resiGeneralised(A,B)
    sA = scipy.sparse.csr_matrix(A)
    sB = scipy.sparse.csr_matrix(B)
    #fig, (ax1,ax2)=plt.subplots(1,2)
    #fig.set_size_inches((10,5))
    #kot = minkot(Triangles,Points)
    #ax1.triplot(Points[:,0],Points[:,1],Triangles)
    #ax1.set_aspect(1)
    #x = np.abs(SOR(A,B,1.5))
    x = np.unique(scipy.sparse.linalg.eigsh(sA,k=100,M=sB,which="SM")[0][:20])
    #x = np.abs(lin.eig(A,B)[0])
    #x = np.abs(lin.eig(A,B)[0])
    #x = np.abs(lin.lstsq(A,B)[0])
    print(x)
    if 1:
        nicle = np.sort(np.concatenate((scipy.special.jn_zeros(1,20),scipy.special.jn_zeros(2,20),scipy.special.jn_zeros(3,20),scipy.special.jn_zeros(4,20),scipy.special.jn_zeros(5,20),scipy.special.jn_zeros(6,20),scipy.special.jn_zeros(7,20),scipy.special.jn_zeros(8,20),scipy.special.jn_zeros(9,20))))
        #plt.title("Relativna napaka lastnih frekvenc")
        #plt.plot(range(1,x.size+1),np.abs(nicle[:x.size]**2-x)/nicle[:x.size]**2,"o")
        plt.plot(range(1,x.size+1),nicle[:x.size]**2,"-.",label="Kvadrati ničel Bessela",color="r")
        plt.plot(range(1,x.size+1),x,label="Izračunane frekvence",ls="-.")
        plt.legend(loc="best")
        #plt.yscale("log")
        plt.xticks([2,4,6,8,10,12,14,16,18])
        plt.ylabel("Napaka")
        plt.xlabel("Zaporedna št. lastnih vrednosti")
        plt.savefig("prva/primerjava.pdf")
        
    #xx = x[1][:,7]
    #cs = plt.tricontourf(Points[:,0],Points[:,1],np.concatenate((np.zeros(3*kr),xx.flatten())),levels=np.linspace(xx[np.argmin(xx)],xx[np.argmax(xx)],50))
    #plt.colorbar(cs)
    #ax2.set_aspect(1)
    #plt.suptitle(r"$\omega ={}$".format(round(np.sqrt(x[0][7]),2)))
    #plt.savefig("prva/7.png")    
    
    
    
    
    
"""    
def trial(m,k,r,fi):
    def pomozna1():
        if r<=m/5:
            return 5*r/m
        else:
            return (1-r)/(1-m/5)
    def pomozna2():
        if fi<=pi*k/5:
            return 5*fi/(pi*k)
        else:
            return (pi-fi)/(pi-pi*k/5)
    return pomozna1()*pomozna2()
def trialGrad(m,k,r,fi):
    def pomozna1():
        if r<=m/5:
            return 5*r/m
        else:
            return (1-r)/(1-m/5)
    def pomozna2():
        if fi<=pi*k/5:
            return 5*fi/(pi*k)
        else:
            return (pi-fi)/(pi-pi*k/5)
    def pomozna3():
        if r<=m/5:
            return 5/m
        else:
            return -1/(1-m/5)
    def pomozna4():
        if fi<=pi*k/5:
            return 5/((r+0.0001)*pi*k)
        else:
            return -1/((r+0.0001)*(pi-pi*k/5))
    return [pomozna3()*pomozna2(),pomozna1()*pomozna4()]

"""    