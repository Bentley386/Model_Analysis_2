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

def naredimacko(st = 10):
    a = np.array([(x,0.25-x) for x in np.linspace(0,0.25,st)])
    b = np.array([(x,0) for x in np.linspace(0.25,0.75,st)])
    c = np.array([(x,-0.75+x) for x in np.linspace(0.75,1,st)])
    d = np.array([(1,x) for x in np.linspace(0.25,0.75,st)])
    e = np.array([(x,1.75-x) for x in np.linspace(0.75,1,st)])
    f = np.array([(x,0.25+x) for x in np.linspace(0.5,0.75,st)])
    g = np.array([(x,1.25-x) for x in np.linspace(0.25,0.5,st)])
    h = np.array([(x,0.75+x) for x in np.linspace(0,0.25,st)])
    i = np.array([(0,x) for x in np.linspace(0.25,0.75,st)])
    j = np.array([(x,0.5) for x in np.linspace(0.25,0.75,st)])
    k = np.array([(0.25,x) for x in np.linspace(0.5,0.75,st)])
    l = np.array([(0.75,x) for x in np.linspace(0.5,0.75,st)])
    m = np.array([(x,1-x) for x in np.linspace(0.25,0.5,st)])
    n = np.array([(x,x) for x in np.linspace(0.5,0.75,st)])
    return np.concatenate((a,b,c,d,e,f,g,h,i,j,k,l,m,n))


def pogojimacka(u):
    """predp da je x.shape (sodo,sodo)"""
    x = u[0]
    y = u[1]
    if y<=0.25-x or y<=0 or y<=-0.75+x:
        return 0.0
    if x>=1 or y>=1.75-x or (y>=0.25+x and x>=0.5):
        return 0.0
    if (y>=1.25-x and x<=0.5) or y>=0.75+x or x<=0:
        return 0.0
    if x>= 0.25 and y>=0.5 and y<=1-x:
        return 0.0
    if x<=0.75 and y>=0.5 and y<=x:
        return 0.0
    return 1.0
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
    return -1/6*((tocke[t[(m+1)%3]][0]-tocke[t[m]][0])*(tocke[t[(m+2)%3]][1]-tocke[t[m]][1])-(tocke[t[(m+1)%3]][1]-tocke[t[m]][1])*(tocke[t[(m+2)%3]][0]-tocke[t[m]][0]))
def sestaviMatriki(tocke,trikotniki,strob):
    aktiv=tocke[strob:].size//2
    A = np.zeros((aktiv,aktiv))
    B = np.zeros(aktiv)
    for trikot in trikotniki:
        for m in range(3):
            for n in range(3):
                if trikot[m] <strob or trikot[n]<strob:
                    continue
                A[trikot[m]-strob][trikot[n]-strob] += Atmn(trikot,m,n,tocke)
        for m in range(3):
            if trikot[m]<strob:
                continue
            B[trikot[m]-strob] += btm(trikot,m,tocke)
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
def enakomerenMacek(stevilo=10,strob=10):
    tocke = [(j/stevilo,i/stevilo) for j in range(stevilo+1) for i in range(stevilo+1)] + [(1/stevilo+j/stevilo,i/stevilo+1/(stevilo)) for j in range(stevilo+1) for i in range(stevilo+1)]
    tocke = [i for i in tocke if pogojimacka(i)]
    tocke = np.array(sorted(sorted(tocke,key=lambda x : x[0]),key=lambda x : x[1]))
    rob = naredimacko(strob)
    return np.concatenate((rob,tocke)) 
def randomMacek(stevilo=10,strob=10):
    tocke = np.random.rand(stevilo,2)
    tocke = [i for i in tocke if pogojimacka(i)]
    tocke = np.array(sorted(sorted(tocke,key=lambda x : x[0]),key=lambda x : x[1]))
    rob = naredimacko(strob)
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
def poisKoef(z,trikotniki,tocke,strob):
    zz = np.concatenate((np.zeros(strob),z.flatten()))
    koeficient=0
    ploscinca=0
    for k in trikotniki:
        temp=0
        for verteks in k:
            temp+= zz[verteks]
        plosc = ploscina(tocke[k[0]],tocke[k[1]],tocke[k[2]])
        koeficient+= temp/3*plosc
        ploscinca+=plosc
    return 8*pi*koeficient/(ploscinca**2)
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

    
    
if 1:
    kr=10 #koliko jih je na robu zdaj bo kr*14
    #A = np.array([[2,3,5],[3,6,8],[5,8,444]])
    #b = np.array([4,9,10])
    #print(lin.solve(A,b))
    #print(SOR(A,b,1.5))
    #print(ploscina([0.7,0.55],[-0.1,0.95],[0,0.9]))
    #tocke=enakomerenMacek(stevilo=20,strob=kr)
    tocke=randomMacek(3000,kr)
    #tocke=enakomerenTriag(stevilo=10,strob=kr)
    tri = scipy.spatial.Delaunay(tocke)
    Points = tri.points
    Triangles = tri.simplices
    Triangles2 = []
    Triangles3 = []
    for i in range(Triangles.size//3):
        if Triangles[i][0]<14*kr and Triangles[i][1]<14*kr and Triangles[i][2]<14*kr:
            continue
        Triangles2.append([Triangles[i][0],Triangles[i][1],Triangles[i][2]])
    Triangles = np.array(Triangles2)
    A, B = sestaviMatriki(Points,Triangles,14*kr)
    #print(Points.shape)
    #print(Triangles.shape)
    #print(Points[30:])
    fig, (ax1,ax2)=plt.subplots(1,2)
    fig.set_size_inches((10,5))
    kot = minkot(Triangles,Points)
    ax1.triplot(Points[:,0],Points[:,1],Triangles)
    #ax1.set_aspect(1)
    #x = np.abs(SOR(A,B,1.5))
    x = np.abs(lin.solve(A,B))
    #x = np.abs(lin.lstsq(A,B)[0])
    C = poisKoef(x,Triangles,Points,14*kr)
    cs = ax2.tricontourf(Points[:,0],Points[:,1],np.concatenate((np.zeros(14*kr),x.flatten())),levels=np.linspace(0,x[np.argmax(x)],50),cmap=plt.get_cmap("hot"))
    plt.colorbar(cs)
    #ax2.set_aspect(1)
    plt.suptitle("Random, Najmanjši kot je {} stopinj, Pois. koef. je {}".format(round(kot,2),round(C,4)))
    plt.savefig("druga/random5.pdf")    
if 0:
    kr=30
    #omege = []
    omege = np.linspace(0.5,1.9,50)
    iteracije = []
    #print(enakomerenTriag(20,10)[30:].size/2)
    tocke=randomMacek(1000,kr)
        #tocke=enakomerenTriag(stevilo=10,strob=kr)
    tri = scipy.spatial.Delaunay(tocke)
    Points = tri.points
    Triangles = tri.simplices
    A, B = sestaviMatriki(Points,Triangles,14*kr)
    #print(SOR(A,B,1))
    #print(SOR(A,B,0.5))
    for omega in omege:
        iteracije.append(SOR(A,B,omega))
    plt.title("SOR - norma razlike zadnjih vektorjev po 100 iteracijah")
    plt.xlabel(r"$\omega$")
    plt.ylabel("razlika")
    plt.yscale("log")
    plt.axvline(omege[np.argmin(iteracije)],color="r")
    plt.text(omege[np.argmin(iteracije)],0.01,r"$\omega={}$".format(round(omege[np.argmin(iteracije)],2)))
    plt.plot(omege,iteracije)
    plt.savefig("druga/odvisnost.pdf")        
if 0:
    kr=50//3 #koliko jih je na robu
    #A = np.array([[2,3,5],[3,6,8],[5,8,444]])
    #b = np.array([4,9,10])
    #print(lin.solve(A,b))
    #print(SOR(A,b,1.5))
    #print(ploscina([0.7,0.55],[-0.1,0.95],[0,0.9]))
    #tocke = randomTriag(stevilo=150,strob=kr)
    tocke=enakomerenTriag(stevilo=20,strob=kr)
    #tocke=enakomerenTriag(stevilo=10,strob=kr)
    tri = scipy.spatial.Delaunay(tocke)
    Points = tri.points
    print(Points.size/2)
    Triangles = tri.simplices
    A, B = sestaviMatriki(Points,Triangles,3*kr)
    #print(Points.shape)
    #print(Triangles.shape)
    #print(Points[30:])
    fig, (ax1,ax2)=plt.subplots(1,2)
    fig.set_size_inches((10,5))
    kot = minkot(Triangles,Points)
    ax1.triplot(Points[:,0],Points[:,1],Triangles)
    #ax1.set_aspect(1)
    #x = np.abs(SOR(A,B,0.5))
    x = np.abs(lin.solve(A,B))
    #x = np.abs(lin.lstsq(A,B)[0])
    C = poisKoef(x,Triangles,Points,kr)
    #cs = ax2.tricontourf(Points[:,0],Points[:,1],np.concatenate((np.zeros(3*kr),x.flatten())),levels=np.linspace(0,x[np.argmax(x)],50),cmap=plt.get_cmap("hot"))
    #plt.colorbar(cs)
    #ax2.set_aspect(1)
    #plt.suptitle("Random, Najmanjši kot je {} stopinj, Pois. koef. je {}".format(round(kot,2),round(C,4)))
    #plt.savefig("prva/random3.pdf")
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    





















def aij(tocke,trikotniki,sttock):
    #prviset=set(list(range(sttock)))
    rezult = 0
    def prodgrad(i,j,k):
        return ((j[0]-k[0])*(k[0]-i[0])+(j[1]-k[1])*(k[1]-i[1]))/(4*ploscina(i,j,k))
    for k in trikotniki:
        """
        if prviset & set(k) != set():
            continue
        """
        rezult+= prodgrad(tocke[k[0]],tocke[k[1]],tocke[k[2]])
    return rezult
    #return prodgrad(tocke[prvi[0]],tocke[prvi[1]],tocke[prvi[2]])+prodgrad(tocke[drugi[0]],tocke[drugi[1]],tocke[drugi[2]])
def bi(tocke,trikotniki,sttock):
    #prviset=set(list(range(sttock)))
    rezult = 0
    for i in trikotniki:
        """
        if prviset & set(i) != set():
            continue
        """
        rezult += ploscina(tocke[i[0]],tocke[i[1]],tocke[i[2]])
    return -1/3*rezult
def narediA(tocke,trikotniki,strob=30):
    sttock=int(tocke[strob:].size/2) #stevilo tock ki niso na robu
    def pomozna(i,j):
        """ i in j indici točk"""
        trikotnika = []
        if i==j:
            rezult=0
            for k in trikotniki:
                if i+strob in k:
                    seznam = [tocke[k[i]] for i in range(3) if k[i]!= (i+strob)]
                    djk2 = (seznam[0][0]-seznam[1][0])**2 + (seznam[0][1]-seznam[1][1])**2
                    return djk2/(4*ploscina(tocke[k[0]],tocke[k[1]],tocke[k[2]]))
                    #trikotnika.append(k)
            #return aij(tocke,trikotnika,strob)
        for k in trikotniki:
            if i+strob in k and j+strob in k:
                trikotnika.append(k)
        if len(trikotnika)<2:
            return 0.0
            return 1/0
            #print(tocke[trik[0]],tocke[trik[1]],tocke[trik[2]])
        if len(trikotnika)>2:
            print(1/0)
        """
        for k in trikotniki:
            if (i+strob in k and j+strob not in k) or (j+strob in k and i+strob not in k):
                trikotnika.append(k)
        """
        return aij(tocke,trikotnika,strob)
    return np.reshape(np.fromfunction(np.vectorize(pomozna),(sttock,sttock),dtype=int),(sttock,sttock))
    return np.matrix(np.fromfunction(np.vectorize(pomozna),(sttock,sttock),dtype=int))
def narediB(tocke,trikotniki,strob=30):
    sttock=int(tocke[strob:].size/2)
    def pomozna(i):
        trikotnicki= []
        for k in trikotniki:
            if i+strob in k:
                trikotnicki.append(k)
        return bi(tocke,trikotnicki,strob)
    return np.reshape(np.vectorize(pomozna)(np.arange(0,sttock,dtype=int)),sttock)
    return np.matrix(np.vectorize(pomozna)(np.arange(0,sttock,dtype=int))).T    
    
    
    
    
    
    
    
    
    
    
    
    