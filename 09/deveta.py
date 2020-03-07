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
HitrostDalec = 100
class Plaketa():
    def __init__(self,levi,desni):
        self.levi = levi
        self.desni = desni
        self.sredina = (levi+desni)/2
        self.tangenta = desni - levi
        self.dolzina = lin.norm(self.tangenta)
        self.tangenta = self.tangenta/self.dolzina
        self.normala = -np.array([-self.tangenta[1],self.tangenta[0]])
        self.ex = self.transformirajNaPlaketo(1,0,True)
        self.ey = self.transformirajNaPlaketo(0,1,True)
    def transformirajNaPlaketo(self,x,y,nitocka=False): #iz laboratorijskega sistema na plaketo
        if nitocka:
            return self.transformirajNaPlaketo(x,y) - self.transformirajNaPlaketo(0,0)
        else:
            vektor = np.array([x,y]) - self.sredina 
        return np.array([-np.dot(vektor,self.tangenta),np.dot(vektor,self.normala)])
    def transformirajIzPlakete(self,x,y,nitocka=False): #obratno
        vektor2 = np.array([np.dot([x,y],self.ex),np.dot([x,y],self.ey)])
        if nitocka:
            return vektor2
        return vektor2 + self.sredina       

def narediPlaketeTrak(funkcija,N):
    tocke = np.linspace(0,1,N)
    levi = np.array([funkcija(i) for i in tocke[:-1]])
    desni = np.array([funkcija(i) for i in tocke[1:]])
    return [Plaketa(levi[i],desni[i]) for i in range(N-1)]
            
def narediPlakete(funkcija,N,A,B): #naredi seznam objektov plaketa
    tocke = np.linspace(0,1,N)
    levi = np.array([funkcija(i,A,B,0.9) for i in tocke[:-1]])
    desni = np.array([funkcija(i,A,B,0.9) for i in tocke[1:]])
    return [Plaketa(levi[i],desni[i]) for i in range(N-1)]
"""
def narediPlakete(funkcija,N,t): #naredi seznam objektov plaketa
    tocke = np.linspace(0,1,N)
    levi = np.array([funkcija(i,t) for i in tocke[:-1]])
    desni = np.array([funkcija(i,t) for i in tocke[1:]])
    return [Plaketa(levi[i],desni[i]) for i in range(N-1)]
"""
def plaketeTrak(i):
    return np.array([-1/2 + i,0])
def plaketeElipsa(i,b):
    fi = 2*pi*i
    return np.array([np.cos(fi),b*np.sin(fi)])

def plaketeZukovski(i,A,B,r):
    zz = A+1j*B + r*np.exp(-1j*2*pi*i)
    z = 0.5*(zz+1/zz)
    return np.array([np.real(z),np.imag(z)])
def plaketeNACA(i,t):
    if i<=0.5:
        x = i*2
        return np.array([x,-t/50*(1.457122*np.sqrt(x)-0.624424*x-1.727016*x*x+1.384087*x*x*x - 0.489769*x*x*x*x)])
    else:
        x = (1-i)*2
        return np.array([x,t/50*(1.457122*np.sqrt(x)-0.624424*x-1.727016*x*x+1.384087*x*x*x - 0.489769*x*x*x*x)])
def potencialZaradiPlakete(xx,yy,plaketa,N,sigma): #tista formula iz navodil
    #x =xx - (-1/2 + 1/(2*N) + i/N)
    x,y = plaketa.transformirajNaPlaketo(xx,yy)
    l = plaketa.dolzina
    prvi = -l + y*(np.arctan((x+l/2)/y)-np.arctan((x-l/2)/y))
    ostali = (x+l/2)/2*np.log((x+l/2)**2+y*y) - (x-l/2)/2*np.log((x-l/2)**2+y*y)
    #return x+y+1/(2*N)
    return sigma/(2*pi)*(prvi+ostali)
def poljeZaradiPlakete(xx,yy,plaketa,N,sigma): #formula iz navodil
    #x = xx-(-1/2 + 1/(2*N)+i/N)
    x, y = plaketa.transformirajNaPlaketo(xx,yy)
    l = plaketa.dolzina
    xmin= x+l/2
    xpl = x-l/2
    iks = -sigma/(4*pi)*np.log((xmin*xmin + y*y)/(xpl*xpl+y*y))
    ips = -sigma/(2*pi)*(np.arctan(xmin/y) - np.arctan(xpl/y))
    return np.array([iks,ips])
    """
    if 0:
        prvix = 1/(1+(xmin/y)**2) - 1/(1+(xpl/y)**2)
        #prvix = 1/(1+xmin**2) - 1/(1+xpl**2) ??
        ostalix = 0.5*(np.log((xmin**2+y**2)/(xpl**2+y**2)) + 2*xmin**2/(xmin**2+y**2) - 2*xpl**2/(xpl**2+y**2))
        prviy = np.arctan(xmin/y) - np.arctan(xpl/y) + y*(-xmin/(xmin**2+y**2)+xpl/(xpl**2+y**2))
        ostaliy= y*xmin/(xmin**2+y**2) - y*xpl/(xpl**2+y**2)
        if koncna:
            return np.array([-sigma/(2*pi)*(prvix+ostalix)+0.00001, -sigma/(2*pi)*(prviy+ostaliy)])
        else:
            return np.array([-sigma/(2*pi)*(prvix+ostalix),-sigma/(2*pi)*(prviy+ostaliy)])
    """
def najdiGostotePotencial(plakete): #rešitev sistema A*sigma = 1
    N = len(plakete)
    U = np.ones(N)
    def naredi(i,j):
        return potencialZaradiPlakete(plakete[i].sredina[0],plakete[i].sredina[1],plakete[j],N,1)
    A = np.fromfunction(np.vectorize(naredi),(N,N),dtype=int)
    return lin.solve(A,U)
def najdiVodnjakePotencial(plakete): #rešitve sistema A*sigma=u
    N = len(plakete)
    def naredi(i,j):
        if i==j:
            return -0.5
        temp = plakete[j].transformirajNaPlaketo(plakete[i].sredina[0],plakete[i].sredina[1])
        polje = poljeZaradiPlakete(temp[0],temp[1],plakete[j],N,1)
        polje2 = plakete[j].transformirajIzPlakete(polje[0],polje[1],True)
        return plakete[i].transformirajNaPlaketo(polje2[0],polje2[1],True)[1]
    A = np.fromfunction(np.vectorize(naredi),(N,N),dtype=int)
    if 1:
        def naredi2(i):
            return -plakete[i].transformirajNaPlaketo(HitrostDalec,0,True)[1] 
        U = [naredi2(i) for i in range(N)]
    #U = np.zeros(N)
    X = lin.solve(A,U)
    return X
def vrednost(x,y,naboji,plakete,polje=False): #izračun potenciala/polja
    rezult = 0
    if polje:
        rezult = np.zeros(2)
        dalec = np.array([HitrostDalec,0])
        #dalec = np.array([0,HitrostDalec])

    N = len(naboji)
    for i in range(N):
        if polje:
            temp = poljeZaradiPlakete(x,y,plakete[i],N,naboji[i])
            rezult+= plakete[i].transformirajIzPlakete(temp[0],temp[1],True)
            continue
        rezult+= potencialZaradiPlakete(x,y,plakete[i],N,naboji[i])
    if polje:
        return rezult + dalec
    else:
        return rezult - x*HitrostDalec

if 1:
    #ZURKOVSKI
    #nabojcki = naboji(
    HitrostDalec = 1
    if 0:
        Aji = np.linspace(0.01,0.1,20)
        Bji = np.linspace(0.01,0.1,20)
        def pomozna(A,B):
            plaketke = narediPlakete(plaketeZukovski,50,A,B)
            nabojcki = najdiVodnjakePotencial(plaketke)
            ipsiloni = [i.sredina[1] for i in plaketke]
            return vrednost(0,max(ipsiloni)+0.05,nabojcki,plaketke,True)[0]-vrednost(0,min(ipsiloni)-0.05,nabojcki,plaketke,True)[0]
        vrednosti = [[pomozna(a,b) for a in Aji] for b in Bji]
        AA, BB = np.meshgrid(Aji,Bji)
        plt.xlabel("A")
        plt.ylabel("B")
        cs = plt.contourf(AA,BB,vrednosti,levels=np.linspace(np.amin(vrednosti),np.amax(vrednosti),50),cmap="hot")
        plt.colorbar(cs)
        plt.title("Razlika hitrosti nad in pod krilom")        
    if 1:
        plaketke = narediPlakete(plaketeZukovski,50,A,B)
        nabojcki = najdiVodnjakePotencial(plaketke)
        ghost = np.array([0.0000001,0.0000001])
    if 1:
        elipsagor = np.array([plaketeZukovski(i,A,B,0.9) for i in np.linspace(0,0.5,100)])
        elipsadol = np.array([plaketeZukovski(i,A,B,0.9) for i in np.linspace(0.5,1,100)])[::-1]
        #plt.plot(elipsagor[:,0],elipsagor[:,1])
        #print(1/0)
        x = np.linspace(-2,2,200)
        y = np.linspace(-1,1,200)
        X,Y = np.meshgrid(x,y)
        XX,YY = np.meshgrid(np.linspace(-2,2,20),np.linspace(-1,1,20))
        Z = [[vrednost(x,y,nabojcki,plaketke) for x in np.linspace(-2,2,200)] for y in np.linspace(-1,1,200)]
        #polja = np.array([[vrednost(xx,yy,nabojcki,plaketke,True) if ((xx>=1 or xx<=-1) or  (yy>b*np.sqrt(1-xx*xx) and yy<(-b)*np.sqrt(1-xx*xx))) else np.array([0.0000001,0.000001]) for xx in np.linspace(-5,5,20) ] for yy in np.linspace(-5,5,20)])
        polja = np.array([[vrednost(xx,yy,nabojcki,plaketke,True) for xx in np.linspace(-2,2,20)] for yy in np.linspace(-1,1,20)])
        cs = plt.contourf(X,Y,Z,levels=np.linspace(np.amin(Z),np.amax(Z),50),cmap="viridis")
        plt.colorbar(cs)
        plt.quiver(XX,YY,polja[:,:,0],polja[:,:,1])
        plt.fill_between(elipsagor[:,0],elipsagor[:,1],elipsadol[:,1],color="grey")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(r"N=50, $A=0.1, B=0.1,  v_{\infty} = 1$")
    plt.savefig("druga/prva.png")        
if 0:
    #NACA
    #nabojcki = naboji(50)
    b=30
    HitrostDalec= -1
    plaketke = narediPlakete(plaketeNACA,50,b)
    nabojcki = najdiVodnjakePotencial(plaketke)
    ghost = np.array([0.0000001,0.0000001])
    if 0:
        plt.title(r"N=50, Porazdelitev izvirov po elipsi $b=1, v_{\infty} = 1$")
        plt.ylabel("izviri")
        plt.xlabel(r"$\varphi$")
    if 1:
        elipsadol = np.array([plaketeNACA(i,b) for i in np.linspace(0,0.5,100)])
        elipsagor = np.array([plaketeNACA(i,b) for i in np.linspace(0.5,1,100)])[::-1]
        #plt.plot(elipsagor[:,0],elipsagor[:,1])
        #print(1/0)
        x = np.linspace(-1,2,200)
        y = np.linspace(-1,1,200)
        X,Y = np.meshgrid(x,y)
        XX,YY = np.meshgrid(np.linspace(-1,2,20),np.linspace(-1,1,20))
        Z = [[vrednost(x,y,nabojcki,plaketke) for x in np.linspace(-1,2,200)] for y in np.linspace(-1,1,200)]
        #polja = np.array([[vrednost(xx,yy,nabojcki,plaketke,True) if ((xx>=1 or xx<=-1) or  (yy>b*np.sqrt(1-xx*xx) and yy<(-b)*np.sqrt(1-xx*xx))) else np.array([0.0000001,0.000001]) for xx in np.linspace(-5,5,20) ] for yy in np.linspace(-5,5,20)])
        polja = np.array([[vrednost(xx,yy,nabojcki,plaketke,True) for xx in np.linspace(-1,2,20)] for yy in np.linspace(-1,1,20)])
        cs = plt.contourf(X,Y,Z,levels=np.linspace(np.amin(Z),np.amax(Z),50),cmap="viridis")
        plt.colorbar(cs)
        plt.quiver(XX,YY,polja[:,:,0],polja[:,:,1])
        plt.fill_between(elipsagor[:,0],elipsagor[:,1],elipsadol[:,1],color="grey")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(r"N=50, Obtekanje okoli NACA krila $t=30, v_{\infty} = -1$")
    #plt.savefig("druga/naca6.png")
if 0:
    #ELIPSA
    #nabojcki = naboji(50)
    b=0.5
    HitrostDalec= 1
    plaketke = narediPlakete(plaketeNACA,50,b)
    nabojcki = najdiVodnjakePotencial(plaketke)
    ghost = np.array([0.0000001,0.0000001])
    if 0:
        plt.title(r"N=50, Porazdelitev izvirov po elipsi $b=1, v_{\infty} = 1$")
        plt.ylabel("izviri")
        plt.xlabel(r"$\varphi$")
    if 1:
        elipsagor = np.array([plaketeNACA(i,b) for i in np.linspace(0,0.5,100)])
        elipsadol = np.array([plaketeNACA(i,b) for i in np.linspace(0.5,1,100)])[::-1]
        x = np.linspace(-5,5,200)
        y = np.linspace(-5,5,200)
        X,Y = np.meshgrid(x,y)
        XX,YY = np.meshgrid(np.linspace(-5,5,20),np.linspace(-5,5,20))
        Z = [[vrednost(x,y,nabojcki,plaketke) for x in np.linspace(-5,5,200)] for y in np.linspace(-5,5,200)]
        #polja = np.array([[vrednost(xx,yy,nabojcki,plaketke,True) if ((xx>=1 or xx<=-1) or  (yy>b*np.sqrt(1-xx*xx) and yy<(-b)*np.sqrt(1-xx*xx))) else np.array([0.0000001,0.000001]) for xx in np.linspace(-5,5,20) ] for yy in np.linspace(-5,5,20)])
        polja = np.array([[ghost if (xx<1 and xx>-1 and yy<b*np.sqrt(1-xx*xx) and yy>-b*np.sqrt(1-xx*xx)) else vrednost(xx,yy,nabojcki,plaketke,True) for xx in np.linspace(-5,5,20) ] for yy in np.linspace(-5,5,20)])
        cs = plt.contourf(X,Y,Z,levels=np.linspace(np.amin(Z),np.amax(Z),50),cmap="viridis")
        plt.colorbar(cs)
        plt.quiver(XX,YY,polja[:,:,0],polja[:,:,1])
        plt.fill_between(elipsagor[:,0],elipsagor[:,1],elipsadol[:,1],color="grey")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(r"N=50, Obtekanje okoli NACA krila $t=, v_{\infty} = 1$")
    plt.savefig("druga/elipsa3.png")
if 0:
    #N = 10
    #print(potencialNaItiZici(-1/2+1/(2*N)+0/N,0,N,1))
    plaketke = narediPlaketeTrak(plaketeTrak,71)
    nabojcki = najdiGostotePotencial(plaketke)
    x = np.linspace(-1,1,200)
    y = np.linspace(-1,1,200)
    X,Y = np.meshgrid(x,y)
    XX,YY = np.meshgrid(np.linspace(-1,1,20),np.linspace(-1,1,20))
    Z = [[vrednost(x,y,nabojcki,plaketke) for x in np.linspace(-1,1,200)] for y in np.linspace(-1,1,200)]
    polja = np.array([[vrednost(xx,yy,nabojcki,plaketke,True) for xx in np.linspace(-1,1,20)] for yy in np.linspace(-1,1,20)])
    #print(polja)
    cs = plt.contourf(X,Y,Z,levels=np.linspace(np.min(Z),np.max(Z),50),cmap="hot")
    #cs = plt.contourf(X,Y,Z)
    plt.plot(np.linspace(-0.5,0.5,50),np.zeros(50),color="g")
    plt.quiver(XX,YY,polja[:,:,0],polja[:,:,1])
    plt.colorbar(cs)
    plt.title("N=70")
    #plt.savefig("prva/wow.pdf")
    #nabojcki = naboji(70)
    #plt.plot([i.sredina[0] for i in plaketke],nabojcki,"-o")
    plt.xlabel("x")
    plt.ylabel("y")
    #plt.ylabel(r"$\sigma$")
    #plt.title("Gostota naboja na traku")
    #plt.title("Potencial v okolici nabitega traku, N=2")
    #plt.savefig("prva/naboj.pdf")
    #plt.savefig("prva/wow.png")















"""
def translacija1(x,N,i):
    return x - (-1/2+1/(2*N) + i/N)
def potencialNaIti(xx,y,i,N,sigma):
    #x =xx - (-1/2 + 1/(2*N) + i/N)
    x = translacija1(xx,N,i)
    l = 1/N
    prvi = -l + y*(np.arctan((x+l/2)/y)-np.arctan((x-l/2)/y))
    ostali = (x+l/2)/2*np.log((x+l/2)**2+y*y) - (x-l/2)/2*np.log((x-l/2)**2+y*y)
    #return x+y+1/(2*N)
    return sigma/(2*pi)*(prvi+ostali)
def poljeNaIti(xx,y,i,N,sigma):
    #x = xx-(-1/2 + 1/(2*N)+i/N)
    x = translacija1(xx,N,i)
    l = 1/N
    xmin= x+l/2
    xpl = x-l/2
    prvix = 1/(1+xmin**2) - 1/(1+xpl**2)
    ostalix = 0.5*(np.log((xmin**2+y**2)/(xpl**2+y**2)) + 2*xmin**2/(xmin**2+y**2) - 2*xpl**2/(xpl**2+y**2))
    prviy = np.arctan(xmin/y) - np.arctan(xpl/y) + y*(-xmin/(xmin**2+y**2)+xpl/(xpl**2+y**2))
    ostaliy= y*xmin/(xmin**2+y**2) - y*xpl/(xpl**2+y**2)
    return np.array([-sigma/(2*pi)*(prvix+ostalix),-sigma/(2*pi)*(prviy+ostaliy)])
def potencialNaItiZici(xx,i,N,sigma):
    l = 1/N
    #x =xx - (-1/2 + 1/(2*N)+i/N)
    x = translacija1(xx,N,i)
    #return x + 1/(2*N)
    return sigma/(2*pi)*(-l+(x+l/2)/2*np.log((x+l/2)**2)-(x-l/2)/2*np.log((x-l/2)**2))
def naboji(N):
    U = np.ones(N)
    def naredi(i,j):
        return potencialNaItiZici(-1/2+1/(2*N)+i/N,j,N,1)
    A = np.fromfunction(np.vectorize(naredi),(N,N))
    return A
    return lin.solve(A,U)
def vrednost(x,y,naboji,polje=False):
    rezult = 0
    if polje:
        rezult = np.zeros(2)
    N = len(naboji)
    for i in range(N):
        if polje:
            rezult+=poljeNaIti(x,y,i,N,naboji[i])
            continue
        rezult+= potencialNaIti(x,y,i,N,naboji[i])
    return rezult
    
"""



