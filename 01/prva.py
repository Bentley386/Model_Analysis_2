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
def mod_euler(p0,q0,fi,korak,cas): #fi je gradient od fi tukaj
    tajm = 0
    p = [p0]
    q = [q0]
    while tajm<cas:
        tajm+= korak
        p.append(p[-1] - korak*fi(q[-1]))
        q.append(q[-1] + korak*p[-1])
    return (np.array([i*korak for i in range(int(cas/korak)+1)]),np.array(p),np.array(q))        
    #return (np.arange(0,cas+korak,korak),np.array(p),np.array(q))
def leapfrog(p0,q0,fi,korak,cas): #fi je gradient od fi tukaj
    tajm = 0
    p = [p0]
    q = [q0]
    while tajm<cas:
        tajm+= korak
        qpol = q[-1] + korak/2 * p[-1]
        p.append(p[-1] - korak*fi(qpol))
        q.append(qpol + korak/2*p[-1])
    return (np.array([i*korak for i in range(int(cas/korak)+1)]),np.array(p),np.array(q))        
    #return (np.arange(0,cas+korak,korak),np.array(p),np.array(q))
def gradfi(x):
    return x/(np.linalg.norm(x)**3)
def energija(x):
    return [np.abs((0.5*np.linalg.norm(x[1][i])**2 - np.linalg.norm(x[2][i])**(-1)+0.5)/0.5) for i in range(int(x[1].size/2))]    
    
podatki = mod_euler(np.array([0,1]),np.array([1,0]),gradfi,0.01,20)
plt.plot(podatki[0],energija(podatki))
podatki = leapfrog(np.array([0,1]),np.array([1,0]),gradfi,0.01,20)
plt.plot(podatki[0],energija(podatki))
#plt.yscale("log")
def de(t,y): #y = [x,y,u,v]
    return [y[2],y[3],-y[0]/((y[0]**2+y[1]**2)**1.5),-y[1]/((y[0]**2+y[1]**2)**1.5)]
    #return [y[2],y[3],-y[0],-y[1]]
def rungekutta4(p0,q0,korak,cas):
    r = ode(de)
    r.set_integrator("dopri5")
    r.set_initial_value(np.concatenate([q0,p0]),0)
    #y = np.ones((int(cas/korak)+1,4))
    #y[0] = np.concatenate([q0,p0])
    y = [np.concatenate([q0,p0])]
    #i = 1
    while r.t < cas:
        #y[i] = r.integrate(r.t+korak)
        #i+=1
        y.append(r.integrate(r.t+korak))
    #print(y[-1])
    return (np.array([i*korak for i in range(int(cas/korak)+1)]),np.array(y))        
    #return (np.arange(0,cas+korak,korak),np.array(y))
def fi(x):
    return -np.linalg.norm(x)**(-1)

def obhodni(t,q):
    obhodni = 0
    indeks = 0
    for kraj in range(t.size):
        if np.abs(q[kraj][0]-1)<0.01 and np.abs(q[kraj][1])<0.01 and kraj > 30:
            obhodni = t[kraj]
            indeks = kraj
            break
    polos = (1- (np.amin(q[:,0][:indeks])))/2
    return np.array([obhodni,polos])
def energija(p,q):
    return 0.5*(p[0]**2 + p[1]**2) - np.linalg.norm(q)**(-1)
def gradfi2(x,t,v0=2,a=1,x0 =-11):
    return x/(np.linalg.norm(x)**3) + np.array([x[0]-(x0 +v0*t),x[1]-1.5*a])/((x[1]-1.5*a)**2 + (x[0]-(x0 +v0*t))**2)**(3/2)

def leapfrog2(p0,q0,fi,korak,cas,v0=2): #fi je gradient od fi tukaj
    tajm = 0
    p = np.ones((int(cas/korak)+1,2))
    q = np.ones((int(cas/korak)+1,2))
    p[0] = p0
    q[0] = q0
    #p = [p0]
    #q = [q0]
    i=0
    while tajm<(cas-korak/2):
        tajm+= korak
        #qpol = q[-1] + korak/2 * p[-1]
        qpol = q[i]+korak/2*p[i]
        p[i+1] = p[i] - korak*fi(qpol,tajm,v0)
        q[i+1] = qpol + korak/2*p[i+1]
        i+=1
        #p.append(p[-1] - korak*fi(qpol,tajm))
        #q.append(qpol + korak/2*p[-1])
    return (np.array([i*korak for i in range(int(cas/korak)+1)]),np.array(p),np.array(q))        
    #return (np.arange(0,cas+korak,korak),np.array(p),np.array(q))        
def potenergija(q1,q2):
    return -1/(((q1[0]-q2[0])**2 + (q1[1]-q2[1])**2)**(0.5))
def gradfi3(q1,q2):
    return (q1-q2)/(((q1[0]-q2[0])**2 + (q1[1]-q2[1])**2)**(1.5))
def leapfrog3(p0,q0,p1,q1,korak,cas): #fi je gradient od fi tukaj
    tajm = 0
    p = np.ones((int(cas/korak)+1,2))
    q = np.ones((int(cas/korak)+1,2))
    pp = np.ones((int(cas/korak)+1,2))
    qq = np.ones((int(cas/korak)+1,2)) 
    p[0] = p0
    q[0] = q0
    pp[0] = p1
    qq[0] = q1
    #p = [p0]
    #q = [q0]
    i=0
    while tajm<(cas-korak/2):
        tajm+= korak
        #qpol = q[-1] + korak/2 * p[-1]
        qpol = q[i]+korak/2*p[i]
        qqpol = qq[i]+korak/2*pp[i]
        p[i+1] = p[i] - korak*gradfi3(qpol,qqpol)
        pp[i+1] = pp[i] - korak*gradfi3(qqpol,qpol)
        q[i+1] = qpol + korak/2*p[i+1]
        qq[i+1] = qqpol + korak/2*pp[i+1]
        i+=1
        #p.append(p[-1] - korak*fi(qpol,tajm))
        #q.append(qpol + korak/2*p[-1])
    return (np.array([i*korak for i in range(int(cas/korak)+1)]),np.array(p),np.array(q),np.array(pp),np.array(qq))        
    #return (np.arange(0,cas+korak,korak)
def leapfrog4(p0,q0,p1,q1,p2,q2,korak,cas): #fi je gradient od fi tukaj
    tajm = 0
    p = np.ones((int(cas/korak)+1,2))
    q = np.ones((int(cas/korak)+1,2))
    pp = np.ones((int(cas/korak)+1,2))
    qq = np.ones((int(cas/korak)+1,2)) 
    ppp = np.ones((int(cas/korak)+1,2))
    qqq = np.ones((int(cas/korak)+1,2)) 
    p[0] = p0
    q[0] = q0
    pp[0] = p1
    qq[0] = q1
    ppp[0] = p2
    qqq[0] = q2
    #p = [p0]
    #q = [q0]
    i=0
    while tajm<(cas-korak/2):
        tajm+= korak
        #qpol = q[-1] + korak/2 * p[-1]
        qpol = q[i]+korak/2*p[i]
        qqpol = qq[i]+korak/2*pp[i]
        qqqpol = qqq[i]+korak/2*ppp[i]
        p[i+1] = p[i] - korak*(gradfi3(qpol,qqpol)+gradfi3(qpol,qqqpol))
        pp[i+1] = pp[i] - korak*(gradfi3(qqpol,qpol)+gradfi3(qqpol,qqqpol))
        ppp[i+1] = ppp[i] - korak*(gradfi3(qqqpol,qpol)+gradfi3(qqqpol,qqpol))
        q[i+1] = qpol + korak/2*p[i+1]
        qq[i+1] = qqpol + korak/2*pp[i+1]
        qqq[i+1] = qqqpol + korak/2*ppp[i+1]
        i+=1
        #p.append(p[-1] - korak*fi(qpol,tajm))
        #q.append(qpol + korak/2*p[-1])
    return (np.array([i*korak for i in range(int(cas/korak)+1)]),np.array(p),np.array(q),np.array(pp),np.array(qq),np.array(ppp),np.array(qqq))        
    #return (np.arange(0,cas+korak,korak)
#zacetna = leapfrog2(-np.array([-np.sin(40*pi/180),np.cos(40*pi/180)]),np.array([np.cos(40*pi/180),np.sin(40*pi/180)]),gradfi2,0.1,44,1.1)
#sonce = -11+1.1*np.linspace(0,44,int(44/0.1)+1)


#podatki = leapfrog4(np.array([0,0.2]),np.array([1,0]),np.array([0,-0.2]),np.array([-1,0]),np.array([0,-1]),np.array([3,3]),0.05,100)
#kineticne1 = 0.5*(podatki[1][:,0]**2 + podatki[1][:,1]**2)
#kineticne2 = 0.5*(podatki[3][:,0]**2 + podatki[3][:,1]**2)
#kineticne3 = 0.5*(podatki[5][:,0]**2 + podatki[5][:,1]**2)
#potencialne = [potenergija(podatki[2][i],podatki[4][i])+potenergija(podatki[2][i],podatki[6][i])+potenergija(podatki[4][i],podatki[6][i]) for i in range(int(podatki[2].size/2))]
#prva = podatki[2]
#druga = podatki[4]
#tretja= podatki[6]
def animiraj(t):
    global ax1
    global ax2
    global podatki
    ax1.clear()
    ax2.clear()
    if t!=0:
        ax1.plot(podatki[2][:t][:,0],podatki[2][:t][:,1],color="blue")
        ax1.plot(podatki[4][:t][:,0],podatki[4][:t][:,1],color="red")
        ax1.plot(podatki[6][:t][:,0],podatki[6][:t][:,1],color="orange")
    #ax1.set_xlim(-5,5)
    #ax1.set_ylim(-1.5,2.5)
    ax1.plot(podatki[2][t][0],podatki[2][t][1],"o",color="blue")
    ax1.plot(podatki[4][t][0],podatki[4][t][1],"o",color="red")
    ax1.plot(podatki[6][t][0],podatki[6][t][1],"o",color="orange")
    if t>0:
        ax2.set_title("E={}".format(str(round(kineticne1[t-1]+kineticne2[t-1]+kineticne3[t-1]+potencialne[t-1],2))))
        ax2.plot(range(t),kineticne1[:t],color="blue",label="Kin. prve")
        ax2.plot(range(t),kineticne2[:t],color="red",label="Kin. druge")
        ax2.plot(range(t),kineticne3[:t],color="orange",label="Kin. tretje")
        ax2.plot(range(t),potencialne[:t],color="magenta",label="Skupna pot.")
        ax2.legend(loc="lower left",fontsize="large")
    #ax1.set_xlim(-1.2,1.2)
    #ax1.set_ylim(-1.2,1.2)
    #ax1.plot([0],[0],"yo",markersize=5)
    #ax1.plot(sonce[t],1.5*np.ones(1),"yo",markersize=5)

#fig = plt.figure(figsize=(20,20))
#ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3,rowspan=2)
#ax2 = plt.subplot2grid((3, 3), (2, 0),colspan=3)
#fig = plt.figure()
#gs = gridspec.GridSpec()
#fig, ax = plt.subplots(2)
#fig,ax1 = plt.subplots()
#ani = animation.FuncAnimation(fig,animiraj,range(0,2000),interval=100)    
#plt.show()
#ani.save("neki3.mp4")

"""
podatki = leapfrog3(np.array([0,0.2]),np.array([1,0]),np.array([0,-0.2]),np.array([-1,0]),0.05,50)
prva = podatki[2]
druga = podatki[4]

def animiraj(t):
    global ax1
    global prva
    global druga
    ax1.clear()
    if t!=0:
        ax1.plot(prva[:t][:,0],prva[:t][:,1],color="blue")
        ax1.plot(druga[:t][:,0],druga[:t][:,1],color="red")
    #ax1.set_xlim(-5,5)
    #ax1.set_ylim(-1.5,2.5)
    ax1.plot(prva[t][0],prva[t][1],"o",color="blue")
    ax1.plot(druga[t][0],druga[t][1],"o",color="red")
    ax1.set_xlim(-1.2,1.2)
    ax1.set_ylim(-1.2,1.2)
    #ax1.plot([0],[0],"yo",markersize=5)
    #ax1.plot(sonce[t],1.5*np.ones(1),"yo",markersize=5)
    plt.suptitle("Binarna zvezda")
fig, ax1 = plt.subplots()
#fig,ax1 = plt.subplots()
ani = animation.FuncAnimation(fig,animiraj,range(0,1000),interval=100)    
#plt.show()
ani.save("binarna.mp4")

ORBITE
def animiraj(t):
    global ax1
    global zacetna
    global sonce
    ax1.clear()
    if t!=0:
        ax1.plot(zacetna[2][:t][:,0],zacetna[2][:t][:,1],color="blue")
        ax1.plot(sonce[:t],1.5*np.ones(t),"y")
    #ax1.set_xlim(-5,5)
    #ax1.set_ylim(-1.5,2.5)
    ax1.plot(zacetna[2][t][0],zacetna[2][t][1],"o",color="blue")
    ax1.plot([0],[0],"yo",markersize=5)
    ax1.plot(sonce[t],1.5*np.ones(1),"yo",markersize=5)
    plt.suptitle("Stabilna okoli drugega sonca")
#fig, ax1 = plt.subplots()
#fig,ax1 = plt.subplots()
#ani = animation.FuncAnimation(fig,animiraj,range(0,440),interval=100)    
#plt.show()
#ani.save("stabilna2.mp4")

hitrost = 0.2
kot = 0
podatek = leapfrog3(np.array([-hitrost*np.sin(kot*pi/180),hitrost*np.cos(kot*pi/180)]),np.array([1,0]),np.array([hitrost*np.sin(kot*pi/180),
                             hitrost*np.cos(kot*pi/180)]),np.array([-1,0]),0.02,40) #cas , y(q,p)
plt.plot(podatek[2][:,0],podatek[2][:,1])
plt.plot(podatek[4][:,0],podatek[4][:,1])
"""
if 0:
    #klasifikacija orbit za binary kurac
    hitrosti = np.linspace(0.1,1,20)
    smeri = np.linspace(0,80,20)
    #plt.plot([0],[0],"yo",markersize=5)
    #faza = 180
    #casi = np.linspace(0,1000,int(1000/0.005)+1)
    #plt.plot(-11+2*casi,np.ones(casi.size)*1.5)
    #v0=

    def orbita(kot,hitrost):
        podatek = leapfrog3(np.array([-hitrost*np.sin(kot*pi/180),hitrost*np.cos(kot*pi/180)]),np.array([1,0]),np.array([hitrost*np.sin(kot*pi/180),-hitrost*np.cos(kot*pi/180)]),np.array([-1,0]),0.02,200) #cas , y(q,p)
        razdalja = np.sqrt(podatek[2][-1][0]**2 + podatek[2][-1][1]**2)
        if razdalja > 20:
            return [1.0,0,0]
        else:
            return [0,1.0,0]
    orbite = [[orbita(i,j) for i in smeri] for j in hitrosti]
    plt.imshow(orbite)
    plt.xticks(np.linspace(0,20,10),[str(round(i,1)) for i in np.linspace(0,80,10)])
    plt.yticks(np.linspace(0,20,10),[str(round(i,1)) for i in np.linspace(0.1,1,10)])
    plt.xlabel("Faza [°]")
    plt.ylabel(r"$|v_0|$")
    plt.savefig("prva/birnani.png")
    #podatek = leapfrog2(np.array([-v0*np.sin(faza*pi/180),v0*np.cos(faza*pi/180)]),np.array([np.cos(faza*pi/180),np.sin(faza*pi/180)]),gradfi2,0.005,1000)
    #plt.plot(podatek[2][:,0],podatek[2][:,1])
    #plt.axes().set_aspect("equal","datalim")
if 0:
    #klasifikacija orbit za mimobezno
    hitrosti = np.linspace(1,3,20)
    faze = np.linspace(0,360,400)
    #plt.plot([0],[0],"yo",markersize=5)
    #faza = 180
    #casi = np.linspace(0,1000,int(1000/0.005)+1)
    #plt.plot(-11+2*casi,np.ones(casi.size)*1.5)
    #v0=1
    def orbita(kot,hitrost):
        podatek = leapfrog2(-np.array([-np.sin(kot*pi/180),np.cos(kot*pi/180)]),np.array([np.cos(kot*pi/180),np.sin(kot*pi/180)]),gradfi2,0.01,int(22/hitrost),v0=hitrost) #cas , y(q,p)
        kineticna = (podatek[1][-1][0]**2 + podatek[1][-1][1]**2)/2
        e1 = kineticna + potenergija(podatek[2][-1],[0,0]) #z mirujocim
        e2 = kineticna + potenergija(podatek[2][-1],[-11+hitrost*int(22/hitrost),1.5])
        
        if kineticna+e1+e2 > 0:
            return [1.0,0,0]
        elif e1<e2:
            return [0,1.0,0]
        else:
            return [0,0,1.0]
        """
        if e1<0 and e2>0:
            #print("aha")
            return [0,1.0,0]
        elif e1>0 and e2>0:
            #print("hm")
            return [1.0,0,0]
        elif e1>0 and e2<0:
            print("ok")
            return [0,0,1.0]
        else:
            if e1<e2:
                return [0,1.0,0]
            else:
                return [0,0,1.0]
        """
    orbite = [[orbita(i,j) for i in faze] for j in hitrosti]
    slika = []
    for i in orbite:
        for j in range(20):
            slika.append(i)
    plt.imshow(slika)
    plt.xticks(np.linspace(0,400,10),[str(round(i,1)) for i in np.linspace(0,360,10)])
    plt.yticks(np.linspace(0,400,10),[str(round(i,1)) for i in np.linspace(1,3,10)])
    plt.xlabel("Faza [°]")
    plt.ylabel(r"$|v_0|$")
    plt.savefig("prva/rezimi2.png")
    #podatek = leapfrog2(np.array([-v0*np.sin(faza*pi/180),v0*np.cos(faza*pi/180)]),np.array([np.cos(faza*pi/180),np.sin(faza*pi/180)]),gradfi2,0.005,1000)
    #plt.plot(podatek[2][:,0],podatek[2][:,1])
    #plt.axes().set_aspect("equal","datalim")
if 0:
    #klasifikacija orbit
    fiji = np.linspace(0,80,41)
    hitrosti = np.linspace(0.2,2,41)
    def orbita(kot,hitrost):
        podatek = leapfrog(np.array([hitrost*np.sin(kot*pi/180),hitrost*np.cos(kot*pi/180)]),np.array([1,0]),gradfi,0.005,200) #cas , y(q,p)
        razdalja = podatek[2][-1][0]**2 + podatek[2][-1][1]**2
        if razdalja > 100:
            return [1.0,0,0]
        else:
            return [0,1.0,0]
    orbite = [[orbita(i,j) for i in fiji] for j in hitrosti]
    #fiji = list(range(6))
    #hitrosti = list(range(16))
    def barva(i,j):
        if (-1)**(i+j)==1:
            return [1.0,0,0]
        elif (-1)**(i+j) == -1:
            return [0,1.0,0]
    #orbite = [[barva(i,j) for i in fiji] for j in hitrosti]
    plt.imshow(orbite)
    plt.xticks([0,10,20,30,40],["0","19.5","39","58.5","78"])
    plt.yticks([0,5,10,15,20,25,30,35,40],["0.2","0.41","0.63","0.86","1.08","1.3","1.52","1.74","1.96"])
    plt.xlabel("Kot [°]")
    plt.ylabel(r"$|v_0|$")
    plt.savefig("prva/stabilnost2.png")
if 0:
    #energija + vritlna + rungelenz
    podatek = leapfrog(np.array([0,1]),np.array([1,0]),gradfi,0.01,1000) #cas , p , q
    energije = [0.5*(podatek[1][i][0]**2 + podatek[1][i][1]**2) - np.linalg.norm(podatek[2][i])**(-1) for i in range(int(podatek[1].size/2))]
    vrtilne = [podatek[2][i][0]*podatek[1][i][1] - podatek[2][i][1]*podatek[1][i][0] for i in range(int(podatek[1].size/2))] #x py - y px
    rungex = [podatek[1][i][1]*vrtilne[i] - podatek[2][i][0]/np.linalg.norm(podatek[2][i]) for i in range(int(podatek[1].size/2))]
    rungey = [-podatek[1][i][0]*vrtilne[i] - podatek[2][i][1]/np.linalg.norm(podatek[2][i]) for i in range(int(podatek[1].size/2))]
    plt.plot(podatek[0],energije[:-1],label="E")
    plt.plot(podatek[0],vrtilne[:-1],label="L")
    plt.plot(podatek[0],rungex[:-1],label=r"$A_x$")
    plt.plot(podatek[0],rungey[:-1],label=r"$A_y$")
    plt.title("Ohranjene količine")
    plt.xlabel("t")
    plt.legend(loc="best")
    plt.savefig("prva/ohranjene.pdf")

if 0:
    #stabilnost
    if 1:
        #podatek = rungekutta4(np.array([0,1]),np.array([1,0]),0.1,100000) #cas , y(q,p) 
        #energije = [0.5*(podatek[1][i][2]**2 + podatek[1][i][3]**2) - (podatek[1][i][0]**2 + podatek[1][i][1]**2)**(-0.5) for i in range(int(podatek[1].size/4))]
        #plt.plot(podatek[0],np.abs((np.array(energije)+0.5)/(-0.5)),label="Runge Kutta")
        #podatek = leapfrog(np.array([0,1]),np.array([1,0]),gradfi,0.1,100000) #cas , p , q 
        #energije = [0.5*(podatek[1][i][0]**2 + podatek[1][i][1]**2) - np.linalg.norm(podatek[2][i])**(-1) for i in range(int(podatek[1].size/2))]
        #plt.plot(podatek[0],np.abs((np.array(energije)+0.5)/(-0.5)),label="Leapfrog")
        podatek = mod_euler(np.array([0,1]),np.array([1,0]),gradfi,0.1,500) #cas , p , q 
        energije = [0.5*(podatek[1][i][0]**2 + podatek[1][i][1]**2) - np.linalg.norm(podatek[2][i])**(-1) for i in range(int(podatek[1].size/2))]
        plt.plot(podatek[0],np.abs((np.array(energije)+0.5)/(-0.5)),label="Modificiran Euler")

        #plt.legend(loc="best")
        #plt.title(r"Napaka energije za $v_0 = (0,1), h=0.1$")
        #plt.yscale("log")
        #plt.ylabel(r"$\Delta E / E_0$")
        #plt.xlabel("Cas")
        #plt.savefig("prva/et.pdf")
    if 1:
        #v odv od h
        hji = np.linspace(0.001,0.5,50)
        e0 = -0.5
        napake = []
        for i in hji:
            podatek = leapfrog(np.array([0,1]),np.array([1,0]),gradfi,i,1000)
            temp = energija(podatek[1][-1],podatek[2][-1])
            napake.append(np.abs(temp-e0)/np.abs(e0))
        plt.plot(hji,napake,label="Leapfrog")
        napake = []
        for i in hji:
            podatek = rungekutta4(np.array([0,1]),np.array([1,0]),i,1000)
            temp = energija(podatek[1][-1][2:],podatek[1][-1][:2])
            napake.append(np.abs(temp-e0)/np.abs(e0))
        plt.plot(hji,napake,label="Runge Kutta")
        plt.legend(loc="best")
        plt.title("Relativna napaka energije po t=1000 v odvisnosti od velikosti koraka")
        plt.ylabel(r"$\Delta E/E_0$")
        plt.xlabel("h")
        plt.yscale("log")
        plt.savefig("prva/eh.pdf")
    if 1:
        #v odv od v0
        v0 = np.linspace(0.5,1.2,25)
        napake = []
        for i in v0:
            e0 = i**2 / 2 - 1
            podatek = leapfrog(np.array([0,i]),np.array([1,0]),gradfi,0.01,1000)
            temp = energija(podatek[1][-1],podatek[2][-1])
            napake.append(np.abs(temp-e0)/np.abs(e0))
        plt.plot(v0,napake,label="Leapfrog")
        napake = []
        for i in v0:
            e0 = i**2 /2 -1
            podatek = rungekutta4(np.array([0,i]),np.array([1,0]),0.01,1000)
            temp = energija(podatek[1][-1][2:],podatek[1][-1][:2])
            napake.append(np.abs(temp-e0)/np.abs(e0))
        plt.plot(v0,napake,label="Runge Kutta")
        plt.legend(loc="best")
        plt.yscale("log")
        plt.title("Relativna napaka energije po t=1000 v odvisnosti od velikosti začetne hitrosti, h=0.01")
        plt.ylabel(r"$\Delta E/E_0$")
        plt.xlabel(r"$v_0$")
        plt.savefig("prva/ev.pdf")        
    if 1:
        #v odv od kota
        alfa = np.linspace(20,70,20)
        e0 = -0.5
        napake = []
        for i in alfa:
            podatek = leapfrog(np.array([-np.sin(i*pi/180),np.cos(i*pi/180)]),np.array([1,0]),gradfi,0.01,1000)
            temp = energija(podatek[1][-1],podatek[2][-1])
            napake.append(np.abs(temp-e0)/np.abs(e0))
        plt.plot(alfa,napake,label="Leapfrog")
        napake = []
        for i in alfa:
            podatek = rungekutta4(np.array([-np.sin(i*pi/180),np.cos(i*pi/180)]),np.array([1,0]),0.01,1000)
            temp = energija(podatek[1][-1][2:],podatek[1][-1][:2])
            napake.append(np.abs(temp-e0)/np.abs(e0))
        plt.plot(alfa,napake,label="Runge Kutta")
        plt.title("Relativna napaka energije po t=1000 v odvisnosti od kota zacetne hitrosti, h=0.01")
        plt.yscale("log")
        plt.ylabel(r"$\Delta E/E_0$")
        plt.xlabel(r"$\alpha$")
        plt.legend(loc="best")
        #plt.savefig("prva/ef.pdf")
if 0:
    #keplerjev 3. zakon
    podatki = []
    for v0 in np.arange(0.5,1.2,0.05):
        fajl = leapfrog(np.array([0,v0]),np.array([1,0]),gradfi,0.005,100)
        podatki.append(obhodni(fajl[0],fajl[2]))
    podatki= np.array(podatki)
    plt.plot(podatki[:,1],podatki[:,0],"ko")
    plt.plot(np.linspace(0.5,1.5),2*pi*np.linspace(0.5,1.5)**1.5,"b")
    plt.title("Odvisnost obhodnega časa od velike polosi eliptične orbite")
    plt.xlabel("Velika polos")
    plt.ylabel("Obhodni čas")
    plt.savefig("prva/kepler.pdf")

if 0:
    #orbite
    cm = plt.get_cmap("copper")
    plt.plot([0],[0],"yo",markersize=5)
    for alfa in np.arange(20,80,10):
        fajl = leapfrog(np.array([-1.5*np.sin(alfa*pi/180),1.5*np.cos(alfa*pi/180)]),np.array([1,0]),gradfi,0.001,50)
        plt.plot(fajl[2][:,0],fajl[2][:,1],color=cm((alfa-10)/90),label=r"$\alpha = {}$".format(str(alfa)))
        plt.xlim(-5,5)
        plt.ylim(-5,5)
    plt.legend(loc="best")
    plt.title(r"Trajektorije za več različnih smeri začetnih hitrosti $|v_0|=1.5$")
    plt.savefig("prva/modeuler.pdf")
    plt.show()






















