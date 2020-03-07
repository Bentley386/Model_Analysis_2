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

def strunaDE(s,y,beta):
    """dy/ds = f(s,y,p) DE za obliko vrvi y = [F,alfa,x,y]"""
    return [-beta*y[2]*np.cos(y[1])+np.sin(y[1]),1/y[0]*(beta*y[2]*np.sin(y[1]) + np.cos(y[1])),np.cos(y[1]),np.sin(y[1])]

def integriraj(x,beta,h,vse = False):
    r = ode(strunaDE).set_integrator("dopri5").set_f_params(beta)
    r.set_initial_value(np.array([x[0],x[1],0,0]),0)
    if not vse:
        return r.integrate(1)
        """
        casi = np.linspace(0,1,int(1/h)+1)
        while r.t < 1:
           y = r.integrate(r.t+h)
        return y        
        """
    else:
        casi = np.linspace(0,1,int(1/h)+1)
        y = np.ones((casi.size,4))
        y[0] = np.array([x[0],x[1],0,0])
        i=1
        while r.t < 1:
           y[i] = r.integrate(r.t+h)
           i+=1
        return y
def streljaj(f0,alfa0,beta,spodnji,h):
    def pomozna(x):
        a = integriraj(x,beta,h)
        return [a[2],a[3]+spodnji]
    param = fsolve(pomozna,[f0,alfa0])
    return param
def resiStruno(f0,alfa0,beta,h,spodnji=1):
    param = streljaj(f0,alfa0,beta,spodnji,h)
    resitev =   integriraj(param,beta,h,vse=True)
    parametri1 = np.matrix(param[0]*np.ones(resitev[:,0].size)).T
    parametri2 = np.matrix(param[1]*np.ones(resitev[:,0].size)).T
    return np.concatenate((resitev,parametri1,parametri2),axis=1)
#a=resiStruno(0.1,0,5.7,0.01,spodnji=0.6)
#plt.plot(a[:,2],a[:,3],".")
    #for i in resitev:
        #resitev[i] = [[resitev[i],param[0]*np.ones(4)],[param[1]]]
    #return np.array(np.concatenate((resitev,parametri1,parametri2),axis=1),dtype=[("f0",float),("alfa0",float),("feg",float),("peder",float),("prvi",float),("drugi",float)])
#podatki = resiStruno(5,pi/8,40,0.01,spodnji=0.8)
#plt.plot(np.linspace(0,1,podatki[:,0].size),podatki[:,0])
def potencialDE(s,y):
    """dy/ds = y = [x,y,v,u]"""
    return [y[2],y[3],-y[0]-2*y[0]*y[1],-y[1]-y[0]*y[0]+y[1]*y[1]]
def integrirajPotencial(y0,v0,u0,maxt,h):
    r = ode(potencialDE).set_integrator("dopri5")
    r.set_initial_value(np.array([0,y0,v0,u0]),0)
    casi = np.linspace(0,maxt,int(maxt/h)+1)
    y = np.ones((casi.size,4))
    y[0] = np.array([0,y0,u0,v0])
    i=1
    while r.t < (maxt-9*h/10):
        try:
            y[i]=r.integrate(r.t+h)
        except:
            return y
        i+=1
    return y
def resiPotencial(y0,u0,E,maxt,h,ejev=False):#x,y,v,u
    if ejev:
        resitev=integrirajPotencial(y0,E,u0,maxt,h)
        return resitev
    try:
        temp=2*(E-(0.5*y0*y0-1/3*y0*y0*y0))-u0*u0
        if temp<0:
            raise NameError("koren od negativnega")
    except:
        print(u0)
    v0 = np.sqrt(temp)

    #print(v0)
    #param = streljajPotencial(y0,v0,u0,maxt,h)
    resitev =   integrirajPotencial(y0,v0,u0,maxt,h)
    #parametri1 = np.matrix(param[0]*np.ones(resitev[:,0].size)).T
    #parametri2 = np.matrix(param[1]*np.ones(resitev[:,0].size)).T
    return resitev
    #return np.concatenate((resitev,parametri1,parametri2),axis=1)
def periodno(resitev,h):
    y0 = resitev[0][1]
    t = int(resitev.size/4)
    for i in range(t):
        if i>4 and np.abs(resitev[i][0])<0.0001 and np.abs(resitev[i][1]-y0)<0.0001:
            print("periodicno")
            return (True,i)
        else:
            return False
def energija(x,y,v,u):
    return 0.5*(v*v+u*u)+0.5*(x*x+y*y)+x*x*y - 1/3*y*y*y
def lyapunov(u0,E,h):
    print("ja")
    startni = resiPotencial(0.1,0.1,0.155,5000,0.05) #s cimer zacnemo
    pogoji1=0 #samo initializacija
    pogoji2=0
    d0 = 0.0001
    print("tle")
    eksponenti=[] #ljapuni
    for i in range(3000,5000):
        if np.abs(startni[:,0][i])<0.001:
            pogoji1=startni[i] #pridemo do atraktorja
            break
    pogoji2=pogoji1+np.array([d0,0,0,0]) #zacetni pogoji za naslednjo iteracijo
    for i in range(8):
        print(i)
        startni1=resiPotencial(pogoji1[1],pogoji1[3],pogoji1[2],5000,0.05,ejev=True)
        for i in range(3000,5000):
            if np.abs(startni1[:,0][i])<0.001:
                pogoji1=startni1[i] #dobimo naslednjo iteracijo
                break
        startni2=resiPotencial(pogoji2[1],pogoji2[3],pogoji2[2],5000,0.05,ejev=True)
        for i in range(3000,5000):
            if np.abs(startni2[:,0][i])<0.001:
                pogoji2=startni2[i] #naslednjo
                break
        d1 = np.linalg.norm(pogoji1[:2]-pogoji2[:2])
        print(d1)
        eksponenti.append(np.log(np.abs(d1/d0)))
        print(eksponenti)
        pogoji2[0] = pogoji1[0]+d0*(pogoji2[0]-pogoji1[0])/d1
        pogoji2[1] = pogoji1[1]+d0*(pogoji2[1]-pogoji1[1])/d1
        d0=d1
    print("haha")
    return eksponenti
#resitev = resiPotencial(0.2,0.1,0.1,400,0.05)
#periodno(resitev,0.05)
eksponenti=lyapunov(0.1,0.155,0.05)
plt.plot(range(len(eksponenti)),eksponenti)
if 0:
    #poincare
    plt.title(r"$y_0=u_0=0.1,E=0.155,|x|<0.001, t=100000$")
    plt.xlabel("y")
    plt.ylabel("u")
    colormap = plt.get_cmap("rainbow")
    barve = np.linspace(0,1,5)
    zacetni = [0.1+0.001*i for i in range(5)]
    for j in range(5):
        resitev = resiPotencial(zacetni[j],0.1,0.155,100000,0.05)
        y = [0.1]
        u = [0.1]
        zadnji = 0
        for i in range(resitev[:,0].size):
            if i-zadnji > 10 and np.abs(resitev[:,0][i])<0.001:
                y.append(resitev[:,1][i])
                u.append(resitev[:,3][i])
                zadnji = i
        plt.plot(y,u,".",color=colormap(barve[j]))
        plt.xlim(-0.5,0.9)
        plt.ylim(-0.6,0.6)
    plt.savefig("druga/poincare2.pdf")
if 0:
    #E0i = np.linspace(0.1,0.16,30)
    #resitev1 = [resiPotencial(0.1,0.1,E0i[t],100,0.05) for t in range(30)]
    #resitev2 = [resiPotencial(0.1,0.1,E0i[t],200,0.05) for t in range(30)]
    #resitev3 = [resiPotencial(0.1,0.1,E0i[t],400,0.05) for t in range(30)]
    def animiraj(t):
        global ax1
        global ax2
        global ax3
        global resitev1
        global resitev2
        global resitev3
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax1.plot(resitev1[t][:,0], resitev1[t][:,1])
        ax1.set_title("t=100")
        ax1.set_aspect(1)
        ax2.plot(resitev2[t][:,0], resitev2[t][:,1])
        ax2.set_title("t=200")
        ax2.set_aspect(1)
        ax3.plot(resitev3[t][:,0], resitev3[t][:,1])
        ax3.set_title("t=400")
        ax3.set_aspect(1)
        plt.suptitle(r"$y_0= u_0 = 0.1, E = {}$".format(str(round(E0i[t],4))),fontsize=30)
    fig, (ax1,ax2,ax3) = plt.subplots(1, 3,figsize=(20,10))
    resitev = resiPotencial(0.1,0.1,0.155,100,0.05)
    ax1.plot(resitev[:,0], resitev[:,1])
    ax1.set_title("t=100")
    ax1.set_aspect(1)
    resitev = resiPotencial(0.1,0.1,0.155,200,0.05)
    ax2.plot(resitev[:,0], resitev[:,1])
    ax2.set_title("t=200")
    ax2.set_aspect(1)
    resitev = resiPotencial(0.1,0.1,0.155,400,0.05)
    ax3.plot(resitev[:,0], resitev[:,1])
    ax3.set_title("t=400")
    ax3.set_aspect(1)    
    plt.suptitle(r"$y_0=u_0=0.1, E=0.155$")
    #ani = animation.FuncAnimation(fig,animiraj,range(0,30),interval=200,save_count=51,repeat=False)
    #plt.show()    
    #ani.save("E0.mp4")
    #plt.subplots_adjust(top=1.4)
    #plt.savefig("druga/kaos.pdf",bbox_inches='tight')
"""
if 0:
    omege = np.array([10+5*i for i in range(13)]) #29
    spodnji = [0.6,0.7,0.8]
    alfe = [0,10,20,30]
    sile = [0.1,1,5,10]
    rezultati = np.array([[[[resiStruno(k,j*pi/180,l,0.01,spodnji=i) for l in omege]for k in sile]for j in alfe] for i in spodnji])
    omega = 0
    alfa = 0
    sila = 0
    spodnja = 0
    xmax = []
    ymin = []
    indeksi = []

    def animiraj(t):
        print(t)
        global ax1
        global ax2
        global ax3
        global ax4
        global rezultati
        global omega
        global alfa
        global sila
        global spodnja
        global xmax
        global ymin
        global indeksi
        napacna = False
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()
        temp = rezultati[spodnja][alfa][sila][omega]
        if temp[0,4]<=0 or temp[0,4]>200:
            napacna = True
        if temp[:,2][np.argmax(np.abs(temp[:,2]))] < 0:
            ax1.plot(-temp[:,2],temp[:,3])
        else:
            ax1.plot(temp[:,2],temp[:,3])
        ax1.set_xlim([-0.4,0.4])
        ax1.set_title(r"$\beta= {0}, F_0^0 = {1}, \alpha_0^0 = {3}, F_0 = {2}, , \alpha_0 = {4}, y_1 = {5}$".format(str(omege[omega]),str(sile[sila]),str(round(temp[0,4],2)),str(alfe[alfa]),str(round(temp[0,5]*180/pi,2)),str(-spodnji[spodnja])),fontsize=30)
        if napacna:
            ax1.annotate("Napačna rešitev", xy=(-0.05,0),xytext=(-0.05,-0.1),fontsize=40,arrowprops=dict(facecolor="black",shrink=0.05))
            indeksi.append(omega)
        ax2.set_title(r"$x_{max}(\sqrt{\beta})$",fontsize=20)
        ax2.set_ylim([0,0.4])
        ax3.set_title(r"$y_{min}(\sqrt{\beta})$",fontsize=20)
        if not napacna:
            xmax.append(np.abs(temp[:,2][np.argmax(np.abs(temp[:,2]))]))
            ymin.append(-np.abs(temp[:,3][np.argmax(-temp[:,3])]))
            if len(indeksi)>0:            
                ax2.plot(np.sqrt(np.delete(omege,indeksi)[:len(xmax)]),xmax,"o",color="magenta")
                ax3.plot(np.sqrt(np.delete(omege,indeksi)[:len(ymin)]),ymin,"o",color="red")
            else:
                ax2.plot(np.sqrt(omege[:len(xmax)]),xmax,"o",color="magenta")
                ax3.plot(np.sqrt(omege[:len(ymin)]),ymin,"o",color="red")
        ax2.set_xlim([3,9])
        ax3.set_xlim([3,9])
        ax4.set_title(r"$F(s)$",fontsize=20)
        ax4.plot(np.linspace(0,1,temp[:,0].size),temp[:,0],color="green")
        omega+=1
        if omega>12:
            xmax=[]
            ymin=[]
            omega = 0
            indeksi = []
            sila+=1
        if sila>3:
            sila = 0
            alfa+=1
        if alfa>3:
            alfa = 0
            spodnja+=1
        if spodnja>2:
            print("wtf")

    
    fig = plt.figure(figsize=(20,20))
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2,rowspan=3) #glavni Ax
    ax2 = plt.subplot2grid((3, 3), (0, 2)) 
    ax3 = plt.subplot2grid((3, 3), (1, 2)) 
    ax4 = plt.subplot2grid((3, 3), (2, 2)) 
    #gs = gridspec.GridSpec()
    #fig, ax = plt.subplots(2)
    #fig,ax1 = plt.subplots()
    ani = animation.FuncAnimation(fig,animiraj,range(0,12*4*4*3-2),interval=400)    
    ani.save("druga.mp4")
if 0:
    colormap = plt.get_cmap("magma")
    omege = (10,20,30,40,50,60,70,80)
    barve = np.linspace(0.1,0.8,len(omege))
    for i in range(len(omege)):
        podatki = resiStruno(0.2,pi/8,omege[i],0.01,spodnji=0.8)
        if podatki[:,2][np.argmax(np.abs(podatki[:,2]))] < 0:
            podatki[:,2] = -podatki[:,2]
        plt.plot(podatki[:,2],podatki[:,3],label=r"$\beta = {}$".format(str(omege[i])),color=colormap(barve[i]))
    plt.legend(loc="best")
    plt.title(r"Rešitev za več $\beta$ pri $F_0^0 =0.2, \alpha_0^0=pi/8$, spodnji rob je pri -0.8")
    #plt.savefig("prva/vecbet4.pdf")

#plan:

"""







