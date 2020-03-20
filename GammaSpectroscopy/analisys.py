import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as fit
import mplhep
from sklearn.linear_model import LinearRegression
plt.style.use(mplhep.style.ATLAS)
plt.rc('text',usetex=True)

filename = "data/run020752.dat"
t,a,b,c,d = np.loadtxt(filename).T

data = a
data = data[data<3000]


def gaus(x, A, mu, sigma,m,q):
    return m*x+q+A*np.exp(-(x-mu)**2/(2*sigma**2))

def get_fit_inrange(binc,y,inf,sup,plot=False):
    entries = np.sum(y)
    x = binc[np.where((binc>inf)&(binc<sup))]
    y = y[np.where((binc>inf)&(binc<sup))]
    s = np.sqrt(y*(1-y/entries))
    mean = np.dot(y,x)/np.sum(y)
    std = 20 # np.std(data)
    A = np.max(y)
    par,cov = fit(gaus,x,y,p0=[A,mean,std,0,0],sigma=s,absolute_sigma=True)
    if plot:
        xfit = np.linspace(np.min(x),np.max(x),500)
        ax.plot(x,y,'.b',label='Data')
        ax.plot(xfit,gaus(xfit,*par),'r',label='Fit')
        ax.set_xlabel('ADC')
        ax.set_ylabel('Entries')
        ax.legend(loc=1,frameon=True,fancybox=False,framealpha=1,edgecolor='k')
        plt.draw()
        plt.pause(0.5)
        plt.cla()
    N = x.size
    return par, cov, N

nbins = 500
hist,bins = np.histogram(data,nbins)
binc = bins[1:]-(bins[1]-bins[0])/2


range_inf = [250,480,600,720,820,1300]
range_sup = [340,560,680,800,940,1500]
energies = [77,185,241,295,351,609]

peak_ADC = []
std_ADC = []
err_ADC = []
par = []
n = []
fig,ax = plt.subplots(1,1,figsize=(8,8))
for i,(I,S) in enumerate(zip(range_inf,range_sup)):
    p,c,N = get_fit_inrange(binc,hist,I,S,False)
    par.append(p)
    peak_ADC.append(p[1])
    std_ADC.append(p[2])
    err = np.sqrt(np.diag(c))
    err_ADC.append(err[1])
    n.append(N)
plt.close(fig)



fig,ax = plt.subplots(1,1,figsize=(16,8))
binsize = np.abs(binc[1]-binc[0])
ax.step(binc,hist,'b',label='Data')
ax.set_xlabel('ADC')
ax.set_ylabel('Entries')
ax.set_ylim(0,)
for i,(I,S) in enumerate(zip(range_inf,range_sup)):
    x = np.linspace(I,S,100)
    string = f'Mean: {par[i][1]:.2f} Std: {par[i][2]:.2f}'
    ax.plot(x,gaus(x,*par[i]),'r',label=string,lw=2)
    ax.legend(loc=1,frameon=True,fancybox=False,framealpha=1,edgecolor='k')

n = np.array(n)
energies = np.array(energies)
peak_ADC = np.array(peak_ADC)
err_ADC = np.array(err_ADC)
std_ADC = np.array(std_ADC)#/np.sqrt(n)
weights = 1/std_ADC**2

def get_line(x,m,q):
    return x*m+q

par,cov = fit(get_line,energies,peak_ADC,sigma=std_ADC,absolute_sigma=True)
err = np.sqrt(np.diag(cov))

plt.figure()
plt.errorbar(energies,peak_ADC,std_ADC,fmt='.b',capsize=2,label='Data')
x_to_fit = np.linspace(50,650,500)
plt.plot(x_to_fit,get_line(x_to_fit,*par),'r',label='Linear Regression')
plt.xlabel('Energy [KeV]')
plt.ylabel('ADC')

m = par[0]
q = par[1]
errm = err[0]
errq = err[1]

string =    f'''{m:.2f}$\pm${errm:.2f} ADC per KeV
            {q:.2f}$\pm${errq:.2f} ADC at 0 KeV'''
plt.text(0.2,0.8,string,transform=plt.gca().transAxes)




plt.show()
