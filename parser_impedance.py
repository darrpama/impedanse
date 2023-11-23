import scipy.optimize as optimization
import matplotlib.pyplot as plt
import matplotlib
import math
import numpy

with open ("data.dat","r", errors = 'replace') as f:
  lines = f.readlines()
  print(len(lines))

n=0
LL,RR = 3,4

def parser(n):
    F, T, e1, e2, s1, s2, S, Z, Zre, Zim = [], [], [], [], [], [], [], [], [], []
    for index in range(3+49*n,51+49*n):
        values = lines[index].split("\t")
        F.append(float(values[0]))
        T.append(float(values[1]))
        e1.append(float(values[2]))
        e2.append(float(values[3]))
        s1.append(float(values[4]))
        s2.append(float(values[5]))
        S.append(complex(float(values[4]), float(values[5])))
        ZZ = 1e-6/complex(float(values[4]), float(values[5]))
        Z.append(ZZ)
        Zre.append(ZZ.real)
        Zim.append(ZZ.imag)
    for k in range(1,47):
        if Zim[k]>Zim[k-1] and Zim[k]>Zim[k+1]:
            break
    R = Zim[k]
    F0 = F[k]

    print ('Z\" = ',R,"M\u03A9, Fmax = ", F0, 'Hz, T = ', T[0],'C')    
    return F, T, e1, e2, s1, s2, S, Z, Zre, Zim, R, F0


def model (x,R,a,b):
    y = 0
    if x<2*R:
        y=(R**2-(x-R)**2)**0.5
    if x > 2.0*R and a*x + b > 0 :
        y+= a*x + b
    return y

def model1(x,R):
    y = 0
    y = (R**2-(x-R)**2)**0.5
    return y

def model2 (x,a,b):
    return a*x + b



def findInitParamsForLine(Zre, Zim, R):
  xmin, xmax, ymin, ymax = 0, 0, 0, 0
  fmin, fmax = 0, 0
  for j in range (0,47):
      if Zre[j] < RR*R and Zre[j+1] >= RR*R:
          xmax, ymax = Zre[j], Zim[j]
          fmax = F[j]
      if Zre[j]< LL*R and Zre[j+1] >= LL*R:
          xmin, ymin = Zre [j], Zim[j]
          fmin = F[j]
  A = (ymax-ymin)/(xmax-xmin)
  B = ymin - A*xmin     
  print('a = ', A, 'b = ', B, 'R = ', R)
  print('fmin, fmax ',fmin, fmax)
  return A,B


def GetFittedR(Zre,Zim, Rinit):
  xdata, ydata = [], []
  for kk in range(0, 48):
      if Zre[kk]<Rinit:
          xdata.append(Zre[kk])
          ydata.append(Zim[kk])
  res = optimization.curve_fit(model1,xdata, ydata, [Rinit])
  print('R after fit:', res[0][0])
  R = res[0][0]
  return R

def GetFittedAB(Zre,Zim):
  xdata, ydata = [], []
  for kk in range(0, 48):
      if Zre[kk]>LL*R and Zre[kk]<RR*R:
          xdata.append(Zre[kk])
          ydata.append(Zim[kk])
  res = optimization.curve_fit(model2, xdata, ydata, [A,B])
  print('A, B after fit:', res[0][0], res[0][1])
  A, B = res[0][0], res[0][1]
  return A, B


F, T, e1, e2, s1, s2, S, Z, Zre, Zim, R, F0 = parser(n)

xmin,xmax,ymin,ymax = 0, 0, 0, 0
fmin,fmax = 0, 0
A,B = findInitParamsForLine(Zre,Zim,R)
print(A,B)

M = []

for kk in range(0, 48):
    M.append(model(Zre[kk], R, A, B))
    


Trevarr, Tarr, Farr, Rarr, R2arr, Aarr, Barr, EpsArr = [], [], [], [], [], [], [], []
for n in range(49):
    Zre, Zim, T, M = [], [], [0]*1, []
    s1,s2 = [], []
    Fmax, R, A, B = 0, 0, 0, 0
    F, T[0], e1, e2, s1, s2, S, Z, Zre, Zim, R, Fmax = parser(n)
    
    A,B = findInitParamsForLine(Zre,Zim,R)
    area = 0.005 * 0.005
    d = 0.001
    #eps0 = 8.85e-12
    eps = d / (R * 8.85e-6 * Fmax * area * 2 * math.pi)
    Tarr.append(T[0])
    Trevarr.append(1000/(T[0]+273))
    Farr.append(Fmax)
    Aarr.append(A)
    Barr.append(B)
    Rarr.append(R)
    R2arr.append(2*R)
    EpsArr.append(eps)

F, T, e1, e2, s1, s2, S, Z, Zre, Zim, R, Fmax = parser(48)

plt.ylim([0,R*35])
plt.xlim([0,R*7])
plt.ylim([0,R*5])
#plt.text(100,100,'I',fontsize = 24 )
plt.plot(Zre, Zim, label = 'T = '+str(T[0])+' \u2103')
plt.legend()

plt.axvline(R,color = 'pink', ls = '--')
plt.axvline(LL*R,color = 'lightgray', ls = '--')
plt.axvline(RR*R,color = 'lightgray', ls = '--')

plt.plot(Zre, M)
plt.xlabel('Z\',M\u03A9')
plt.ylabel('- Z\",M\u03A9')
plt.show()

plt.ylabel('R, M\u03A9')
plt.xlabel('1000/T, K')
#plt.xscale("log")

plt.yscale("log")
plt.plot(Trevarr, R2arr)
plt.show()


n1, n2 = 20,26
print(Tarr[n1],Tarr[n2])
print(Trevarr[n1],Trevarr[n2])
print(Rarr[n1],Rarr[n2])
print((Rarr[n2]-Rarr[n1])/(Trevarr[n2]-Trevarr[n1]))
#print((Rarr[n2]-Rarr[n1])/(Trevarr[n2]-Trevarr[n1]) / (1.6e-19/1.38e-20), 'eV')
print((math.log10(Rarr[n2])-math.log10(Rarr[n1]))/(Trevarr[n2]-Trevarr[n1]) / (1.6e-19/(1.38e-20/math.log10(math.exp(1)))), 'eV')
n1, n2 = 0,8 
print(Tarr[n1],Tarr[n2])
#print((Rarr[n2]-Rarr[n1])/(Trevarr[n2]-Trevarr[n1]) / (1.6e-19/1.38e-20), 'eV')
print((math.log10(Rarr[n2])-math.log10(Rarr[n1]))/(Trevarr[n2]-Trevarr[n1]) / (1.6e-19/(1.38e-20/math.log10(math.exp(1)))), 'eV')


plt.ylabel('a, Z\" = a\u22C5Z\' + b')
plt.xlabel('T, \u2103')
plt.yscale("linear")
plt.plot(Tarr, Aarr)
plt.show()

plt.ylabel('b, Z\" = a\u22C5Z\' + b')
plt.xlabel('T, \u2103')
plt.yscale("linear")
plt.plot(Tarr, Barr)
plt.show()
'''
plt.ylabel('Fx')
plt.xlabel('T, \u2103')
plt.yscale("linear")
plt.plot(Tarr, Farr)
plt.show()
'''
plt.ylabel('\u03B5')
plt.xlabel('T, \u2103')
plt.yscale("linear")
plt.axhline(76,color = 'lightgray', ls = '-')
plt.axhline(76-11,color = 'lightgray', ls = '--')
plt.axhline(76+11,color = 'lightgray', ls = '--')
plt.plot(Tarr, EpsArr, 'o')
plt.show()

f.close()