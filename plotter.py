import scipy.optimize as optimization
import matplotlib.pyplot as plt
import matplotlib
import math
import numpy

with open ("data_chopped.dat","r", errors = 'replace') as f:
  lines = f.readlines()
  print(len(lines))

n=10 #0
LL,RR = 3,4

def parser(n):
    F, T, e1, e2, s1, s2, S, Z, Zre, Zim, tan= [], [], [], [], [], [], [], [], [], [], []
    for index in range(3+42*n,45+42*n):
        values = lines[index].split("\t")
        F.append(float(values[0]))
        T.append(float(values[1]))
        e1.append(float(values[2]))
        e2.append(float(values[3]))
        s1.append(float(values[4]))
        tan.append(float(values[5]))
        s2.append(abs(float(values[4])/float(values[5])))
        S.append(complex(float(values[4]), abs(float(values[4])/float(values[5]))))
        ZZ = 1e-6/complex(float(values[4]), abs(float(values[4])/float(values[5])))
        Z.append(ZZ)
        Zre.append(ZZ.real)
        Zim.append(-ZZ.imag)
    print(len(Zre))

    for k in range(1,36):
        if Zim[k]>Zim[k-1] and Zim[k]>Zim[k+1]:
            break
    print(s1)
    print(s2)    
    R = Zim[k]
    F0 = F[k]

    print ('Z\" = ',R,"M\u03A9, Fmax = ", F0, 'Hz, T = ', T[0],'C')    
    return F, T, e1, e2, s1, s2, S, Z, Zre, Zim, R, F0


F, T, e1, e2, s1, s2, S, Z, Zre, Zim, R, F0 = parser(n)

plt.plot(Zre, Zim, label = 'T = '+str(T[0])+' \u2103')
plt.legend()


plt.xlabel('Z\',M\u03A9')
plt.ylabel('- Z\",M\u03A9')
plt.show()