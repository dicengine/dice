#Import everything from matplotlib (numpy is accessible via 'np' alias)
from pylab import *
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# skiprows used to skip the header comments
DATA = loadtxt("results/DICe_solution_0.txt",skiprows=21,delimiter=',')
ID     = DATA[:,0]
X      = DATA[:,1]
Y      = DATA[:,2]
DISP_X = DATA[:,3]
DISP_Y = DATA[:,4]
EX     = DATA[:,5]
EY     = DATA[:,6]
SIGMA  = DATA[:,7]
GAMMA  = DATA[:,8]
MATCH  = DATA[:,9]
triang = tri.Triangulation(X, Y)
font = {'family' : 'sans-serif',
    'weight' : 'regular',
    'size'   : 8}
matplotlib.rc('font', **font)

fig = figure(dpi=150,figsize=(5,10))
ax1 = tripcolor(triang, DISP_X, shading='flat', cmap=plt.cm.rainbow)
axis('equal')
xlabel('X (px)')
ylabel('Y (px)')
title('Displacement-X')
plt.gca().invert_yaxis()
colorbar()
savefig('dispX.pdf',dpi=150, format='pdf')

fig = figure(dpi=150,figsize=(5,10))
ax1 = tripcolor(triang, DISP_Y, shading='flat', cmap=plt.cm.rainbow)
axis('equal')
xlabel('X (px)')
ylabel('Y (px)')
title('Displacement-Y')
plt.gca().invert_yaxis()
colorbar()
savefig('dispY.pdf',dpi=150, format='pdf')

fig = figure(dpi=150,figsize=(5,10))
ax1 = tripcolor(triang, EX, shading='flat', cmap=plt.cm.rainbow)
axis('equal')
xlabel('X (px)')
ylabel('Y (px)')
title('EX')
plt.gca().invert_yaxis()
colorbar()
savefig('Exx.pdf',dpi=150, format='pdf')

fig = figure(dpi=150,figsize=(5,10))
ax1 = tripcolor(triang, EY, shading='flat', cmap=plt.cm.rainbow)
axis('equal')
xlabel('X (px)')
ylabel('Y (px)')
title('EY')
plt.gca().invert_yaxis()
colorbar()
savefig('Eyy.pdf',dpi=150, format='pdf')

fig = figure(dpi=150,figsize=(5,10))
ax1 = tripcolor(triang, SIGMA, shading='flat', cmap=plt.cm.rainbow)
axis('equal')
xlabel('X (px)')
ylabel('Y (px)')
title('Sigma')
plt.gca().invert_yaxis()
colorbar()
savefig('Sigma.pdf',dpi=150, format='pdf')


show()
#Other figures similar to above
