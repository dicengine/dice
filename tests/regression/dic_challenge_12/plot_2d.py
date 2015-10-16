#Import everything from matplotlib (numpy is accessible via 'np' alias)
from pylab import *
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# skiprows used to skip the header comments
DATA = loadtxt("results/DICE_solution_10.txt",skiprows=21,delimiter=',')
X = DATA[:,0]
Y = DATA[:,1]
DISP_X = DATA[:,2]
DISP_Y = DATA[:,3]
E_XX = DATA[:,4]
E_YY = DATA[:,5]
E_XY = DATA[:,6]
MATCH = DATA[:,8]
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
ax1 = tripcolor(triang, E_XX, shading='flat', cmap=plt.cm.rainbow)
axis('equal')
xlabel('X (px)')
ylabel('Y (px)')
title('Green-Lagrange Strain-XX')
plt.gca().invert_yaxis()
colorbar()
savefig('strainXX.pdf',dpi=150, format='pdf')

fig = figure(dpi=150,figsize=(5,10))
ax1 = tripcolor(triang, E_YY, shading='flat', cmap=plt.cm.rainbow)
axis('equal')
xlabel('X (px)')
ylabel('Y (px)')
title('Green-Lagrange Strain-YY')
plt.gca().invert_yaxis()
colorbar()
savefig('strainYY.pdf',dpi=150, format='pdf')

fig = figure(dpi=150,figsize=(5,10))
ax1 = tripcolor(triang, E_XY, shading='flat', cmap=plt.cm.rainbow)
axis('equal')
xlabel('X (px)')
ylabel('Y (px)')
title('Green-Lagrange Strain-XY')
plt.gca().invert_yaxis()
colorbar()
savefig('strainXY.pdf',dpi=150, format='pdf')

fig = figure(dpi=150,figsize=(5,10))
ax1 = tripcolor(triang, MATCH, shading='flat', cmap=plt.cm.rainbow)
axis('equal')
xlabel('X (px)')
ylabel('Y (px)')
title('MATCH')
plt.gca().invert_yaxis()
colorbar()
savefig('match.pdf',dpi=150, format='pdf')

show()
#Other figures similar to above
