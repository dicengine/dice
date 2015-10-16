#Import everything from matplotlib (numpy is accessible via 'np' alias)
from pylab import *
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# skiprows used to skip the header comments
DATA = loadtxt("./results/dic_challenge_15_0.txt",skiprows=21,delimiter=",")
X = DATA[:,0]
Y = DATA[:,1]
DISP_X = DATA[:,2]
DISP_Y = DATA[:,3]
GAMMA = DATA[:,7]
MATCH = DATA[:,8]
triang = tri.Triangulation(X, Y)

font = {'family' : 'sans-serif',
    'weight' : 'regular',
    'size'   : 8}
matplotlib.rc('font', **font)

fig = figure(dpi=150,figsize=(12,4))
ax1 = tripcolor(triang, DISP_Y, shading='flat', cmap=plt.cm.rainbow)
xlim([0,2000])
ylim([0,1000])
xlabel('X (px)')
ylabel('Y (px)')
title('Displacement-X')
plt.gca().invert_yaxis()
colorbar()


fig = figure(dpi=150,figsize=(12,4))
ax1 = tripcolor(triang, MATCH, shading='flat', cmap=plt.cm.rainbow)
xlim([0,2000])
ylim([0,1000])
xlabel('X (px)')
ylabel('Y (px)')
title('Match')
plt.gca().invert_yaxis()
colorbar()

#savefig('DispX0.pdf',dpi=150, format='pdf')
show()
