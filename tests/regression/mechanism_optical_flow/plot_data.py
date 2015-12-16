# Import everything from matplotlib (numpy is accessible via 'np' alias)
from pylab import *
import matplotlib.pyplot as plt

NUM_SUBSETS = 3
FILE_PREFIX = "./results/DICe_solution_"

font = {'family' : 'sans-serif',
        'weight' : 'regular',
        'size'   : 8}
matplotlib.rc('font', **font)

fig = figure(figsize=(8,8), dpi=150)
for i in range(0,NUM_SUBSETS):
  FILE = FILE_PREFIX+str(i)+".txt"
  DATA = loadtxt(FILE,delimiter=',',skiprows=21)
  FRAME    = DATA[:,0]
  DISP_X   = DATA[:,3]
  DISP_Y   = DATA[:,4]
  subplot(2,2,i+1)
  plot(FRAME,DISP_X,'-b')
  plot(FRAME,DISP_Y,'-r')
  title("Subset ID: "+str(i))
  if i==0:
    legend(["$u$","$v$"], loc=2)
  xlabel('Image Number')
  ylabel('Displacement (pixels)')
  fig.set_tight_layout(True)
  savefig('Disp.pdf',dpi=150, format='pdf')

fig = figure(figsize=(8,8), dpi=150)
for i in range(0,NUM_SUBSETS):
  FILE = FILE_PREFIX+str(i)+".txt"
  DATA = loadtxt(FILE,delimiter=',',skiprows=21)
  FRAME    = DATA[:,0]
  THETA    = DATA[:,5]
  subplot(2,2,i+1)
  plot(FRAME,THETA,'-g')
  title("Subset ID: "+str(i))
  if i==0:
    legend(["theta"],loc=2)
  xlabel('Image Number')
  ylabel('Rotation (Rad)')
  fig.set_tight_layout(True)
  savefig('Theta.pdf',dpi=150, format='pdf')

fig = figure(figsize=(8,8), dpi=150)
for i in range(0,NUM_SUBSETS):
  FILE = FILE_PREFIX+str(i)+".txt"
  DATA = loadtxt(FILE,delimiter=',',skiprows=21)
  FRAME    = DATA[:,0]
  GAMMA    = DATA[:,7]
  subplot(2,2,i+1)
  plot(FRAME,GAMMA,'-m')
  title("Subset ID: "+str(i))
  if i==0:
    legend(["Gamma"],loc=2)
  xlabel('Image Number')
  ylabel('Gamma (.)')
  fig.set_tight_layout(True)
  savefig('Gamma.pdf',dpi=150, format='pdf')

fig = figure(figsize=(8,8), dpi=150)
for i in range(0,NUM_SUBSETS):
  FILE = FILE_PREFIX+str(i)+".txt"
  DATA = loadtxt(FILE,delimiter=',',skiprows=21)
  FRAME    = DATA[:,0]
  SIGMA    = DATA[:,6]
  subplot(2,2,i+1)
  plot(FRAME,SIGMA,'-c')
  title("Subset ID: "+str(i))
  if i==0:
    legend(["Sigma"],loc=2)
  xlabel('Image Number')
  ylabel('Sigma (.)')
  fig.set_tight_layout(True)
  savefig('Sigma.pdf',dpi=150, format='pdf')

fig = figure(figsize=(8,8), dpi=150)
for i in range(0,NUM_SUBSETS):
  FILE = FILE_PREFIX+str(i)+".txt"
  DATA = loadtxt(FILE,delimiter=',',skiprows=21)
  FRAME    = DATA[:,0]
  BETA     = DATA[:,8]
  subplot(2,2,i+1)
  plot(FRAME,BETA,'-k')
  title("Subset ID: "+str(i))
  if i==0:
    legend(["Beta"],loc=2)
  xlabel('Image Number')
  ylabel('Beta (.)')
  fig.set_tight_layout(True)
  savefig('Beta.pdf',dpi=150, format='pdf')
  
show()    

