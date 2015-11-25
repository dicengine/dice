# Import everything from matplotlib (numpy is accessible via 'np' alias)
from pylab import *
import matplotlib.pyplot as plt

NUM_SUBSETS = 4
FILE_PREFIX = "./results/DICe_solution_"

font = {'family' : 'sans-serif',
        'weight' : 'regular',
        'size'   : 8}
matplotlib.rc('font', **font)

for i in range(0,NUM_SUBSETS):
  FILE = FILE_PREFIX+str(i)+".txt"
  PDFTHETA = "Results_"+str(i)+".pdf"
  print(FILE)
  DATA = loadtxt(FILE,delimiter=',',skiprows=21)
  FRAME    = DATA[:,0]
  X_TILDE  = DATA[:,1]
  Y_TILDE  = DATA[:,2]
  DISP_X   = DATA[:,3]
  DISP_Y   = DATA[:,4]
  THETA    = DATA[:,5]
  SIGMA    = DATA[:,6]
  GAMMA    = DATA[:,7]
  fig = figure(figsize=(12,5), dpi=150)
  plot(FRAME,DISP_X,'-b')
  plot(FRAME,DISP_Y,'-r')
  plot(FRAME,THETA,'-g')
  fig.set_tight_layout(True)
  legend(["Displacement X","Displacement Y","Rotation"])
  xlabel('Image Number')
  ylabel('Displacement (pixels) or Rotation (Rad)')
  title("Subset ID: "+str(i))
  savefig(PDFTHETA,dpi=150, format='pdf')
  
show()    

