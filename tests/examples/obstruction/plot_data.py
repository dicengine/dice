# Import everything from matplotlib (numpy is accessible via 'np' alias)
from pylab import *
import matplotlib.pyplot as plt

NUM_SUBSETS = 2
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
  THETA    = DATA[:,3]
  SIGMA    = DATA[:,4]
  GAMMA    = DATA[:,5]
  fig = figure(figsize=(12,5), dpi=150)
  plot(FRAME,THETA,'-g')
  fig.set_tight_layout(True)
  legend(["Rotation"])
  xlabel('Image Number')
  ylabel('Rotation (Rad)')
  title("Subset ID: "+str(i))
  savefig(PDFTHETA,dpi=150, format='pdf')
  
show()    

