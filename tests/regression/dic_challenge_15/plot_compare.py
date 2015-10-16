# Import everything from matplotlib (numpy is accessible via 'np' alias)
from pylab import *
import matplotlib.pyplot as plt

font = {'family' : 'sans-serif',
    'weight' : 'regular',
    'size'   : 8}
matplotlib.rc('font', **font)

DATA = loadtxt('avg_results_K50.txt')
X        = DATA[:,0]
DISP     = DATA[:,1]
COMMAND  = DATA[:,2]
DIFF     = DATA[:,3]

fig = figure(figsize=(12,5), dpi=150)
plot(X,DISP,'-b')
plot(X,COMMAND,'-k')
fig.set_tight_layout(True)
xlabel('x position(px)')
ylabel('x displacement(px)')
xlim(0.0,1000.0)
legend(["Computed","Command"])

show()   
