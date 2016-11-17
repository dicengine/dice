# Import everything from matplotlib (numpy is accessible via 'np' alias)
from pylab import *
import matplotlib.pyplot as plt
import numpy

font = {'family' : 'sans-serif',
        'weight' : 'regular',
        'size'   : 12}
matplotlib.rc('font', **font)

fig = figure(figsize=(7,5), dpi=150)

FILE1 = "synthetic_results/spatial_resolution.txt"
DATA1 = loadtxt(FILE1,skiprows=1)
X1 = 1.0/DATA1[:,6]
Y1 = 1.0 - DATA1[:,16]/100;
SD1 = DATA1[:,17]/100;
SP1 = Y1 + SD1
SM1 = Y1 - SD1

ax_f = fig.add_subplot(111)
ax_f.fill_between(X1,SP1,SM1,alpha=0.25,lw=0.0)
ax_f.plot(X1,Y1,'-b',lw=1.5)

# draw the speckle period line
#ax_f.plot([0.24, 0.24], [0.0, 2.0], color='k', linestyle='--', linewidth=1)
#ax_f.annotate('speckle period/motion period', xy=(0.24, 0.15), xytext=(0.35, 0.25),
#                        arrowprops=dict(facecolor='black', shrink=0.05,width=1,headwidth=5),)
#ax_f.annotate('log slope: -1.125', xy=(0.4, 0.58), xytext=(0.55, 0.78),
#                        arrowprops=dict(facecolor='black', shrink=0.05,width=1,headwidth=5),)
ax_f.plot([0.0, 0.006], [0.90, 0.90], color='k', linestyle='--', linewidth=1)
ax_f.plot([0.006, 0.006], [0.0, 0.90], color='k', linestyle='--', linewidth=1)
ax_f.annotate(r'$f_{90}$ = 0.006', xy=(0.007, 0.95), xytext=(0.007, 0.90))
ax_f.annotate('std dev = 0.10', xy=(0.007, 0.90), xytext=(0.007, 0.85))
ax_f.plot([0.025, 0.025], [-0.5, 2.0], color='r', linestyle='--', linewidth=1)
ax_f.annotate('step size\nNyquist limit', xy=(0.025, 0.95), xytext=(0.028, 0.85),
              arrowprops=dict(facecolor='black', shrink=0.05,width=1,headwidth=5),)
ax_f.plot([0.02, 0.02], [-0.5, 2.0], color='g', linestyle='--', linewidth=1)
ax_f.annotate('subset size\nNyquist limit', xy=(0.02, 0.55), xytext=(0.022, 0.45),
              arrowprops=dict(facecolor='black', shrink=0.05,width=1,headwidth=5),)
ax_f.set_xlabel("Motion frequency (Hz), amp = 1.0km")
ax_f.set_ylabel(r"Motion attenuation (ratio)")
ax_f.set_ylim([0,1.0])

fig.set_tight_layout(True)
savefig('ResCurve.pdf',dpi=150, format='pdf')

plt.show()

