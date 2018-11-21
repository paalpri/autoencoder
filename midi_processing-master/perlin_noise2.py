#%pylab inline
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from noise import pnoise1, pnoise3
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams['savefig.dpi'] = 120
sns.set_color_codes()


def get_perlin_noise(numWindows, zDim, alpha):
	tFinal = numWindows//10 +1
	time      = np.linspace(0.1, tFinal, num = numWindows)

	all_noises = []
	for i in range(zDim):
		starting_point = i*tFinal
		base_noise = pnoise1(starting_point)
		signal  = np.zeros(numWindows)
		norm_signal  = np.zeros(numWindows)
		for ctr, t in enumerate(time):
		    signal[ctr]  = pnoise1((t+starting_point) * alpha) - base_noise
		
		beta = 0.01
		for i in range(len(signal)):
			xnew = (signal[i] - min(signal)) / (max(signal) - min(signal))
			xnew = (2*xnew) - 1
			xnew = beta * xnew
			norm_signal[i] = xnew

		all_noises.append(norm_signal)

	return all_noises

# plt.plot(time, signal,  label = '0.5 octave',  linewidth = 2)
# plt.plot(time, signal2, label = '2 octaves', linewidth = 1.5)
# plt.plot(time, signal3, label = '4 octaves', linewidth = 1.0)
# plt.xlabel('Time')
# plt.legend()
# plt.show()