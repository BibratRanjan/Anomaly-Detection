from matplotlib.pyplot import plot
import numpy as np

error = np.load('unm_total_error.npy')
plt = plot(error)