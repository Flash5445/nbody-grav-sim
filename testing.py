from main import System
import numpy as np
import scipy.constants as const

m = np.random.rand(10, 1) * 1e13
v = np.zeros((10, 3))
p = np.random.rand(10, 3) * 1e2

sys = System(m, v, p, const.G)

