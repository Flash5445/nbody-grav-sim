import numpy as np
import scipy.constants as const

class System:
    def __init__(self, m : np.ndarray, v : np.ndarray, p: np.ndarray, G : float):
        self.m = m
        self.v = v
        self.p = p
        self.G = G

    def center_of_mass_correction(m, v, p):
        # position
        p_com = (np.sum(m * p, axis=0) / np.sum(m, axis=0))
        p = p - p_com 
        # velocity
        v_com = (np.sum(m*v, axis=0) / np.sum(m, axis=0))
        v = v - v_com
        return p, v

    @staticmethod
    def acceleration(m, p, G):
        # so first off we need to find a vector and a distance
        diff = p[np.newaxis, :, :] - p[:, np.newaxis, :]
        
        # Calculate pairwise squared distances + softening factor to avoid division by zero
        dist_sq = np.sum(diff**2, axis=-1) + 1e-12 # Softening
        
        # Calculate inverse cube of the distance
        inv_dist_cubed = dist_sq**(-1.5)
        
        # Set self-interaction to zero
        np.fill_diagonal(inv_dist_cubed, 0.)

        # Calculate acceleration: a_i = sum_j(G * m_j * (r_j - r_i) / |r_ij|^3)
        # Note the sign change in `diff` is handled by the subtraction order
        # Broadcasting m and inv_dist_cubed to match the shape of diff
        acc = G * np.sum(m * inv_dist_cubed[..., np.newaxis] * diff, axis=1)
        return acc
    
    @staticmethod
    def f(m, x, t):
        # loop through x[pos, 0] gives p
        # loop through x[pos, 1] gives v
        newp = x[:, 1]
        newv = System.acceleration(m, x[:, 0], const.G)
        newx = np.stack((newp, newv), axis=1)
        return newx
    
    @staticmethod
    def rk4(m, x, t, dt):
        f1 = System.f(m, x, t)
        f2 = System.f(m, x + 0.5*dt*f1, t + 0.5*dt)
        f3 = System.f(m, x + 0.5*dt*f2, t + 0.5*dt)
        f4 = System.f(m, x + dt*f3, t + dt)
        x_n = x + (dt * (f1 + 2*f2 + 2*f3 + f4)) / 6
        return x_n

    def simulate(self, m, p, v, t, dt, T):
        x = np.stack((p, v), axis=1)
        iterations = T/dt
        history = np.zeros((int(iterations), x.shape[0], x.shape[1], x.shape[2]))
        for i in range(int(iterations)):
            x = System.rk4(m, x, t, dt)
            history[i] = x
        return history
