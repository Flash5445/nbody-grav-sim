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
        acc = np.zeros((p.shape))
        for pos_ind, pos in enumerate(p): # this iterates pos through all positions
            # directions
            dir = np.delete(p - pos, pos_ind, axis=0) # this takes the position, broadcasts it to be subtracted by the p database, and then deletes the row that the position was taken from
            #magnitude
            mag = np.array([(G * np.delete(m.T, pos_ind, axis=None).T)/(np.power(np.sqrt(np.sum(np.square(dir), axis=1)), 3) + 1e-6)]).T
            # acceleration = sum(magnitude * direction)
            acc_row = np.sum(mag * dir, axis=0)
            acc[pos_ind] = acc_row
            pos_ind += 1
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
