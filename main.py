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
        pos_ind = 0
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

    def acceleration_calculator(self):
        return self.acceleration(self.m, self.p, self.G)
