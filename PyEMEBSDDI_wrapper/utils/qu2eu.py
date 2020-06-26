from math import sin, cos, pi, sqrt, atan2
import numpy as np

def qu2eu(qu):
    '''
    input: quaternions, 1darray (4,)
    output: euler angles, 1darray (3,), unit is in rad
    default value of eps = 1
    '''

    eps = 1

    q03 = qu[0]**2 + qu[3]**2
    q12 = qu[1]**2 + qu[2]**2
    chi = sqrt(q03*q12)

    if chi == 0 and q12 == 0:
        result = np.array([atan2(-2*eps*qu[0]*qu[3], qu[0]**2-qu[3]**2), 0, 0])
    elif chi == 0 and q03 == 0:
        result = np.array([atan2(2*qu[1]*qu[2], qu[1]**2-qu[2]**2), pi, 0])
    else:
        result = np.array([atan2((qu[1]*qu[3]-eps*qu[0]*qu[2])/chi, (-eps*qu[0]*qu[1]-qu[2]*qu[3])/chi),
                            atan2(2*chi, q03-q12),
                            atan2((eps*qu[0]*qu[2]+qu[1]*qu[3])/chi, (-eps*qu[0]*qu[1]+qu[2]*qu[3])/chi)])

    # reduce Euler angles to definition ranges (and positive values only)
    if result[0] < 0.0:
        result[0] = (result[0]+100.*pi)%(2.*pi)
    if result[1] < 0.0:
        result[1] = (result[1]+100.*pi)%pi
    if result[2] < 0.0:
        result[2] = (result[2]+100.*pi)%(2.*pi)

    return result