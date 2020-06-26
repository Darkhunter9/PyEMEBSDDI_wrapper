from math import sin, cos
import numpy as np

def eu2qu(eu):
    '''
    input: euler angles, array-like (3,)
    output: quaternions, array-like (4,)
    default value of eps = 1
    '''

    eps = 1

    sigma = 0.5 * (eu[0] + eu[2])
    delta = 0.5 * (eu[0] - eu[2])
    c = cos(eu[1]/2)
    s = sin(eu[1]/2)

    q0 = c * cos(sigma)

    if q0 >= 0:
        q = np.array([c*cos(sigma), -eps*s*cos(delta), -eps*s*sin(delta), -eps*c*sin(sigma)], dtype=float)
    else:
        q = np.array([-c*cos(sigma), eps*s*cos(delta), eps*s*sin(delta), eps*c*sin(sigma)], dtype=float)

    # set values very close to 0 as 0
    # thr = 10**(-10)
    # q[np.where(np.abs(q)<thr)] = 0.

    return q