"""
BSD 3-Clause License

Copyright (c) 2020, Zihao Ding, Marc De Graef Research Group/Carnegie Mellon University
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

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