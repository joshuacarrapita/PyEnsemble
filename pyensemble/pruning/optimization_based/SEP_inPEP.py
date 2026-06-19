# coding: utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from copy import deepcopy
import gc
gc.enable()
import numpy as np

from pyensemble.utils_const import DTY_INT, DTY_BOL
from pyensemble.pruning.utils_inPEP import PEP_f_Hs
from pyensemble.pruning.utils_inPEP import PEP_flipping_uniformly



#----------------------------------
# OEP, SEP
#----------------------------------


# Simple Ensemble Pruning
#
def PEP_SEP(yt, y, nb_cls, rho):
    s = np.random.randint(2, size=nb_cls).tolist()
    if sum(s) == 0:
        s[np.random.randint(nb_cls)] = 1
    #
    nb_count = int(np.ceil((nb_cls**3) * np.log(nb_cls))) if nb_cls > 1 else 1
    while nb_count >= 0:
        sp = PEP_flipping_uniformly(s)
        f1, _ = PEP_f_Hs(y, yt, sp)
        f2, _ = PEP_f_Hs(y, yt, s)
        if f1 <= f2:
            s = deepcopy(sp)
        #   #
        nb_count = nb_count - 1  # nb_count -= 1
        del sp, f1, f2
    #   #   #
    yo = np.array(yt)[np.array(s) == 1].tolist()
    P = np.array(s, dtype=DTY_BOL).tolist()
    del nb_count, s
    gc.collect()
    return deepcopy(yo), deepcopy(P)
