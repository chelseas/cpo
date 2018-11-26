from rllab.core.parameterized import Parameterized
from rllab.core.serializable import Serializable
from sandbox.cpo.safety_constraints.base import *
import numpy as np

# ?-? constraint. 
class LunarLanderSafetyConstraintAbs(SafetyConstraint, Serializable):

    def __init__(self, max_value=1., idx=-3, **kwargs):
        self.max_value = max_value
        self.idx = idx
        Serializable.quick_init(self, locals())
        super(LunarLanderSafetyConstraintAbs,self).__init__(max_value, **kwargs)

    def evaluate(self, path):
        return np.abs(path['observations'][:,self.idx]) >= self.max_value
