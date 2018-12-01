from rllab.core.parameterized import Parameterized
from rllab.core.serializable import Serializable
from sandbox.cpo.safety_constraints.base import *
import numpy as np

# This constraint is always, trivially, satisfied. 
# Hm, not sure about that...
class TrivialSafetyConstraint(SafetyConstraint, Parameterized):

    def __init__(self, max_value=1., **kwargs):
        self.max_value = max_value # max value of expected discounted constraint cost
        super(TrivialSafetyConstraint,self).__init__(**kwargs)


    def evaluate(self, path):
    # compute constraint cost. 
        #return np.zeros_like(path['rewards'])
        return 0.5*np.random.rand(len(path['rewards']))

    def fit(self, paths):
        pass

