import backtrace

import rllab.training_configs
import _pickle as cPickle

backtrace.hook(
    reverse=False,
    align=False,
    strip_path=True,
    enable_on_envvar_only=False,
    on_tty=False,
    conservative=False,
    #styles={}
    )

import sys

sys.path.append(".")

from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
import lasagne.nonlinearities as NL

# Policy
from rllab.policies.meta_policy import MetaPolicy

# Baseline
from sandbox.cpo.baselines.gaussian_mlp_baseline import GaussianMLPBaseline

# Environment
from rllab.envs.gym_env import GymEnv
from rllab.spaces.discrete import Discrete

# Sampler
from sandbox.cpo.algos.safe.meta_sampler_safe import MetaBatchSamplerSafe

# Policy optimization
from sandbox.cpo.algos.safe.cpo import CPO
from sandbox.cpo.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.cpo.safety_constraints.trivial import TrivialSafetyConstraint
from sandbox.cpo.safety_constraints.lunar_lander import LunarLanderSafetyConstraintAbs # absolute

# Utils
from copy import deepcopy

ec2_mode = False

class dummy(object):
    def __init__(self):
        pass
    def get_action(self, o):
        return [0.0,0.0], "info"

def run_task(*_):
        trpo_stepsize = 0.01
        trpo_subsample_factor = 0.2 
        log_dir=rllab.training_configs.LOG_DIR_PATH+"/meta/"
        env = GymEnv('LunarLanderContinuous-v2', record_video=True, log_dir=log_dir, force_reset=True)
        #
        # load subpolicies
        ######################################################
        # Change this code 
        ######################################################
        #subpolices = load_subpolicies()

        with open(rllab.training_configs.up_subpol_path, 'rb') as f:
            sp_up = cPickle.load(f)
        with open(rllab.training_configs.down_subpol_path, 'rb') as f:
            sp_down = cPickle.load(f)
        with open(rllab.training_configs.right_subpol_path, 'rb') as f:
            sp_right = cPickle.load(f)
        with open(rllab.training_configs.left_subpol_path, 'rb') as f:
            sp_left = cPickle.load(f)

        subpolicies = {0: sp_up, 1:sp_down, 2:sp_right, 3:sp_left}
        ######################################################
        # # #
        ######################################################
        #
        # make a copy of env spec and change action space dim to be categorical distribution in the number of policies
        env_mod = deepcopy(env)
        n_sub_pols = len(subpolicies.keys())
        env_mod.spec._action_space = Discrete(n_sub_pols)
        #
        # construct meta-policy
        policy = MetaPolicy(env_mod.spec,
                    subpolicies,
                    hidden_sizes=(64,32)
                 )
        #
        # baseline for removing variance
        baseline = GaussianMLPBaseline(
            env_spec=env_mod.spec,
            regressor_args={
                    'hidden_sizes': (64,32),
                    'hidden_nonlinearity': NL.tanh,
                    'learn_std':False,
                    'step_size':trpo_stepsize,
                    'optimizer':ConjugateGradientOptimizer(subsample_factor=trpo_subsample_factor)
                    }
        )
        # cost shaping for safety constraint? Or advantage estimator for the cost?
        safety_baseline = GaussianMLPBaseline(
            env_spec=env_mod.spec,
            regressor_args={
                    'hidden_sizes': (64,32),
                    'hidden_nonlinearity': NL.tanh,
                    'learn_std':False,
                    'step_size':trpo_stepsize,
                    'optimizer':ConjugateGradientOptimizer(subsample_factor=trpo_subsample_factor)
                    },
            target_key='safety_returns',
            )
        #
        # index 4 is angle, abs(angle) >= 0.4 rads is bad
        # index 5 is angular velocity, guessing that abs(ang_vel)>=0.8 rads/sec is bad   
        # limit is the max angle/angular velocity value
        # max_value is the budget
        safety_constraint = LunarLanderSafetyConstraintAbs(max_value=5.0, idx=4, limit=0.4, baseline=safety_baseline) 
        algo = CPO(
            env=env,
            policy=policy,
            baseline=baseline,
            safety_constraint=safety_constraint,
            safety_gae_lambda=1,
            batch_size=2500,
            max_path_length=80, 
            n_itr=100,
            gae_lambda=0.95,
            discount=0.995,
            step_size=trpo_stepsize,
            optimizer_args={'subsample_factor':trpo_subsample_factor},
            sampler_cls=MetaBatchSamplerSafe,
            #plot=True,
        )
        algo.train()


run_experiment_lite(
    run_task,
    n_parallel=1,
    snapshot_mode="last",
    exp_prefix='CPO-LunarLanderNonHierarchicalSafeAngle',
    seed=1,
    mode = "ec2" if ec2_mode else "local"
    #plot=True
)

