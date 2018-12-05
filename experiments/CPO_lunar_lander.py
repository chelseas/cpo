import sys

sys.path.append(".")

import rllab.training_configs

from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
import lasagne.nonlinearities as NL

# Policy
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

# Baseline
from sandbox.cpo.baselines.gaussian_mlp_baseline import GaussianMLPBaseline

# Environment
from rllab.envs.gym_env import GymEnv

# Policy optimization
from sandbox.cpo.algos.safe.cpo import CPO
from sandbox.cpo.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.cpo.safety_constraints.trivial import TrivialSafetyConstraint
from sandbox.cpo.safety_constraints.lunar_lander import LunarLanderSafetyConstraintAbs # absolute


ec2_mode = False

subpol = rllab.training_configs.SUBPOLICY

def run_task(*_):
        trpo_stepsize = 0.01
        trpo_subsample_factor = 0.2 
        log_dir=rllab.training_configs.LOG_DIR_PATH+"/"+subpol+"/"
        env = GymEnv('LunarLanderContinuous'+subpol+'-v2', record_video=True, log_dir=log_dir, force_reset=True)
        # main policy
        policy = GaussianMLPPolicy(env.spec,
                    hidden_sizes=(64,32)
                 )
        # baseline for removing variance
        baseline = GaussianMLPBaseline(
            env_spec=env.spec,
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
            env_spec=env.spec,
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
            batch_size=500, # 5000
            max_path_length=200, # 300
            n_itr=60,#100
            gae_lambda=0.95,
            discount=0.995,
            step_size=trpo_stepsize,
            optimizer_args={'subsample_factor':trpo_subsample_factor},
            #plot=True,
        )
        algo.train()


run_experiment_lite(
    run_task,
    n_parallel=4,
    snapshot_mode="last",
    exp_prefix='CPO-LunarLanderNonHierarchicalSafeAngle',
    seed=1,
    mode = "ec2" if ec2_mode else "local"
    #plot=True
)

