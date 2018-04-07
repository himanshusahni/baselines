import os
import os.path as osp
import gym
import time
import joblib
import logging
import numpy as np
import tensorflow as tf
from baselines import logger

from baselines.common import set_global_seeds, explained_variance
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.atari_wrappers import wrap_deepmind
from baselines.common import tf_util

from baselines.a2c.utils import discount_with_dones
from baselines.a2c.utils import Scheduler, make_path, find_trainable_variables
from baselines.a2c.utils import cat_entropy, mse

class Model(object):

    def __init__(self, policy, ob_space, ac_space, noptions, nenvs, nsteps,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
            alpha=0.99, epsilon=1e-5, deliberation_cost=0.01, total_timesteps=int(80e6), lrschedule='linear'):

        sess = tf_util.make_session()
        nact = ac_space.n
        nbatch = nenvs*nsteps

        A = tf.placeholder(tf.int32, [nbatch])     # Actions
        Q_U = tf.placeholder(tf.float32, [nbatch])
        A_OHM = tf.placeholder(tf.float32, [nbatch])  # Advantages for beta updates
        LR = tf.placeholder(tf.float32, [])        # Learning Rates

        step_model = policy(
            sess,
            ob_space=ob_space,
            ac_space=ac_space,
            noptions=noptions,
            nbatch=nenvs,
            nsteps=1,
            reuse=False)
        train_model = policy(
            sess,
            ob_space=ob_space,
            ac_space=ac_space,
            noptions=noptions,
            nbatch=nenvs*nsteps,
            nsteps=nsteps,
            reuse=True)

        neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.opt_pi_logits, labels=A)
        logpterm = tf.sigmoid(train_model.beta_logits)

        # TODO: Need to compute Q_U(s,w,a) and A_OHM(s', w)

        # Intra-option policy gradient loss.
        pg_loss = tf.reduce_mean(Q_U * neglogpac)
        # Intra-option termination gradient loss.
        tg_loss = tf.reduce_mean((A_OHM + deliberation_cost) * logpterm)

        entropy = tf.reduce_mean(cat_entropy(train_model.opt_pi_logits))
        # loss = pg_loss - entropy*ent_coef

        loss = pg_loss + tg_loss - entropy*ent_coef

        params = find_trainable_variables("model")
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        # TODO: Learning rates for updating the termination weights and intra-option weights may be different.
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)
        _train = trainer.apply_gradients(grads)

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, options, Qus, Aohms, actions):
            for step in range(len(obs)):
                cur_lr = lr.value()
            td_map = {train_model.X:obs, train_model.opt:options, Q_U:Qus, A_OHM:Aohms, A:actions}
            # if states is not None:
                # td_map[train_model.S] = states
                # td_map[train_model.M] = masks
            # TODO: Figure out - loss for option policies, beta, q_opt
            policy_loss, value_loss, policy_entropy, _ = sess.run(
                [pg_loss, vf_loss, entropy, _train],
                td_map
            )
            return policy_loss, value_loss, policy_entropy

        def save(save_path):
            ps = sess.run(params)
            make_path(osp.dirname(save_path))
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            ps = sess.run(restores)

        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.save = save
        self.load = load
        self.sess = sess
        tf.global_variables_initializer().run(session=sess)

class Runner(object):

    def __init__(self, env, model, noptions=1, nsteps=5, gamma=0.99):
        self.env = env
        self.model = model
        self.noptions = noptions
        nh, nw, nc = env.observation_space.shape
        nenv = env.num_envs
        self.batch_ob_shape = (nenv*nsteps, nh, nw, nc)
        self.obs = np.zeros((nenv, nh, nw, nc), dtype=np.uint8)
        self.nc = nc
        obs = env.reset()
        self.gamma = gamma
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [True for _ in range(nenv)]
        self.opts = [0 for _ in range(nenv)]
        self.opt_dones = [True for _ in range(nenv)]
        self.opt_eps_start = 1.0
        self.opt_eps_end = 0.05
        self.opt_eps_steps = 1000000
        self.opt_eps = self.opt_eps_start

    def run(self):
        mb_obs, mb_opt_values, mb_options, mb_actions, mb_betas, mb_dones, mb_rewards = [],[],[],[],[],[],[]
        mb_options.append(np.copy(self.opts))
        # mb_states = self.states
        for n in range(self.nsteps):
            # get option values
            Qopt = self.model.step_model.Qopt(self.obs)
            # sample new options if ended last step
            for i, opt_done in enumerate(self.opt_dones):
                if opt_done:
                    # pick epsilon greedy
                    if np.random.uniform() < self.opt_eps:
                        self.opts[i] = np.random.choice(np.arange(self.noptions))
                    else:
                        self.opts[i] = np.argmax(Qopt[i])
            actions, betas, opt_dones = self.model.step_model.step(self.obs, self.opts)
            #TODO: update self.opt_dones
            self.opt_dones = opt_dones
            mb_obs.append(np.copy(self.obs))
            mb_opt_values.append(Qopt)
            mb_options.append(np.copy(self.opts))
            mb_actions.append(actions)
            mb_betas.append(betas)
            mb_dones.append(np.copy(self.dones))
            obs, rewards, dones, _ = self.env.step(actions)
            # self.states = states
            self.dones = dones
            for i, done in enumerate(dones):
                if done:
                    self.obs[i] = self.obs[i]*0
                    self.opt_dones[i] = True
            self.obs = obs
            mb_rewards.append(rewards)
            if self.opt_eps > self.opt_eps_end:
                self.opt_eps -= (self.opt_eps_start - self.opt_eps_end)/self.opt_eps_steps
        mb_dones.append(self.dones)
        # mb_options -> (nenvs, nsteps+1) -> w_-1, w_0, w_1, w_2... w_nsteps-1
        # mb_dones -> (nenvs, nsteps+1) -> d_-1, d_0, d_1, d_2... d_nsteps-1
        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_opt_values = np.asarray(mb_opt_values, dtype=np.float32).swapaxes(1, 0)
        mb_options = np.asarray(mb_options, dtype=np.int32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_betas = np.asarray(mb_betas, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        # mb_masks = mb_dones[:, :-1]

        # discount/bootstrap off option value fn
        last_values = self.model.Qopt(self.obs).tolist()
        _, last_betas, _ = self.model.step(self.obs, self.opts)
        last_betas = last_betas.tolist()
        mb_Qu = []
        mb_Aohm = []
        for n, (opt_values, options, betas, dones, rewards, last_value, last_beta) in enumerate(zip(mb_opt_values, mb_options,mb_betas, mb_dones, mb_rewards, last_values, last_betas)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            betas = betas.tolist()
            options = options.tolist()
            # # last option value is bootstrapped
            betas += [last_beta]
            opt_values = np.concatenate((opt_values, np.expand_dims(last_values, axis=0)), axis=0)
            # betas -> (nsteps+1,) -> b_0, b_1, .... b_nsteps
            # opt_values -> (nsteps+1, noptions) -> Q_0, Q_1, .... Q_nsteps
            # reverse the option values
            Qu = []
            Aohm = []
            for i in range(self.nsteps, 0, -1):
                q = rewards[i-1] + (1-dones[i])*self.gamma*((1-betas[i])*opt_values[i,options[i-1]] + betas[i]*np.amax(opt_values[i]))
                Qu.append(q)
                a = opt_values[i-1, options[i-1]] - np.amax(opt_values[i-1])
                Aohm.append(a)
            mb_Qu.append(Qu)
            mb_Aohm.append(Aohm)
        mb_options = mb_options.flatten()
        mb_Qu = mb_Qu.flatten()
        mb_Aohm = mb_Aohm.flatten()
        mb_actions = mb_actions.flatten()
        # mb_betas = mb_betas.flatten()
        # mb_masks = mb_masks.flatten()
        # TODO: comput Aohm
        return mb_obs, mb_options, mb_Qu, mb_Aohm, mb_actions

def learn(
        policy, env, seed, noptions=1, nsteps=5, total_timesteps=int(80e6),
        vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, lr=7e-4,
        lrschedule='linear', epsilon=1e-5, alpha=0.99, gamma=0.99,
        log_interval=100):
    tf.reset_default_graph()
    set_global_seeds(seed)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    model = Model(
        policy=policy,
        ob_space=ob_space,
        ac_space=ac_space,
        noptions=noptions,
        nenvs=nenvs,
        nsteps=nsteps,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        lr=lr,
        alpha=alpha,
        epsilon=epsilon,
        total_timesteps=total_timesteps,
        lrschedule=lrschedule)
    sample_x = np.random.normal(size=(16,84,84,4))
    opt = np.random.randint(0,4,size=(16,))
    q = model.sess.run([model.step_model.betas, model.step_model.beta_logits], feed_dict={model.step_model.X:sample_x, model.step_model.opt:opt})
    print(q[0][3])
    print(q[1][3])
    print(opt)

    exit()



    runner = Runner(env, model, noptions=noptions, nsteps=nsteps, gamma=gamma)

    nbatch = nenvs*nsteps
    tstart = time.time()
    for update in range(1, total_timesteps//nbatch+1):
        obs, options, Qus, Aohms, actions = runner.run()
        #TODO: match the outputs
        policy_loss, value_loss, policy_entropy = model.train(obs, options, Qus, Aohms, actions)
        nseconds = time.time()-tstart
        fps = int((update*nbatch)/nseconds)
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, rewards)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update*nbatch)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("explained_variance", float(ev))
            logger.dump_tabular()
    env.close()
