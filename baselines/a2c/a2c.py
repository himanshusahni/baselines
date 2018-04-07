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

        A = tf.placeholder(tf.int32, [nbatch])        # Actions
        Q_U = tf.placeholder(tf.float32, [nbatch])    # Q_U(s,w,a) values
        A_OHM = tf.placeholder(tf.float32, [nbatch])  # Advantages for beta updates
        PG_LR = tf.placeholder(tf.float32, [])        # Policy Learning Rates
        TG_LR = tf.placeholder(tf.float32, [])        # Termination Learning Rates
        Q_LR = tf.placeholder(tf.float32, [])         # Option-value Learning Rates

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

        # Intra-option policy gradient loss.
        entropy = tf.reduce_mean(cat_entropy(train_model.opt_pi_logits))
        pg_loss = tf.reduce_mean(Q_U * neglogpac) - entropy*ent_coef
        # Intra-option termination gradient loss.
        tg_loss = tf.reduce_mean((A_OHM + deliberation_cost) * logpterm)
        q_loss = tf.reduce_mean(mse(tf.squeeze(train_model.Qopt_vals), Q_U))

        params = find_trainable_variables("model")
        pg_grads = tf.gradients(pg_loss, params)
        tg_grads = tf.gradients(tg_loss, params)
        q_grads = tf.gradients(q_loss, params)

        if max_grad_norm is not None:
            pg_grads, pg_grad_norm = tf.clip_by_global_norm(pg_grads, max_grad_norm)
            tg_grads, tg_grad_norm = tf.clip_by_global_norm(tg_grads, max_grad_norm)
            q_grads, q_grad_norm = tf.clip_by_global_norm(q_grads, max_grad_norm)

        pg_grads = list(zip(pg_grads, params))
        tg_grads = list(zip(tg_grads, params))
        q_grads = list(zip(q_grads, params))

        pg_trainer = tf.train.RMSPropOptimizer(learning_rate=PG_LR, decay=alpha, epsilon=epsilon)
        _pg_train = pg_trainer.apply_gradients(pg_grads)

        tg_trainer = tf.train.RMSPropOptimizer(learning_rate=TG_LR, decay=alpha, epsilon=epsilon)
        _tg_train = tg_trainer.apply_gradients(tg_grads)

        q_trainer = tf.train.RMSPropOptimizer(learning_rate=Q_LR, decay=alpha, epsilon=epsilon)
        _q_train = q_trainer.apply_gradients(q_grads)

        # TODO: Learning rates for updating the termination weights, intra-option weights, and q_value weights may be different.
        pg_lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)
        tg_lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)
        q_lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, options, Qus, Aohms, actions):
            for step in range(len(obs)):
                cur_pg_lr = pg_lr.value()
                cur_tg_lr = tg_lr.value()
                cur_q_lr = q_lr.value()
            td_map = {train_model.X:obs, train_model.opt:options, Q_U:Qus, A_OHM:Aohms, A:actions, PG_LR:cur_pg_lr, TG_LR:cur_tg_lr, Q_LR:cur_q_lr}
            # if states is not None:
                # td_map[train_model.S] = states
                # td_map[train_model.M] = masks
            # TODO: Figure out - loss for option policies, beta, q_opt
            policy_loss, term_loss, q_value_loss, policy_entropy, _, _, _ = sess.run(
                [pg_loss, tg_loss, q_loss, entropy, _pg_train, _tg_train, _q_train],
                td_map
            )
            return policy_loss, term_loss, q_value_loss, policy_entropy

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
        # self.states = model.initial_state
        self.dones = [True for _ in range(nenv)]
        self.opts = [0 for _ in range(nenv)]
        self.opt_dones = [1 for _ in range(nenv)]
        self.opt_eps_start = 1.0
        self.opt_eps_end = 0.05
        self.opt_eps_steps = 1000000
        self.opt_eps = self.opt_eps_start
        self.exit_counter = 0

    def run(self):
        mb_obs, mb_opt_values, mb_options, mb_actions, mb_betas, mb_dones, mb_rewards = [],[],[],[],[],[],[]
        mb_options.append(np.copy(self.opts))
        mb_opt_pi = []
        # mb_states = self.states
        for n in range(self.nsteps):
            # get option values
            Qopt = self.model.step_model.Qopt(self.obs)
            # sample new options if ended last step
            # print(self.opt_dones)
            for i, opt_done in enumerate(self.opt_dones):
                if opt_done:
                    # pick epsilon greedy
                    if np.random.uniform() < self.opt_eps:
                        self.opts[i] = np.random.choice(np.arange(self.noptions))
                    else:
                        self.opts[i] = np.argmax(Qopt[i])
            actions, betas, opt_dones = self.model.step_model.step(self.obs, self.opts)
            self.opt_dones = opt_dones
            mb_obs.append(np.copy(self.obs))
            mb_opt_values.append(Qopt)
            mb_opt_pi.append(self.model.sess.run(self.model.step_model.opt_pi, feed_dict={self.model.step_model.X:self.obs, self.model.step_model.opt:self.opts}))
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

        # Q-opts -> (nenvs, nsteps+1) -> Q_1, Q_2, ..., Q_nsteps, Q_nsteps+1
        # betas -> (nenvs, nsteps+1) -> b1, b2, ..., b_nsteps, b_nsteps+1
        # mb_options -> (nenvs, nsteps+1) -> w_0, w_1, w_2, w_3..., w_nsteps
        # mb_dones -> (nenvs, nsteps+1) -> d_0, d_1, d_0, d_1, d_2..., d_nsteps
        # batch of steps to batch of rollouts

        mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_opt_values = np.asarray(mb_opt_values, dtype=np.float32).swapaxes(1, 0)
        mb_options = np.asarray(mb_options, dtype=np.int32).swapaxes(1, 0)
        mb_opt_pi = np.asarray(mb_opt_pi, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_betas = np.asarray(mb_betas, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        # mb_masks = mb_dones[:, :-1]

        # discount/bootstrap off option value fn
        last_values = self.model.step_model.Qopt(self.obs).tolist()
        _, last_betas, _ = self.model.step(self.obs, self.opts)
        last_betas = last_betas.tolist()
        mb_Qu = []
        mb_Aohm = []
        for n, (opt_values, options, betas, dones, rewards, last_value, last_beta) in enumerate(zip(mb_opt_values, mb_options, mb_betas, mb_dones, mb_rewards, last_values, last_betas)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            betas = betas.tolist()
            options = options.tolist()
            # # last option value is bootstrapped
            betas += [last_beta]
            opt_values = np.concatenate((opt_values, np.expand_dims(last_value, axis=0)), axis=0)
            # betas -> (nsteps+1,) -> b1, b2, ..., b_nsteps, b_nsteps+1
            # opt_values -> (nsteps+1, noptions) -> Q_1, Q_2, ..., Q_nsteps, Q_nsteps+1
            # dones -> (nsteps+1,) -> d_0, d_1, ..., d_nsteps
            # options -> (nsteps,) -> w_0, w_1, ..., w_nsteps
            # reverse the option values
            Qu = []
            Aohm = []
            for i in range(self.nsteps, 0, -1):
                q = rewards[i-1] + (1-dones[i])*self.gamma*((1-betas[i])*opt_values[i,options[i-1]] + betas[i]*np.amax(opt_values[i]))
                Qu.append(q)
                a = (1 - dones[i-1]) * (opt_values[i-1, options[i-1]] - np.amax(opt_values[i-1]))
                Aohm.append(a)

            mb_Qu.append(Qu[::-1])
            mb_Aohm.append(Aohm[::-1])
        # print("option values")
        # print(mb_opt_values[0])
        # print("selected options")
        # print(mb_options[0])
        # print("option policies")
        # print(mb_opt_pi[0])
        # print("selected actions")
        # print(mb_actions[0])
        # print("option betas")
        # print(mb_betas[0])
        # print("terminations")
        # print(mb_dones[0][1:])
        # print("rewards")
        # print(mb_rewards[0])
        # print("option value targets")
        # print(mb_Qu[0])
        # print("termination advantage")
        # print(mb_Aohm[0])
        # self.exit_counter += 1
        # if self.exit_counter > 2:
            # exit()

        mb_Qu = np.asarray(mb_Qu, dtype=np.float32)
        mb_Aohm = np.asarray(mb_Aohm, dtype=np.float32)
        mb_options = mb_options[:,1:].flatten()
        mb_Qu = mb_Qu.flatten()
        mb_Aohm = mb_Aohm.flatten()
        mb_actions = mb_actions.flatten()
        # mb_betas = mb_betas.flatten()
        # mb_masks = mb_masks.flatten()
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

    runner = Runner(env, model, noptions=noptions, nsteps=nsteps, gamma=gamma)

    nbatch = nenvs*nsteps
    tstart = time.time()
    for update in range(1, total_timesteps//nbatch+1):
        obs, options, Qus, Aohms, actions = runner.run()
        policy_loss, term_loss, q_value_loss, policy_entropy = model.train(obs, options, Qus, Aohms, actions)
        nseconds = time.time()-tstart
        fps = int((update*nbatch)/nseconds)
        if update % log_interval == 0 or update == 1:
            # ev = explained_variance(values, rewards)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update*nbatch)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("q_value_loss", float(q_value_loss))
            # logger.record_tabular("explained_variance", float(ev))
            logger.dump_tabular()
    env.close()
