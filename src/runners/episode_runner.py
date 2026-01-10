from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import wandb

import copy
import random
import torch as th
import torch.nn.functional as F


class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self.wolfpack_returns = []
        self.wolfpack_stats = {}
        self.test_wolfpack_returns = []
        self.test_wolfpack_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac, qdifference_transformer, planning_transformer, timeseries_ode_model,obs_predictor):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.qdiff_transformer = qdifference_transformer
        self.planning_transformer = planning_transformer
        self.timeseries_ode_model = timeseries_ode_model
        self.obs_predictor = obs_predictor
    def setup_mac_for_attack(self, mac):
        self.mac_for_attack = copy.deepcopy(mac)
        self.mac_for_attack.cuda()

    def setup_learner(self, learner):
        self.learner = learner

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
                wandb.log({"epsilon": self.mac.action_selector.epsilon}, step=self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def followup_agent_group_selection_l2(self, batch, t, initial_agent, agents_set):
        obs = batch["obs"][:, t]
        initial_agent_obs = obs[:, initial_agent]

        obs_l2_distances = [(i, th.sum((initial_agent_obs - obs[:, i]) ** 2)) for i in agents_set]

        obs_l2_distances.sort(key=lambda x: x[1])
        followup_agents = [obs_l2_distances[i][0] for i in range(self.args.num_followup_agents)]

        return followup_agents

    def followup_agent_group_selection(self, batch, t, initial_agent, agents_set):
        actions = batch["actions"][:, t]
        state = batch["state"][:, t]
        attacker_actions = batch["attacker_actions"][:, t]
        hidden_states = self.mac.return_hidden()

        self.mac_for_attack.agent.load_state_dict(copy.deepcopy(self.learner.mac.agent.state_dict()))
        normal_mac, attack_mac = copy.deepcopy(self.mac_for_attack), copy.deepcopy(self.mac_for_attack)
        mixer = copy.deepcopy(self.learner.mixer)
        optimizer = th.optim.Adam(list(attack_mac.parameters()), lr=self.args.lr)

        indi_attack_q_first = attack_mac.forward_q_attack(batch, t, hidden_states.detach()).detach()
        indi_normal_q = normal_mac.forward_q_attack(batch, t, hidden_states.detach())
        indi_attack_q = attack_mac.forward_q_attack(batch, t, hidden_states.detach())

        normal_q = th.gather(indi_normal_q, dim=2, index=actions).squeeze(2)    # 1,

        do_actions = actions.clone().detach()
        for agent in initial_agent:
            do_actions[:, agent] = copy.deepcopy(attacker_actions[:, agent].detach())
        attack_q = th.gather(indi_attack_q, dim=2, index=do_actions).squeeze(2)

        if self.args.mixer == "vdn":
            normal_q = normal_q.unsqueeze(0)
            attack_q = attack_q.unsqueeze(0)
            total_normal_q = mixer(normal_q, state)
            total_attack_q = mixer(attack_q, state)
        elif self.args.mixer == "dmaq":
            total_normal_q = mixer(normal_q, state, is_v=True)
            total_attack_q = mixer(attack_q, state, is_v=True)
        else:
            total_normal_q = mixer(normal_q, state)
            total_attack_q = mixer(attack_q, state)

        loss = ((total_attack_q - total_normal_q.detach()) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        updated_attack_q = attack_mac.forward_q_attack(batch, t, hidden_states.detach())

        def normalize_q(q_values):
            q_mean = q_values.mean(dim=2, keepdim=True)
            q_std = q_values.std(dim=2, keepdim=True)
            normalized = (q_values - q_mean) / q_std
            return normalized / normalized.std(dim=2, keepdim=True)

        indi_attack_q_first_norm = th.nn.functional.softmax(normalize_q(indi_attack_q_first), dim=-1)
        updated_attack_q_norm = th.nn.functional.softmax(normalize_q(updated_attack_q), dim=-1)

        kl = [
            (i, th.sum(indi_attack_q_first_norm[:, i] * (th.log(indi_attack_q_first_norm[:, i]) - th.log(updated_attack_q_norm[:, i])), dim=-1).mean())
            for i in agents_set
        ]
        kl.sort(key=lambda x: x[1], reverse=True)

        followup_agents = [kl[i][0] for i in range(self.args.num_followup_agents)]

        return followup_agents

    def _flatten_joint_obs(self, obs):
        """
        Flatten joint observations for predictor input
        obs: [B, T, N, obs_shape] or [B, N, obs_shape]
        returns: [B, T, N*obs_shape] or [B, N*obs_shape]
        """
        if len(obs.shape) == 4:  # [B, T, N, obs_shape]
            B, T, N, O = obs.shape
            return obs.reshape(B, T, N * O)
        elif len(obs.shape) == 3:  # [B, N, obs_shape]
            B, N, O = obs.shape
            return obs.reshape(B, N * O)
        else:
            raise ValueError(f"Unexpected obs shape: {obs.shape}")
    
    def _flatten_joint_action(self, action_onehot):
        """
        Flatten joint actions (one-hot) for predictor input
        action_onehot: [B, T, N, n_actions] or [B, N, n_actions]
        returns: [B, T, N*n_actions] or [B, N*n_actions]
        """
        if len(action_onehot.shape) == 4:  # [B, T, N, n_actions]
            B, T, N, A = action_onehot.shape
            return action_onehot.reshape(B, T, N * A)
        elif len(action_onehot.shape) == 3:  # [B, N, n_actions]
            B, N, A = action_onehot.shape
            return action_onehot.reshape(B, N * A)
        else:
            raise ValueError(f"Unexpected action shape: {action_onehot.shape}")

    def select_followup_target_continuous(self, obs_fact, actions_fact, actions_cf, prev_attacked_agent_idx):
        """
        Select follow-up attack targets based on obs predictor + counterfactual reasoning.
        Constraint: cannot attack the same agent as the previous timestep.
        
        Args:
            obs_fact: [B, T, N, obs_shape] - factual observations (actually executed)
            actions_fact: [B, T, N, n_actions] - factual actions (one-hot)
            actions_cf: [B, T, N, n_actions] - counterfactual actions (one-hot)
            prev_attacked_agent_idx: int - index of agent attacked in previous timestep (to avoid)
        
        Returns:
            followup_targets: [B, T] - follow-up target agent indices for each timestep
        """
        B, T, N, O = obs_fact.shape
        device = obs_fact.device
        
        followup_targets = th.zeros(B, T, dtype=th.long, device=device)
        
        # Iterate through each timestep (except last, since we need t+1)
        for t in range(T - 1):
            # Get observations and actions at timestep t
            obs_t = obs_fact[:, t]  # [B, N, obs_shape]
            actions_t = actions_fact[:, t]  # [B, N, n_actions]
            actions_t_cf = actions_cf[:, t]  # [B, N, n_actions]
            
            # Flatten for predictor
            obs_t_joint = self._flatten_joint_obs(obs_t)  # [B, N*obs_shape]
            actions_t_joint = self._flatten_joint_action(actions_t)  # [B, N*n_actions]
            actions_t_cf_joint = self._flatten_joint_action(actions_t_cf)  # [B, N*n_actions]
            
            # Predict t+1 observations under factual and counterfactual actions
            with th.no_grad():
                obs_next_fact_pred = self.obs_predictor(obs_t_joint, actions_t_joint)  # [B, N*obs_shape]
                obs_next_cf_pred = self.obs_predictor(obs_t_joint, actions_t_cf_joint)  # [B, N*obs_shape]
            
            # Reshape back to per-agent
            obs_next_fact_pred = obs_next_fact_pred.reshape(B, N, O)  # [B, N, obs_shape]
            obs_next_cf_pred = obs_next_cf_pred.reshape(B, N, O)  # [B, N, obs_shape]
            
            # Compute scores for each agent (observation change magnitude)
            scores = th.zeros(B, N, device=device)
            
            for j in range(N):
                # Skip agent that was attacked in previous timestep
                if j == prev_attacked_agent_idx:
                    scores[:, j] = -float('inf')
                else:
                    # Score: how much agent j's obs changes between factual and cf
                    obs_diff_j = (obs_next_fact_pred[:, j] - obs_next_cf_pred[:, j]).abs().mean(dim=-1)
                    scores[:, j] = obs_diff_j
            
            # Select agent with highest score as follow-up target
            followup_target_t = scores.argmax(dim=1)  # [B]
            followup_targets[:, t] = followup_target_t
            
            # Update prev_attacked for next iteration
            prev_attacked_agent_idx = int(followup_target_t[0].item())
        
        # Last timestep: fallback to first non-attacked agent
        scores_last = th.zeros(B, N, device=device)
        for j in range(N):
            if j != prev_attacked_agent_idx:
                scores_last[:, j] = 1.0
            else:
                scores_last[:, j] = -float('inf')
        followup_targets[:, -1] = scores_last.argmax(dim=1)
        
        return followup_targets

    def get_attack_step(self, pre_initial_attack_step):
        state = self.batch["state"]

        agent_id = th.eye(self.args.n_agents, device=self.batch.device).unsqueeze(0).expand(1, -1, -1)

        if self.t < self.args.trans_input_len:
            B, T, F = state.size()
            time_step_temp = th.arange(start=0, end=self.args.trans_input_len, step=1, device=state.device).reshape(1, self.args.trans_input_len)

            now_state = th.zeros(1, self.args.trans_input_len, F, device=state.device)
            time_step = th.zeros(1, self.args.trans_input_len, device=state.device)

            now_state[:, self.args.trans_input_len - (self.t+1):, :] = state[:, :self.t + 1, :]
            time_step[:, self.args.trans_input_len - (self.t+1):] = time_step_temp[:, :self.t + 1]
        else:
            now_state = state[:, self.t-self.args.trans_input_len+1:self.t+1]
            time_step = th.arange(start=self.t, end=self.t+self.args.trans_input_len, step=1, device=state.device).reshape(1, self.args.trans_input_len)

        agent_id = agent_id.unsqueeze(1).repeat(1, now_state.shape[1], 1, 1)
        predict_q_diff_list = []
        for i in range(self.args.n_agents):
            predict_q_diff = self.qdiff_transformer(time_step, now_state, agent_id[:, :, i])
            predict_q_diff_list.append(predict_q_diff[:, -1])
        predict_q_diff_total = th.stack(predict_q_diff_list, dim=1)
        predict_q_diff, initial_agent = th.max(predict_q_diff_total, dim=1)
        predict_q_diff = predict_q_diff.squeeze(0).detach()  # [20]

        if self.t == 0:
            predict_q_diff = predict_q_diff[:]
        else:
            predict_q_diff = predict_q_diff[:-(min(self.t-pre_initial_attack_step, self.args.attack_period-1))]
            padding_size = self.args.attack_period - predict_q_diff.shape[0]
            if padding_size > 0:
                predict_q_diff = th.nn.functional.pad(predict_q_diff, (0, padding_size), mode='constant', value=-9999999)

        predict_q_diff_softmax = th.softmax(predict_q_diff / self.args.temperature, dim=-1)
        critical_step = th.multinomial(predict_q_diff_softmax, num_samples=1)

        if critical_step == 0:
            attack_prob = 1
            initial_agent = initial_agent.squeeze(0)[0]
        else:
            attack_prob = 0
            initial_agent = 0

        return attack_prob, initial_agent

    def run_wolfpack_attacker(self, test_mode=False):
        self.reset()
        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        if test_mode==True:
            attack_num = self.args.num_attack_test
        else:
            attack_num = self.args.num_attack_train

        attack_cnt = 0
        do_attack_num = 0
        initial_attack_flag = copy.deepcopy(self.args.attack_duration)

        pre_initial_attack_step = 0

        initial_agent = [self.args.n_agents]
        followup_agents = [self.args.n_agents, self.args.n_agents]

        while not terminated:
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()],
            }

            self.batch.update(pre_transition_data, ts=self.t)

            ori_actions, chosen_actions, random_actions, attacker_actions = self.mac.select_actions_wolfpack(self.batch,
                                                                            t_ep=self.t, t_env=self.t_env,
                                                                            test_mode=test_mode)

            hidden_states = self.mac.return_hidden()

            middle_transition_data = {
                "actions": ori_actions.to("cpu").numpy(),
                "attacker_actions": attacker_actions.to("cpu").numpy(),
                "hidden_states": hidden_states
            }

            self.batch.update(middle_transition_data, ts=self.t)

            if self.args.init_attack_step_method == "planner":
                attack_prob, chosen_agent = self.get_attack_step(pre_initial_attack_step)

                if self.args.init_agent_random == True:
                    initial_agent = random.sample(range(self.args.n_agents), 1)
                else:
                    initial_agent = [chosen_agent]

            do_actions = copy.deepcopy(ori_actions)
            prob = 1 / 10

            if initial_attack_flag == self.args.attack_duration:
                if self.args.init_attack_step_method == "random":
                    if random.random() < prob:
                        initial_agent = random.sample(range(self.args.n_agents), 1)
                        previous_initial_agent = copy.deepcopy(initial_agent)
                        for i in previous_initial_agent:
                            do_actions[:, i] = copy.deepcopy(attacker_actions[:, i])

                    if not ori_actions.equal(do_actions):
                        pre_initial_attack_step = self.t
                        attack_cnt += 1
                        initial_attack_flag = initial_attack_flag - 1

                elif self.args.init_attack_step_method == "planner":
                    if attack_prob == 1:
                        previous_initial_agent = copy.deepcopy(initial_agent)
                        for i in previous_initial_agent:
                            do_actions[:, i] = copy.deepcopy(attacker_actions[:, i])

                    if not ori_actions.equal(do_actions):
                        pre_initial_attack_step = self.t
                        attack_cnt += 1
                        initial_attack_flag = initial_attack_flag - 1
            else:
                for i in followup_agents_eval:
                    do_actions[:, i] = copy.deepcopy(attacker_actions[:, i])

                if initial_attack_flag <= 0:
                    initial_attack_flag = copy.deepcopy(self.args.attack_duration)
                else:
                    initial_attack_flag = initial_attack_flag - 1

                attack_cnt += 1

            if attack_cnt > attack_num:
                do_actions = ori_actions

            if not ori_actions.equal(do_actions):
                do_attack_num += 1

            reward, terminated, env_info = self.env.step(do_actions[0])
            episode_return += reward

            post_transition_data = {
                "actions": ori_actions.to("cpu").numpy(),
                "forced_actions": do_actions.to("cpu").numpy(),
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            all_agents = set(range(self.args.n_agents))
            if initial_attack_flag == self.args.attack_duration - 1:
                excluded_agents = set(agent for agent in previous_initial_agent)
            else:
                excluded_agents = set(agent for agent in initial_agent)
            agents_set = all_agents - excluded_agents

            if self.args.followup_l2 == True:
                followup_agents = self.followup_agent_group_selection_l2(self.batch, self.t, initial_agent, agents_set)
            elif self.args.followup_l2 == False:
                followup_agents = self.followup_agent_group_selection(self.batch, self.t, initial_agent, agents_set)
            if initial_attack_flag == self.args.attack_duration - 1:
                followup_agents_eval = copy.deepcopy(followup_agents)
            last_transition_data = {
                "initial_agent": [initial_agent],
                "followup_agents": [followup_agents],
            }
            self.batch.update(last_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        ori_actions, chosen_actions, random_actions, attacker_actions = self.mac.select_actions_wolfpack(self.batch,
                                                                            t_ep=self.t, t_env=self.t_env,
                                                                            test_mode=test_mode)

        self.batch.update({"actions": ori_actions.to("cpu").numpy()}, ts=self.t)
        self.batch.update({"forced_actions": ori_actions.to("cpu").numpy()}, ts=self.t)

        cur_stats = self.test_wolfpack_stats if test_mode else self.wolfpack_stats
        cur_returns = self.test_wolfpack_returns if test_mode else self.wolfpack_returns
        log_prefix = "test_Wolfpack_" if test_mode else "Wolfpack_"

        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        battle_won = env_info.get("battle_won", 0)

        self._log(cur_returns, cur_stats, log_prefix)

        if test_mode:
            wandb.log({"test wolfpack attack num": do_attack_num}, step=self.t_env)
        else:
            wandb.log({"wolfpack attack num": do_attack_num}, step=self.t_env)

        return self.batch

    def run_continuous_attack(self, test_mode=False):
        """
        Continuous follow-up attack:
        - time t: launch initial attack on a randomly chosen agent (respecting attack budget/interval)
        - time t+k (within attack_duration): select a single follow-up target based on obs predictor + counterfactual
        Uses learner.obs_predictor to predict obs under counterfactual scenarios.
        """
        self.reset()
        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        # attack budgeting consistent with wolfpack
        attack_num = self.args.num_attack_test if test_mode else self.args.num_attack_train
        attack_cnt = 0
        do_attack_num = 0
        attack_duration = self.args.attack_duration
        attack_interval = getattr(self.args, "attack_period", 1)

        last_attack_step = -attack_interval
        initial_attack_step = None
        initial_agent = None
        followup_targets_cached = []

        # history buffers
        obs_hist = []           # list of [N, obs]
        actions_clean_hist = [] # list of [N] int
        actions_fact_hist = []  # list of [N] int (after attack)

        while not terminated:
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()],
            }
            self.batch.update(pre_transition_data, ts=self.t)

            # clean policy actions
            ori_actions, chosen_actions, random_actions, attacker_actions = self.mac.select_actions_wolfpack(
                self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode
            )

            # store clean actions (int list) for history
            actions_int = ori_actions.clone().detach()

            # decide attack
            do_actions = copy.deepcopy(ori_actions)

            can_start_attack = (attack_cnt < attack_num) and (self.t - last_attack_step >= attack_interval)

            if initial_attack_step is None and can_start_attack:
                # launch initial attack randomly (or could use planner); here random for parity
                initial_agent = random.randrange(self.args.n_agents)
                do_actions[:, initial_agent] = copy.deepcopy(attacker_actions[:, initial_agent])
                initial_attack_step = self.t
                last_attack_step = self.t
                attack_cnt += 1
                followup_targets_cached = []  # reset cache
            elif initial_attack_step is not None and self.t - initial_attack_step < attack_duration and attack_cnt < attack_num:
                # follow-up selection: ensure we have enough history (at least 1 step after initial)
                if len(obs_hist) >= 1:
                    # build tensors for runner's predictor selector
                    B = 1
                    T_hist = len(obs_hist)
                    obs_tensor = th.tensor(obs_hist, dtype=th.float32, device=self.args.device).unsqueeze(0)  # [1,T,N,O]
                    act_clean_tensor = th.tensor(actions_clean_hist, dtype=th.long, device=self.args.device).unsqueeze(0)  # [1,T,N]
                    act_fact_tensor = th.tensor(actions_fact_hist, dtype=th.long, device=self.args.device).unsqueeze(0)  # [1,T,N]
                    act_fact_onehot = F.one_hot(act_fact_tensor, num_classes=self.args.n_actions).float()  # [1,T,N,A]
                    act_clean_onehot = F.one_hot(act_clean_tensor, num_classes=self.args.n_actions).float()  # [1,T,N,A]

                    # counterfactual: replace initial agent actions with clean ones (to measure its impact)
                    action_cf = act_fact_onehot.clone()
                    action_cf[:, :, initial_agent, :] = act_clean_onehot[:, :, initial_agent, :]

                    # Get the agent attacked in the last timestep (to avoid attacking same agent consecutively)
                    prev_attacked_agent = followup_targets_cached[-1] if followup_targets_cached else initial_agent

                    # Use obs_predictor to select follow-up target
                    followups = self.select_followup_target_continuous(
                        obs_fact=obs_tensor,
                        actions_fact=act_fact_onehot,
                        actions_cf=action_cf,
                        prev_attacked_agent_idx=prev_attacked_agent,
                    )  # [1,T]
                    target_now = int(followups[0, -1].item())
                    followup_targets_cached.append(target_now)
                    do_actions[:, target_now] = copy.deepcopy(attacker_actions[:, target_now])
                    attack_cnt += 1
                    last_attack_step = self.t
            else:
                # no attack this step
                pass

            # track stats
            if not ori_actions.equal(do_actions):
                do_attack_num += 1

            # env step
            reward, terminated, env_info = self.env.step(do_actions[0])
            episode_return += reward

            post_transition_data = {
                "actions": ori_actions.to("cpu").numpy(),
                "forced_actions": do_actions.to("cpu").numpy(),
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }
            self.batch.update(post_transition_data, ts=self.t)

            # record attack targets for learner logging
            # Format: initial_agent is [agent_id], followup_agents is list with num_followup_agents elements
            init_agent_store = [self.args.n_agents] if initial_agent is None else [initial_agent]
            followup_store = [followup_targets_cached[-1]] if followup_targets_cached else [self.args.n_agents]
            
            # Pad followup_agents to match num_followup_agents schema
            while len(followup_store) < self.args.num_followup_agents:
                followup_store.append(self.args.n_agents)
            
            self.batch.update({
                "initial_agent": [init_agent_store],
                "followup_agents": [followup_store],
            }, ts=self.t)

            # update histories
            obs_hist.append(pre_transition_data["obs"][0])
            actions_clean_hist.append(actions_int.squeeze(0).cpu().numpy().tolist())
            actions_fact_hist.append(do_actions.squeeze(0).cpu().numpy().tolist())

            # keep history bounded (optional): use attack_duration window
            max_hist = attack_duration + 1
            if len(obs_hist) > max_hist:
                obs_hist.pop(0); actions_clean_hist.pop(0); actions_fact_hist.pop(0)

            self.t += 1

        # tail state
        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()],
        }
        self.batch.update(last_data, ts=self.t)

        ori_actions, chosen_actions, random_actions, attacker_actions = self.mac.select_actions_wolfpack(
            self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode
        )
        self.batch.update({"actions": ori_actions.to("cpu").numpy()}, ts=self.t)
        self.batch.update({"forced_actions": ori_actions.to("cpu").numpy()}, ts=self.t)

        # logging
        cur_stats = self.test_wolfpack_stats if test_mode else self.wolfpack_stats
        cur_returns = self.test_wolfpack_returns if test_mode else self.wolfpack_returns
        log_prefix = "test_Continuous_" if test_mode else "Continuous_"
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)
        if not test_mode:
            self.t_env += self.t
        cur_returns.append(episode_return)
        self._log(cur_returns, cur_stats, log_prefix)

        wandb.log({f"{log_prefix.lower()}attack_num": do_attack_num}, step=self.t_env)

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        wandb.log({prefix + "return_mean": np.mean(returns)}, step=self.t_env)
        wandb.log({prefix + "return_std": np.std(returns)}, step=self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean", v / stats["n_episodes"], self.t_env)
                wandb.log({prefix + k + "_mean": v / stats["n_episodes"]}, step=self.t_env)
        stats.clear()

    # -----------------------
    # ODE 相关辅助方法
    # -----------------------
    def _one_hot_actions(self, actions_idx, n_actions):
        # actions_idx: [B, T, N, 1] 或 [B, T, N]
        if actions_idx.dim() == 4:
            actions_idx = actions_idx.squeeze(-1)
        assert actions_idx.dim() == 3, f"actions_idx expected [B,T,N,1], got {actions_idx.shape}"
        B, T, N = actions_idx.shape
        oh = F.one_hot(actions_idx.long(), num_classes=n_actions)  # [B,T,N,A]
        return oh.float()

    def _pad_left_time(self, x, target_T):
        # x: [B, t, ...], 左侧补 0 到 target_T
        B, t = x.shape[:2]
        if t >= target_T:
            return x[:, -target_T:]
        pad_shape = list(x.shape)
        pad_shape[1] = target_T - t
        pad = th.zeros(pad_shape, device=x.device, dtype=x.dtype)
        return th.cat([pad, x], dim=1)

    def _build_history_nodes(self, batch, t, window_size):
        """
        构造历史窗口节点特征: [B, W, N, obs+n_actions]
        - 历史 obs 用 batch["obs"]（到 t）。
        - 历史动作优先用 actions_onehot；否则 forced_actions_onehot；
        再不行用 actions/forced_actions 索引转 onehot；都没有时用 0 占位。
        """
        B = batch.batch_size
        N = self.args.n_agents
        A = self.args.n_actions

        # ---- 历史obs ----
        obs_all = batch["obs"]                 # [B, T+1, N, obs_dim]
        t_obs = obs_all[:, :t+1]               # [B, t+1, N, obs_dim]

        # ---- 历史动作（对齐到 :t）----
        def one_hot_idx(idx_tensor):
            # idx_tensor: [B, T, N, 1] 或 [B, T, N]
            if idx_tensor.dim() == 4:
                idx_tensor = idx_tensor.squeeze(-1)
            oh = F.one_hot(idx_tensor.long(), num_classes=A)  # [B, T, N, A]
            return oh.float()

        t_act_oh = None
        if hasattr(batch.data, "actions_onehot"):
            ao = batch["actions_onehot"]      # [B, T, N, A] 或 [B, T+1, N, A]
            t_act_oh = ao[:, :t] if ao.size(1) >= t else ao[:, :0]
        elif hasattr(batch.data, "forced_actions_onehot"):
            fao = batch["forced_actions_onehot"]
            t_act_oh = fao[:, :t] if fao.size(1) >= t else fao[:, :0]
        elif hasattr(batch.data, "actions"):
            ai = batch["actions"]             # [B, T+1, N, 1]
            t_act_oh = one_hot_idx(ai[:, :t]) # [B, t, N, A]
        elif hasattr(batch.data, "forced_actions"):
            fai = batch["forced_actions"]
            t_act_oh = one_hot_idx(fai[:, :t])
        else:
            # 什么都没有（例如 t==0 或你在写入前就调用了该函数）——用 0 占位
            t_act_oh = batch["obs"].new_zeros((B, 0, N, A))  # 空时间维

        # ---- 左侧补0，长度对齐到 W ----
        obs_hist = self._pad_left_time(t_obs, window_size)[:, -window_size:]       # [B, W, N, obs]
        act_hist = self._pad_left_time(t_act_oh, window_size)[:, -window_size:]    # [B, W, N, A]

        # ---- 断言检查 ----
        assert obs_hist.shape[:3] == (B, window_size, N), \
            f"obs_hist shape {tuple(obs_hist.shape)} != (B,W,N,obs)"
        assert act_hist.shape[:3] == (B, window_size, N), \
            f"act_hist shape {tuple(act_hist.shape)} != (B,W,N,A)"

        # ---- 拼接 ----
        node_in = th.cat([obs_hist, act_hist], dim=-1)  # [B, W, N, obs + A]
        exp_feat = self.args.obs_shape + self.args.n_actions
        assert node_in.shape[-1] == exp_feat, \
            f"node_in feat={node_in.shape[-1]} != obs+n_actions({exp_feat})"

        return node_in



    def _encode_z0(self, node_history):
        """用 GNN encoder 得 z0: [B,N,ode_hidden_dim]
        node_history: [B, W, N, D]
        """
        from torch_geometric.data import Data, Batch as GeoBatch
        
        B, W, N, D = node_history.shape
        device = node_history.device
        
        # 构建图数据（参考 load_data_covid.py:transfer_one_graph）
        graph_list = []
        for b in range(B):
            # 1. 展平时间维度
            x = node_history[b].reshape(W * N, D)  # [W*N, D]
            
            # 2. 时间位置编码
            pos = th.cat([th.full((N,), t / W, device=device) for t in range(W)])  # [W*N]
            
            # 3. 构建全连接边索引（每个时间步内 + 跨时间）
            edge_list = []
            edge_weights = []
            edge_times = []
            
            for t in range(W):
                offset = t * N
                for i in range(N):
                    for j in range(N):
                        # 同一时间步内的边
                        edge_list.append([offset + i, offset + j])
                        edge_weights.append(1.0 / N)
                        edge_times.append(0.0)
                        
                        # 跨时间边（t -> t-1）
                        if t > 0:
                            prev_offset = (t - 1) * N
                            edge_list.append([offset + i, prev_offset + j])
                            edge_weights.append(0.5 / N)
                            edge_times.append(-1.0 / W)
            
            edge_index = th.tensor(edge_list, dtype=th.long).t().to(device)
            edge_weight = th.tensor(edge_weights, dtype=th.float, device=device)
            edge_time = th.tensor(edge_times, dtype=th.float, device=device)
            
            # 4. y：每个 agent 有 W 个观测
            y = th.full((N,), W, dtype=th.long, device=device)
            
            graph_data = Data(
                x=x,
                edge_index=edge_index,
                edge_weight=edge_weight,
                edge_time=edge_time,
                pos=pos,
                y=y
            )
            graph_list.append(graph_data)
        
        # 5. 合并成 Batch
        batch_geo = GeoBatch.from_data_list(graph_list)
        
        # 6. 调用 encoder（传入完整参数）
        z0_mu, z0_std = self.timeseries_ode_model.encoder_z0(
            batch_geo.x,         # [B*W*N, D]
            batch_geo.edge_weight,
            batch_geo.edge_index,
            batch_geo.pos,       # [B*W*N]
            batch_geo.edge_time,
            batch_geo.batch,     # PyG 自动生成的 batch 索引
            batch_geo.y          # [B*N]，每个 agent 的时间步数
        )
        
        # 7. 重塑回 [B, N, ode_hidden_dim]
        H = self.args.ode_dims
        assert z0_mu.shape == (B * N, H), f"encoder output shape {z0_mu.shape} != [{B*N}, {H}]"
        z0_mu = z0_mu.reshape(B, N, H)
        z0_std = z0_std.reshape(B, N, H)
        
        # 简单用 mean（可改成 reparameterize）
        z0 = z0_mu
        return z0

    def _ode_rollout_edges(self, z0, horizon, treatment=None):
        """从 z0 rollout 未来 H 步的边权: 返回 [B,H,N,N]"""
        assert hasattr(self, "timeseries_ode_model"), "timeseries_ode_model not set"
        model = self.timeseries_ode_model

        B, N, Hdim = z0.shape
        assert Hdim == self.args.ode_dims, f"z0 hidden {Hdim} != ode_dims {self.args.ode_dims}"

        # 时间轴：归一化到 [0, 1]
        time_steps = th.linspace(0, 1, horizon, device=z0.device)  # [H]

        # treatment: [B,H,N,t_dim]
        tdim = getattr(self.args, "t_dim", 1)
        if treatment is None:
            treatment = th.zeros(B, horizon, N, tdim, device=z0.device)

        # 将 treatment 转置：[B,H,N,t_dim] -> [N,H,t_dim]（只用第一个batch）
        treatment_for_solver_shape = treatment[0].permute(1, 0, 2)  # [N, H, t_dim]
        
        # 扩展到 max_T（用0填充）
        max_T = getattr(self.args, 'episode_limit', 100)
        treatment_for_solver = th.zeros(N, max_T, tdim, device=z0.device)
        treatment_for_solver[:, :horizon, :] = treatment_for_solver_shape
        
        # 设置 treatment 到 ODE solver（必须在 set_index_and_graph 之前）
        model.set_treatments(treatment_for_solver)
        
        # 关键新增：设置 policy_starting_ending_points
        # 假设每个 agent 的 policy 从时间 0 开始，到 max_T 结束
        # [N, t_dim, 2]：对于每个 agent 的每个 treatment 维度，记录 [start, end]
        policy_starting_ending_points = th.zeros(N, tdim, 2, device=z0.device)
        policy_starting_ending_points[:, :, 0] = 0          # 开始时间
        policy_starting_ending_points[:, :, 1] = max_T - 1  # 结束时间
        model.set_policy_starting_ending_points(policy_starting_ending_points)
        
        # z0: [B,N,Hdim] -> [B*N,Hdim]
        z0_flat = z0.reshape(B * N, Hdim)
        
        # time_absolute: [B*N,H]
        time_absolute = th.arange(horizon, device=z0.device, dtype=th.float).unsqueeze(0).repeat(B * N, 1)
        
        # 设置索引
        K_N = B * N
        K = B
        model.diffeq_solver.ode_func.set_index_and_graph(K_N, K, time_steps)
        
        # 设置 t_treatments
        model.diffeq_solver.ode_func.set_t_treatments(time_absolute)
        
        # 调用 solver
        try:
            sol_y, K_N, treatment_rep = model.diffeq_solver(
                first_point=z0_flat,
                time_steps_to_predict=time_steps,
                times_absolute=time_absolute,
                w_node_to_edge_initial=model.w_node_to_edge_initial
            )
            
            # 提取边的轨迹
            edge_latent = sol_y[K_N:, :, :]  # [K*N*N, H, D]
            
            # 通过 decoder 解码
            pred_edge = model.decoder_edge(edge_latent)  # [K*N*N, H, 1]
            
            # 重塑为 [B, H, N, N]
            pred_edge = pred_edge.squeeze(-1)  # [K*N*N, H]
            pred_edge = pred_edge.reshape(B, N, N, horizon)  # [B, N, N, H]
            pred_edge = pred_edge.permute(0, 3, 1, 2)  # [B, H, N, N]
            
        except Exception as e:
            print(f"[ERROR] ODE rollout failed: {e}")
            print(f"  z0 shape: {z0.shape}")
            print(f"  treatment shape: {treatment.shape}")
            print(f"  treatment_for_solver shape: {treatment_for_solver.shape}")
            raise e
        
        # 去掉自环
        eye = th.eye(N, device=pred_edge.device).view(1, 1, N, N)
        pred_edge = pred_edge * (1.0 - eye)
        
        return pred_edge  # [B,H,N,N]

    def _select_initial_by_total_outgoing(self, edges_future):
        """在 H 步窗口内按出边和累加选择 initial agent"""
        # edges_future: [B,H,N,N]
        B, H, N, _ = edges_future.shape
        outgoing = edges_future.sum(dim=-1).sum(dim=1)  # [B,N], sum over j and time
        scores, idx = th.max(outgoing, dim=-1)  # [B]
        return idx, scores, outgoing  # 均是 [B]

    def _inject_initial_treatment(self, B, H, N, initial_idx):
        """构造干预: 对 initial agent 在整个未来窗口置 1（t_dim=1）"""
        tdim = getattr(self.args, "t_dim", 1)
        assert tdim == 1, "建议 t_dim=1；否则在这里构建 one-hot/multi-dim 干预"
        treatment = th.zeros(B, H, N, tdim, device=self.batch.device)
        for b in range(B):
            i = int(initial_idx[b].item())
            treatment[b, :, i, 0] = 1.0
        return treatment  # [B,H,N,1]

    def _select_followup_each_step(self, edges_future_with_attack, initial_idx, num_steps):
        """
        按 time step 选出边权最大的 **2 个 agent** 作为 follow-up（每步选 2 个）
        edges_future_with_attack: [B, H, N, N]
        initial_idx: [B]
        num_steps: 选择的时间步数（attack_duration）
        返回: [B, num_steps, 2] - 每个时间步 2 个 agent ID
        """
        B, H, N, _ = edges_future_with_attack.shape
        outgoing_t = edges_future_with_attack.sum(dim=-1)  # [B, H, N] - 每个时刻每个节点的出边和
        followups = []
        
        for t in range(min(H, num_steps)):
            scores = outgoing_t[:, t, :]  # [B, N]
            
            # Mask initial agent（排除自己）
            for b in range(B):
                scores[b, int(initial_idx[b])] = -1e9
            
            # 选择 top-2
            top2_idx = th.topk(scores, k=2, dim=-1).indices  # [B, 2]
            followups.append(top2_idx)
        
        # 如果 num_steps > H，用最后一步的 top-2 填充
        if len(followups) < num_steps:
            last_followup = followups[-1] if followups else th.full((B, 2), initial_idx[0].item(), device=edges_future_with_attack.device)
            for _ in range(num_steps - len(followups)):
                followups.append(last_followup)
        
        followups = th.stack(followups, dim=1)  # [B, num_steps, 2]
        return followups

    # ============================================================
    # ODE：initial + follow-up 的攻击器（使用 H = args.trans_input_len）
    # ============================================================
    def run_timeseries_attacker(self, test_mode=False):
        """
        用 ODE 模型选择 initial 和 follow-up 攻击目标
        ✅ 新增：最小攻击间隔控制
        """
        self.reset()
        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        # 统计
        do_attack_num = 0
        attack_cnt = 0
        initial_attack_flag = copy.deepcopy(self.args.attack_duration)  # 倒计时
        pre_initial_attack_step = 0

        # ✅ 新增：攻击间隔控制
        attack_period = getattr(self.args, 'attack_period', 10)  # 默认 10 步间隔
        last_attack_step = -attack_period  # 初始化为可以立即攻击

        # 初始化攻击目标（非法 ID）
        initial_agent = [self.args.n_agents]
        dynamic_followup_agents = [self.args.n_agents, self.args.n_agents]
        dynamic_followup_agents_full = [[self.args.n_agents, self.args.n_agents]] * self.args.attack_duration

        H = self.args.trans_input_len  # ODE 预测窗口
        max_attacks = getattr(self.args, 'num_attack_test', 10) if test_mode else getattr(self.args, 'num_attack_train', 10)

        while not terminated:
            # 1) 收集环境数据
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()],
            }
            self.batch.update(pre_transition_data, ts=self.t)

            ori_actions, chosen_actions, random_actions, attacker_actions = self.mac.select_actions_wolfpack(
                self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode
            )
            hidden_states = self.mac.return_hidden()

            middle_transition_data = {
                "actions": ori_actions.to("cpu").numpy(),
                "attacker_actions": attacker_actions.to("cpu").numpy(),
                "hidden_states": hidden_states,
            }
            self.batch.update(middle_transition_data, ts=self.t)

            # 2) 攻击逻辑
            do_actions = copy.deepcopy(ori_actions)

            # ✅ 修复1: 只在满足间隔要求且攻击周期开始时选择目标
            can_attack = (self.t - last_attack_step >= attack_period)  # 是否满足最小间隔
            
            if initial_attack_flag == self.args.attack_duration and attack_cnt < max_attacks and can_attack:
                # 2.1 构造历史窗口
                node_hist = self._build_history_nodes(batch=self.batch, t=self.t, window_size=H)
                node_hist = node_hist.to(self.args.device)
                z0 = self._encode_z0(node_hist)

                # 2.2 rollout 无干预的边权
                edges_future_base = self._ode_rollout_edges(z0, H, treatment=None)

                # 2.3 选择 initial agent
                initial_idx, _, _ = self._select_initial_by_total_outgoing(edges_future_base)
                initial_agent = [int(initial_idx[0].item())]

                # 2.4 注入干预，重新 rollout
                treatment_init = self._inject_initial_treatment(
                    B=self.batch.batch_size,
                    H=H,
                    N=self.args.n_agents,
                    initial_idx=initial_idx
                )
                edges_future_attack = self._ode_rollout_edges(z0, H, treatment=treatment_init)

                # 2.5 选择 follow-up agents（每步 2 个）
                followup_idx_seq = self._select_followup_each_step(
                    edges_future_attack, initial_idx, num_steps=self.args.attack_duration
                )  # [B, attack_duration, 2]

                # 2.6 构造完整攻击序列
                dynamic_followup_agents_full = []
                for k in range(followup_idx_seq.shape[1]):
                    step_agents = [
                        int(followup_idx_seq[0, k, 0].item()),
                        int(followup_idx_seq[0, k, 1].item())
                    ]
                    dynamic_followup_agents_full.append(step_agents)
                
                dynamic_followup_agents = dynamic_followup_agents_full[0]  # 记录第一步（供 Scheme）

                # ✅ 修复2: 执行 initial 攻击并更新状态
                do_actions[:, initial_agent[0]] = copy.deepcopy(attacker_actions[:, initial_agent[0]])
                initial_attack_flag -= 1
                attack_cnt += 1
                pre_initial_attack_step = self.t
                last_attack_step = self.t  # ← 更新最后攻击时间

            elif initial_attack_flag > 0 and attack_cnt < max_attacks:
                # ✅ 修复3: Follow-up 阶段
                step_offset = self.args.attack_duration - initial_attack_flag
                
                if 0 <= step_offset < len(dynamic_followup_agents_full):
                    followup_pair = dynamic_followup_agents_full[step_offset]
                    for f_id in followup_pair:
                        if f_id < self.args.n_agents:
                            do_actions[:, f_id] = copy.deepcopy(attacker_actions[:, f_id])
            
                initial_attack_flag -= 1
                attack_cnt += 1
                last_attack_step = self.t  # ← 更新最后攻击时间

            elif initial_attack_flag == 0:
                # ✅ 修复4: 攻击周期结束，重置
                initial_attack_flag = self.args.attack_duration
                initial_agent = [self.args.n_agents]
                dynamic_followup_agents_full = [[self.args.n_agents, self.args.n_agents]] * self.args.attack_duration
                dynamic_followup_agents = [self.args.n_agents, self.args.n_agents]
                # ⚠️ 注意：不更新 last_attack_step，保持间隔控制

            # 3) 统计实际攻击次数
            if not ori_actions.equal(do_actions):
                do_attack_num += 1

            # 4) 与环境交互
            reward, terminated, env_info = self.env.step(do_actions[0])
            episode_return += reward

            post_transition_data = {
                "actions": ori_actions.to("cpu").numpy(),
                "forced_actions": do_actions.to("cpu").numpy(),
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)]
            }
            self.batch.update(post_transition_data, ts=self.t)

            # 记录攻击目标
            last_transition_data = {
                "initial_agent": [initial_agent],
                "followup_agents": [dynamic_followup_agents],
            }
            self.batch.update(last_transition_data, ts=self.t)

            self.t += 1

        # 收尾
        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()],
        }
        self.batch.update(last_data, ts=self.t)

        ori_actions, chosen_actions, random_actions, attacker_actions = self.mac.select_actions_wolfpack(
            self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode
        )
        self.batch.update({"actions": ori_actions.to("cpu").numpy()}, ts=self.t)
        self.batch.update({"forced_actions": ori_actions.to("cpu").numpy()}, ts=self.t)

        # 日志
        cur_stats = self.test_wolfpack_stats if test_mode else self.wolfpack_stats
        cur_returns = self.test_wolfpack_returns if test_mode else self.wolfpack_returns
        log_prefix = "test_ODE_Timeseries_" if test_mode else "ODE_Timeseries_"
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)
        if not test_mode:
            self.t_env += self.t
        cur_returns.append(episode_return)
        self._log(cur_returns, cur_stats, log_prefix)

        if test_mode:
            wandb.log({"test_ode_timeseries_attack_num": do_attack_num}, step=self.t_env)
        else:
            wandb.log({"ode_timeseries_attack_num": do_attack_num}, step=self.t_env)

        return self.batch
