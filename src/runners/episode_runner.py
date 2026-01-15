from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import wandb

import copy
import random
import torch as th
import torch.nn.functional as F


def _to_idx_list(x, n_agents: int):
    """
    Normalize action tensor/list to python list of length N (each element int action index).
    Handles shapes: [1,N,1], [1,N], [N,1], [N], list[N], list[[1],...]
    """
    if isinstance(x, list):
        if len(x) == 0:
            return [0] * n_agents
        if isinstance(x[0], list):
            return [int(v[0]) for v in x]
        return [int(v) for v in x]

    if isinstance(x, th.Tensor):
        a = x.detach()
        if a.dim() >= 1 and a.size(0) == 1:
            a = a.squeeze(0)
        if a.dim() >= 2 and a.size(-1) == 1:
            a = a.squeeze(-1)
        a = a.reshape(-1)  # [N]
        return [int(v.item()) for v in a]

    raise TypeError(f"Unsupported action type: {type(x)}")


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

        self.cont_returns = []
        self.cont_stats = {}
        self.test_cont_returns = []
        self.test_cont_stats = {}

        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac, qdifference_transformer, planning_transformer, obs_predictor):
        self.new_batch = partial(
            EpisodeBatch,
            scheme,
            groups,
            self.batch_size,
            self.episode_limit + 1,
            preprocess=preprocess,
            device=self.args.device,
        )
        self.mac = mac
        self.qdiff_transformer = qdifference_transformer
        self.planning_transformer = planning_transformer
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

    # ============================================================
    # Normal Runner
    # ============================================================
    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()],
            }
            self.batch.update(pre_transition_data, ts=self.t)

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
            "obs": [self.env.get_obs()],
        }
        self.batch.update(last_data, ts=self.t)

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
                if getattr(self.args, "use_wandb", False):
                    wandb.log({"epsilon": self.mac.action_selector.epsilon}, step=self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    # ============================================================
    # Wolfpack Follow-up selection helpers (your original)
    # ============================================================
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

        normal_q = th.gather(indi_normal_q, dim=2, index=actions).squeeze(2)

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
            (
                i,
                th.sum(
                    indi_attack_q_first_norm[:, i]
                    * (th.log(indi_attack_q_first_norm[:, i]) - th.log(updated_attack_q_norm[:, i])),
                    dim=-1,
                ).mean(),
            )
            for i in agents_set
        ]
        kl.sort(key=lambda x: x[1], reverse=True)
        followup_agents = [kl[i][0] for i in range(self.args.num_followup_agents)]
        return followup_agents

    # ============================================================
    # Qdiff planner initial-step (your original)
    # ============================================================
    def get_attack_step(self, pre_initial_attack_step):
        state = self.batch["state"]
        agent_id = th.eye(self.args.n_agents, device=self.batch.device).unsqueeze(0).expand(1, -1, -1)

        if self.t < self.args.trans_input_len:
            _, _, F = state.size()
            time_step_temp = th.arange(
                start=0, end=self.args.trans_input_len, step=1, device=state.device
            ).reshape(1, self.args.trans_input_len)

            now_state = th.zeros(1, self.args.trans_input_len, F, device=state.device)
            time_step = th.zeros(1, self.args.trans_input_len, device=state.device)

            now_state[:, self.args.trans_input_len - (self.t + 1):, :] = state[:, : self.t + 1, :]
            time_step[:, self.args.trans_input_len - (self.t + 1):] = time_step_temp[:, : self.t + 1]
        else:
            now_state = state[:, self.t - self.args.trans_input_len + 1: self.t + 1]
            time_step = th.arange(
                start=self.t, end=self.t + self.args.trans_input_len, step=1, device=state.device
            ).reshape(1, self.args.trans_input_len)

        agent_id = agent_id.unsqueeze(1).repeat(1, now_state.shape[1], 1, 1)

        predict_q_diff_list = []
        for i in range(self.args.n_agents):
            predict_q_diff = self.qdiff_transformer(time_step, now_state, agent_id[:, :, i])
            predict_q_diff_list.append(predict_q_diff[:, -1])

        predict_q_diff_total = th.stack(predict_q_diff_list, dim=1)
        predict_q_diff, initial_agent = th.max(predict_q_diff_total, dim=1)
        predict_q_diff = predict_q_diff.squeeze(0).detach()

        if self.t == 0:
            predict_q_diff = predict_q_diff[:]
        else:
            predict_q_diff = predict_q_diff[: -(min(self.t - pre_initial_attack_step, self.args.attack_period - 1))]
            padding_size = self.args.attack_period - predict_q_diff.shape[0]
            if padding_size > 0:
                predict_q_diff = th.nn.functional.pad(
                    predict_q_diff, (0, padding_size), mode="constant", value=-9999999
                )

        predict_q_diff_softmax = th.softmax(predict_q_diff / self.args.temperature, dim=-1)
        critical_step = th.multinomial(predict_q_diff_softmax, num_samples=1)

        if critical_step == 0:
            attack_prob = 1
            initial_agent = initial_agent.squeeze(0)[0]
        else:
            attack_prob = 0
            initial_agent = 0

        return attack_prob, int(initial_agent)

    # ============================================================
    # Wolfpack Attacker (kept; safe legal ids)  [unchanged]
    # ============================================================
    def run_wolfpack_attacker(self, test_mode=False):
        self.reset()
        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        attack_num = self.args.num_attack_test if test_mode else self.args.num_attack_train
        attack_cnt = 0
        do_attack_num = 0
        initial_attack_flag = copy.deepcopy(self.args.attack_duration)
        pre_initial_attack_step = 0

        initial_agent = [0]
        followup_agents = [0 for _ in range(self.args.num_followup_agents)]
        followup_agents_eval = copy.deepcopy(followup_agents)

        while not terminated:
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

            if self.args.init_attack_step_method == "planner":
                attack_prob, chosen_agent = self.get_attack_step(pre_initial_attack_step)
                if self.args.init_agent_random:
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
            if len(agents_set) == 0:
                agents_set = all_agents

            if self.args.followup_l2:
                followup_agents = self.followup_agent_group_selection_l2(self.batch, self.t, initial_agent, agents_set)
            else:
                followup_agents = self.followup_agent_group_selection(self.batch, self.t, initial_agent, agents_set)

            followup_agents = [(int(a) % self.args.n_agents) for a in followup_agents]
            if len(followup_agents) < self.args.num_followup_agents:
                followup_agents += [followup_agents[-1]] * (self.args.num_followup_agents - len(followup_agents))
            followup_agents = followup_agents[: self.args.num_followup_agents]

            if initial_attack_flag == self.args.attack_duration - 1:
                followup_agents_eval = copy.deepcopy(followup_agents)

            max_followup_size = max(
                getattr(self.args, "num_followup_agents", 1),
                getattr(self.args, "num_followup_agents", 1),
            )
            followup_agents_padded = followup_agents[:]
            if len(followup_agents_padded) < max_followup_size:
                followup_agents_padded += [followup_agents_padded[-1]] * (max_followup_size - len(followup_agents_padded))
            followup_agents_padded = followup_agents_padded[:max_followup_size]

            last_transition_data = {
                "initial_agent": [initial_agent],
                "followup_agents": [followup_agents_padded],
            }
            self.batch.update(last_transition_data, ts=self.t)

            self.t += 1

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

        cur_stats = self.test_wolfpack_stats if test_mode else self.wolfpack_stats
        cur_returns = self.test_wolfpack_returns if test_mode else self.wolfpack_returns
        log_prefix = "test_Wolfpack_" if test_mode else "Wolfpack_"

        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t
        cur_returns.append(episode_return)
        if test_mode:
            if len(cur_returns) == self.args.test_nepisode:
                self._log(cur_returns, cur_stats, log_prefix)
        else:
            self._log(cur_returns, cur_stats, log_prefix)

        if getattr(self.args, "use_wandb", False):
            if test_mode:
                wandb.log({"test wolfpack attack num": do_attack_num}, step=self.t_env)
            else:
                wandb.log({"wolfpack attack num": do_attack_num}, step=self.t_env)

        return self.batch

    # ============================================================
    # Continuous Attack helpers (your original)
    # ============================================================
    def _flatten_joint_obs(self, obs):
        if len(obs.shape) == 4:
            B, T, N, O = obs.shape
            return obs.reshape(B, T, N * O)
        elif len(obs.shape) == 3:
            B, N, O = obs.shape
            return obs.reshape(B, N * O)
        else:
            raise ValueError(f"Unexpected obs shape: {obs.shape}")

    def _flatten_joint_action(self, action_onehot):
        if len(action_onehot.shape) == 4:
            B, T, N, A = action_onehot.shape
            return action_onehot.reshape(B, T, N * A)
        elif len(action_onehot.shape) == 3:
            B, N, A = action_onehot.shape
            return action_onehot.reshape(B, N * A)
        else:
            raise ValueError(f"Unexpected action shape: {action_onehot.shape}")

    @th.no_grad()
    def _policy_from_obs_preaction(self, obs_t, pre_a_onehot, hidden_state, avail_actions_t, test_mode=True):
        obs_in = obs_t.unsqueeze(1)          # [B,1,N,O]
        pre_a_in = pre_a_onehot.unsqueeze(1) # [B,1,N,A]

        logits = self.learner.mac.forward_action(
            self.batch,
            obs=obs_in,
            pre_actions=pre_a_in,
            hidden_state=hidden_state,
            attack=False,
            test_mode=test_mode,
            forced=False,
        )  # [B,1,N,A]
        logits = logits.squeeze(1)  # [B,N,A]

        logits = logits.clone()
        logits[avail_actions_t == 0] = -1e10
        probs = th.softmax(logits, dim=-1)
        return probs

    @th.no_grad()
    def _select_followup_by_one_step_KL(
        self,
        obs_tm1_np,
        obs_t_np,
        a_tm1_fact_idx,
        a_tm1_clean_idx,
        initial_agent_idx,
        hidden_before,
        avail_t,
        prev_attacked,
    ):
        N = self.args.n_agents
        A = self.args.n_actions
        K = int(getattr(self.args, "num_followup_agents", 1))

        obs_tm1 = th.tensor(np.asarray(obs_tm1_np, dtype=np.float32), device=self.args.device).unsqueeze(0)   # [1,N,O]
        obs_t_fact = th.tensor(np.asarray(obs_t_np, dtype=np.float32), device=self.args.device).unsqueeze(0) # [1,N,O]

        a_tm1_fact = th.tensor(a_tm1_fact_idx, dtype=th.long, device=self.args.device).unsqueeze(0)   # [1,N]
        a_tm1_clean = th.tensor(a_tm1_clean_idx, dtype=th.long, device=self.args.device).unsqueeze(0) # [1,N]

        a_tm1_cf = a_tm1_fact.clone()
        a_tm1_cf[:, initial_agent_idx] = a_tm1_clean[:, initial_agent_idx]

        a_tm1_fact_oh = F.one_hot(a_tm1_fact, num_classes=A).float()  # [1,N,A]
        a_tm1_cf_oh = F.one_hot(a_tm1_cf, num_classes=A).float()      # [1,N,A]

        obs_tm1_joint = self._flatten_joint_obs(obs_tm1)               # [1, N*O]
        a_tm1_cf_joint = self._flatten_joint_action(a_tm1_cf_oh)       # [1, N*A]
        obs_t_cf_joint = self.obs_predictor(obs_tm1_joint, a_tm1_cf_joint)  # [1, N*O]
        obs_t_cf = obs_t_cf_joint.reshape(1, N, -1)                    # [1,N,O]

        pi_fact = self._policy_from_obs_preaction(
            obs_t=obs_t_fact,
            pre_a_onehot=a_tm1_fact_oh,
            hidden_state=hidden_before,
            avail_actions_t=avail_t,
            test_mode=True,
        )
        pi_cf = self._policy_from_obs_preaction(
            obs_t=obs_t_cf,
            pre_a_onehot=a_tm1_cf_oh,
            hidden_state=hidden_before,
            avail_actions_t=avail_t,
            test_mode=True,
        )

        scores = (pi_fact * (th.log(pi_fact + 1e-10) - th.log(pi_cf + 1e-10))).sum(dim=-1).squeeze(0)  # [N]
        for j in prev_attacked:
            if 0 <= j < N:
                scores[j] = -1e9

        k = min(K, N)
        topk = th.topk(scores, k=k).indices.tolist()
        topk = [(int(x) % N) for x in topk]
        if len(topk) < K:
            topk += [topk[-1]] * (K - len(topk))
        topk = topk[:K]
        return topk

    # ============================================================
    # Continuous Attack (FIXED to match your requirement)
    # ============================================================
    def run_continuous_attack(self, test_mode=False):
        """
        Requirement-aligned continuous attack:
        1) Planner ONLY decides when to start initial attack + who to attack (attack_prob==1).
        2) Once initial starts, follow-up attack happens EVERY subsequent timestep (consecutive),
           until total attacked timesteps == num_attack_{train/test} (includes initial+followup).
        3) Store legal indices in batch: initial_agent in [0..N-1], followup_agents in [0..N-1].
           (No n_agents padding to avoid learner gather/index OOB.)
        """

        self.reset()
        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        attack_num = int(self.args.num_attack_test if test_mode else self.args.num_attack_train)
        remaining = attack_num  # total attacked timesteps remaining (initial+followup)

        attack_started = False
        pre_initial_attack_step = 0

        # initial agent (fixed once started) and legal value to store even before start
        initial_agent_idx = None
        last_initial_agent_valid = 0

        # previous attacked (for masking followup selection)
        prev_attacked_agents = []

        # one-step history for KL
        obs_prev = None
        a_prev_clean_idx = None
        a_prev_fact_idx = None

        # followup size (scheme may be max of two)
        K = int(getattr(self.args, "num_followup_agents", 1))
        max_followup_size = max(
            int(getattr(self.args, "num_followup_agents", 1)),
            int(getattr(self.args, "num_followup_agents", 1)),
        )

        while not terminated:
            # snapshot hidden before updating this step (for policy eval)
            hidden_before = self.mac.return_hidden().detach().clone()

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()],
            }
            self.batch.update(pre_transition_data, ts=self.t)

            ori_actions, chosen_actions, random_actions, attacker_actions = self.mac.select_actions_wolfpack(
                self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode
            )
            clean_idx_list = _to_idx_list(ori_actions, self.args.n_agents)

            do_actions = copy.deepcopy(ori_actions)
            attacked_agents_this_step = []  # list[int], victims that we will force this step

            # ------------------------------------------------------------
            # Decide attack for this timestep
            # ------------------------------------------------------------
            if remaining > 0:
                if not attack_started:
                    # Planner decides initial start
                    # Only when attack_prob == 1 we start the attack and consume 1 budget
                    if getattr(self.args, "init_attack_step_method", "planner") == "planner":
                        attack_prob, chosen_agent = self.get_attack_step(pre_initial_attack_step)
                        if attack_prob == 1:
                            if getattr(self.args, "init_agent_random", False):
                                initial_agent_idx = random.randrange(self.args.n_agents)
                            else:
                                initial_agent_idx = int(chosen_agent) % self.args.n_agents

                            last_initial_agent_valid = int(initial_agent_idx)
                            attack_started = True
                            pre_initial_attack_step = self.t

                            # initial step attacks ONLY initial agent
                            attacked_agents_this_step = [int(initial_agent_idx)]
                            prev_attacked_agents = [int(initial_agent_idx)]
                            remaining -= 1
                    else:
                        # If not using planner, start immediately at first step (still consecutive afterwards)
                        initial_agent_idx = random.randrange(self.args.n_agents)
                        last_initial_agent_valid = int(initial_agent_idx)
                        attack_started = True
                        pre_initial_attack_step = self.t

                        attacked_agents_this_step = [int(initial_agent_idx)]
                        prev_attacked_agents = [int(initial_agent_idx)]
                        remaining -= 1
                else:
                    # Attack already started -> MUST attack every step until remaining==0
                    # Choose followup victims; if KL cannot be computed, fallback to L2/random
                    N = self.args.n_agents
                    all_agents = set(range(N))
                    excluded = set(int(a) for a in prev_attacked_agents) if len(prev_attacked_agents) > 0 else set()
                    agents_set = list(all_agents - excluded)
                    if len(agents_set) == 0:
                        agents_set = list(all_agents)

                    followup = None
                    try:
                        if (
                            obs_prev is not None
                            and a_prev_clean_idx is not None
                            and a_prev_fact_idx is not None
                            and initial_agent_idx is not None
                        ):
                            avail_t = self.batch["avail_actions"][:, self.t]  # [1,N,A]
                            followup = self._select_followup_by_one_step_KL(
                                obs_tm1_np=obs_prev,
                                obs_t_np=pre_transition_data["obs"][0],
                                a_tm1_fact_idx=a_prev_fact_idx,
                                a_tm1_clean_idx=a_prev_clean_idx,
                                initial_agent_idx=int(initial_agent_idx),
                                hidden_before=hidden_before,
                                avail_t=avail_t,
                                prev_attacked=prev_attacked_agents,
                            )
                    except Exception:
                        followup = None

                    if followup is None or len(followup) == 0:
                        # fallback: l2 if enabled & possible, else random
                        if getattr(self.args, "followup_l2", False):
                            try:
                                # l2 helper expects (batch, t, initial_agent:int, agents_set:set)
                                followup = self.followup_agent_group_selection_l2(
                                    self.batch, self.t, int(prev_attacked_agents[0]) if len(prev_attacked_agents) > 0 else 0,
                                    set(agents_set)
                                )
                            except Exception:
                                followup = None

                        if followup is None or len(followup) == 0:
                            # random fallback
                            k = min(K, len(agents_set))
                            followup = random.sample(agents_set, k=k)
                            if len(followup) < K:
                                followup += [followup[-1]] * (K - len(followup))
                            followup = followup[:K]

                    # ensure size K and legal
                    followup = [int(x) % N for x in followup]
                    if len(followup) < K:
                        followup += [followup[-1]] * (K - len(followup))
                    followup = followup[:K]

                    attacked_agents_this_step = followup
                    prev_attacked_agents = followup[:]  # next step mask uses this
                    remaining -= 1

            # ------------------------------------------------------------
            # Apply forced actions for selected victims
            # ------------------------------------------------------------
            for j in attacked_agents_this_step:
                if 0 <= int(j) < self.args.n_agents:
                    do_actions[:, int(j)] = copy.deepcopy(attacker_actions[:, int(j)])

            # Count "actually changed" attacks for logging only (budget is tracked by `remaining`)
            did_change = (not ori_actions.equal(do_actions))

            reward, terminated, env_info = self.env.step(do_actions[0])
            episode_return += reward

            post_transition_data = {
                "actions": ori_actions.to("cpu").numpy(),
                "forced_actions": do_actions.to("cpu").numpy(),
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }
            self.batch.update(post_transition_data, ts=self.t)

            # ------------------------------------------------------------
            # Write to batch (ALWAYS legal ids)
            # ------------------------------------------------------------
            init_agent_store = [int(last_initial_agent_valid) % self.args.n_agents]

            if not attack_started:
                # before attack starts: store zeros (legal) – do not use n_agents padding
                followup_store = [0] * max_followup_size
            else:
                if len(attacked_agents_this_step) == 0:
                    # should not happen after started, but keep safe
                    followup_store = [0] * K
                else:
                    # For initial step attacked_agents_this_step is [initial] (len=1), pad to K
                    tmp = attacked_agents_this_step[:]
                    if len(tmp) < K:
                        tmp += [tmp[-1]] * (K - len(tmp))
                    tmp = tmp[:K]
                    followup_store = [int(x) % self.args.n_agents for x in tmp]

                # pad to scheme size
                if len(followup_store) < max_followup_size:
                    followup_store += [followup_store[-1]] * (max_followup_size - len(followup_store))
                followup_store = followup_store[:max_followup_size]

            self.batch.update(
                {"initial_agent": [init_agent_store], "followup_agents": [followup_store]},
                ts=self.t,
            )

            # update one-step history (used for KL at next step)
            obs_prev = pre_transition_data["obs"][0]
            a_prev_clean_idx = clean_idx_list
            a_prev_fact_idx = _to_idx_list(do_actions, self.args.n_agents)

            self.t += 1

        # tail
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

        cur_stats = self.test_cont_stats if test_mode else self.cont_stats
        cur_returns = self.test_cont_returns if test_mode else self.cont_returns
        log_prefix = "test_Continuous_" if test_mode else "Continuous_"

        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode:
            if len(cur_returns) == self.args.test_nepisode:
                self._log(cur_returns, cur_stats, log_prefix)
        else:
            self._log(cur_returns, cur_stats, log_prefix)

        if getattr(self.args, "use_wandb", False):
            attacked_steps = attack_num - remaining
            wandb.log({f"{log_prefix}attack_steps": attacked_steps}, step=self.t_env)

        return self.batch

    # ============================================================
    # Random Attack (Random timestep, random agents)
    # ============================================================
    def run_randomattack(self, test_mode=False):
        """
        Random attack mode:
        - Total attacked agent-steps = num_attack * num_followup_agents
        - Each timestep has a probability to attack
        - When attacking, randomly select num_followup_agents from all agents
        """
        self.reset()
        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        attack_num = int(self.args.num_attack_test if test_mode else self.args.num_attack_train)
        total_agent_steps = attack_num * self.args.num_followup_agents
        remaining_agent_steps = total_agent_steps
        
        K = int(getattr(self.args, "num_followup_agents", 1))
        max_followup_size = max(
            int(getattr(self.args, "num_followup_agents", 1)),
            int(getattr(self.args, "num_followup_agents", 1)),
        )

        while not terminated and remaining_agent_steps > 0:
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()],
            }
            self.batch.update(pre_transition_data, ts=self.t)

            ori_actions, chosen_actions, random_actions, attacker_actions = self.mac.select_actions_wolfpack(
                self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode
            )

            do_actions = copy.deepcopy(ori_actions)
            attacked_agents_this_step = []

            # Randomly decide if we attack this timestep
            # Calculate probability to uniformly distribute attacks across episode
            attack_prob = remaining_agent_steps / max(1, self.episode_limit - self.t)
            
            if random.random() < attack_prob and remaining_agent_steps > 0:
                # Randomly select num_followup_agents agents to attack
                # We can attack up to min(K, remaining_agent_steps, n_agents) agents this step
                n_agents_to_attack = min(K, remaining_agent_steps, self.args.n_agents)
                attacked_agents_this_step = random.sample(range(self.args.n_agents), k=n_agents_to_attack)
                remaining_agent_steps -= len(attacked_agents_this_step)

            # Apply forced actions for attacked agents
            for j in attacked_agents_this_step:
                if 0 <= int(j) < self.args.n_agents:
                    do_actions[:, int(j)] = copy.deepcopy(attacker_actions[:, int(j)])

            reward, terminated, env_info = self.env.step(do_actions[0])
            episode_return += reward

            post_transition_data = {
                "actions": ori_actions.to("cpu").numpy(),
                "forced_actions": do_actions.to("cpu").numpy(),
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }
            self.batch.update(post_transition_data, ts=self.t)

            # Write to batch (ALWAYS legal ids)
            # For random attack, we use attacked agents as both initial and followup
            if len(attacked_agents_this_step) > 0:
                init_agent_store = [int(attacked_agents_this_step[0]) % self.args.n_agents]
                followup_store = [int(x) % self.args.n_agents for x in attacked_agents_this_step]
            else:
                init_agent_store = [0]
                followup_store = [0]

            # Pad to scheme size
            if len(followup_store) < K:
                followup_store += [followup_store[-1]] * (K - len(followup_store))
            followup_store = followup_store[:K]

            if len(followup_store) < max_followup_size:
                followup_store += [followup_store[-1]] * (max_followup_size - len(followup_store))
            followup_store = followup_store[:max_followup_size]

            self.batch.update(
                {"initial_agent": [init_agent_store], "followup_agents": [followup_store]},
                ts=self.t,
            )

            self.t += 1

        # tail
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

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_Random_" if test_mode else "Random_"

        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode:
            if len(cur_returns) == self.args.test_nepisode:
                self._log(cur_returns, cur_stats, log_prefix)
        else:
            self._log(cur_returns, cur_stats, log_prefix)

        if getattr(self.args, "use_wandb", False):
            attacked_agent_steps = total_agent_steps - remaining_agent_steps
            wandb.log({f"{log_prefix}attack_agent_steps": attacked_agent_steps}, step=self.t_env)

        return self.batch

    # ============================================================
    # Logging
    # ============================================================
    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)

        if getattr(self.args, "use_wandb", False):
            wandb.log({prefix + "return_mean": np.mean(returns)}, step=self.t_env)
            wandb.log({prefix + "return_std": np.std(returns)}, step=self.t_env)

        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean", v / stats["n_episodes"], self.t_env)
                if getattr(self.args, "use_wandb", False):
                    wandb.log({prefix + k + "_mean": v / stats["n_episodes"]}, step=self.t_env)
        stats.clear()
