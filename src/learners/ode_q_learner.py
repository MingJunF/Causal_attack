import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
from torch.optim import RMSprop
import wandb
from torch import optim
import random
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from torch_geometric.data import Data, Batch as GeoBatch

class ODEQLearner:
    def __init__(self, mac, scheme, logger, args, timeseries_ode_model):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())
        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1

        # === ODE 模型与优化器（用于攻击选择 + 自监督拟合 per-agent Q）===
        self.timeseries_ode_model = timeseries_ode_model
        self.optimizer_influence = optim.Adam(self.timeseries_ode_model.parameters(), lr=args.lr)

    # 你原有的工具函数：时间滑窗（抽起点 si，取长度 window_size）
    def sliding_windows(self, data, sis_list, window_size=20):
        X = []
        for i in range(data.shape[0]):
            si = sis_list[i]
            X.append(data[i, si:si + window_size])
        X_tensor = th.stack(X, dim=0)
        return X_tensor

    # 你原有的工具函数：把每个时间步都带上最近 window_size 个历史并做前置 padding
    def sliding_window_with_padding(self, tensor, window_size=20):
        if len(tensor.shape) == 3:
            B, T, F = tensor.size()
            result = th.zeros(B, T, window_size, F, device=tensor.device)
            for t in range(T):
                if t + 1 >= window_size:
                    result[:, t, :, :] = tensor[:, t - window_size + 1:t + 1, :]
                else:
                    result[:, t, window_size - (t+1):, :] = tensor[:, :t + 1, :]
        else:
            B, T, agent, F = tensor.size()
            result = th.zeros(B, T, window_size, agent, F, device=tensor.device)
            for t in range(T):
                if t + 1 >= window_size:
                    result[:, t, :, :, :] = tensor[:, t - window_size + 1:t + 1, :, :]
                else:
                    result[:, t, window_size - (t+1):, :, :] = tensor[:, :t + 1, :, :]
        return result

    def _construct_initial_edge_weights_from_q(self, indi_q, n_agents):
        """
        基于 per-agent Q 值变化构造初始边权重
        indi_q: [B, T, N] - 历史 Q 值
        返回: [B, N, N] - 初始边权重矩阵
        """
        B, T, N = indi_q.shape
        device = indi_q.device
        
        # 计算每个 agent 的 Q 值变化（相邻时间步的差分）
        if T > 1:
            q_diff = indi_q[:, 1:, :] - indi_q[:, :-1, :]  # [B, T-1, N]
            q_change = q_diff.abs().mean(dim=1)  # [B, N] - 平均变化量
        else:
            q_change = indi_q[:, 0, :].abs()  # [B, N]
        
        # 构造边权重：agent i 对 agent j 的影响 = j 的 Q 变化量
        edge_weights = th.zeros(B, N, N, device=device)
        for b in range(B):
            for i in range(N):
                for j in range(N):
                    if i != j:
                        # i->j 的边权 = j 的 Q 变化（表示 i 可能影响 j）
                        edge_weights[b, i, j] = q_change[b, j]
        
        # 归一化到 [0, 1]
        for b in range(B):
            max_val = edge_weights[b].max()
            if max_val > 0:
                edge_weights[b] = edge_weights[b] / max_val
            # 添加自环（小权重）
            edge_weights[b] += th.eye(N, device=device) * 0.1
        
        return edge_weights  # [B, N, N]

    def _construct_graph_batch(self, node_feat, treatment, batch_size, window_size, n_agents, initial_edge_weights=None):
        """
        构建符合 ODE 模型要求的图批次数据（参考 load_data_covid.py:transfer_one_graph）
        node_feat: [B, W, N, D_node]
        treatment: [B, W, N, t_dim]
        initial_edge_weights: [B, N, N] - 基于 Q 值的初始边权（可选）
        返回: encoder_batch (Batch), decoder_batch (dict)
        """
        device = node_feat.device
        B, W, N, D = node_feat.shape
        
        # ========== 构建 Encoder Batch（关键修复） ==========
        graph_list = []
        for b in range(B):
            # 1. 展平所有时间步的节点特征：[W*N, D]
            x = node_feat[b].reshape(W * N, D)
            
            # 2. 时间位置编码：每个节点对应的归一化时间 [0, 1]
            pos = th.cat([th.full((N,), t / W, device=device) for t in range(W)])  # [W*N]
            
            # 3. 构建边索引和边权重（每个时间步内的全连接 + 跨时间的连接）
            edge_list = []
            edge_weights_list = []
            edge_times_list = []
            
            # 3.1 使用基于 Q 值的初始边权（如果提供）
            if initial_edge_weights is not None:
                base_edge_weight = initial_edge_weights[b]  # [N, N]
            else:
                base_edge_weight = th.ones(N, N, device=device) / N  # 均匀分布
            
            # 3.2 为每个时间步构建图
            for t in range(W):
                offset = t * N
                for i in range(N):
                    for j in range(N):
                        # 当前时间步内的边
                        edge_list.append([offset + i, offset + j])
                        edge_weights_list.append(base_edge_weight[i, j].item())
                        edge_times_list.append(0.0)  # 同一时刻，相对时间为 0
                        
                        # 添加跨时间的边（i_t -> j_{t-1}，如果 t > 0）
                        if t > 0:
                            prev_offset = (t - 1) * N
                            edge_list.append([offset + i, prev_offset + j])
                            edge_weights_list.append(base_edge_weight[i, j].item() * 0.5)  # 跨时间边权重减半
                            edge_times_list.append(-1.0 / W)  # 负的相对时间
            
            # 4. 转换为张量
            edge_index = th.tensor(edge_list, dtype=th.long).t().to(device)  # [2, num_edges]
            edge_weight = th.tensor(edge_weights_list, dtype=th.float, device=device)
            edge_time = th.tensor(edge_times_list, dtype=th.float, device=device)
            
            # 5. y：每个 agent 有 W 个时间步
            y = th.full((N,), W, dtype=th.long, device=device)
            
            # 6. 创建 PyG Data 对象
            graph_data = Data(
                x=x,                    # [W*N, D]
                edge_index=edge_index,  # [2, num_edges]
                edge_weight=edge_weight, # [num_edges]
                edge_time=edge_time,    # [num_edges]
                pos=pos,                # [W*N]
                y=y                     # [N]
            )
            graph_list.append(graph_data)
        
        # 7. 合并成 Batch
        encoder_batch = GeoBatch.from_data_list(graph_list)
        
        # ========== 构建 Decoder Batch（保持不变） ==========
        node_feat_reshaped = node_feat.permute(0, 2, 1, 3).reshape(B * N, W, -1)
        treatment_reshaped = treatment.permute(0, 2, 1, 3).reshape(B * N, W, -1)
        time_steps = th.linspace(0, 1, W, device=device)
        time_absolute = th.arange(W, device=device, dtype=th.float).unsqueeze(0).repeat(B * N, 1)
        
        decoder_batch = {
            "data": node_feat_reshaped,
            "time_steps": time_steps,
            "treatment": treatment_reshaped,
            "interference": th.zeros_like(treatment_reshaped),
            "time_absolute": time_absolute
        }
        
        return encoder_batch, decoder_batch

    def _extract_edge_weights_from_ode(self, pred_edge, batch_size, window_size, n_agents):
        """
        从 ODE 输出中提取边权重
        pred_edge 可能的形状: [B*N*N, W, 1] 或其他
        返回: [B, W, N, N]
        """
        # 先移除最后一维（如果是 [B*N*N, W, 1]）
        if pred_edge.dim() == 3 and pred_edge.size(-1) == 1:
            pred_edge = pred_edge.squeeze(-1)  # [B*N*N, W]
        
        # 现在应该是 [B*N*N, W]
        # 重塑为 [B, N*N, W]
        try:
            pred_edge = pred_edge.reshape(batch_size, n_agents * n_agents, window_size)
        except RuntimeError as e:
            # 如果形状不匹配，尝试其他可能的维度
            print(f"Warning: pred_edge shape {pred_edge.shape} doesn't match expected [B*N*N, W]")
            # 假设是 [B*W, N*N] 或其他变体
            if pred_edge.numel() == batch_size * n_agents * n_agents * window_size:
                pred_edge = pred_edge.reshape(batch_size, n_agents * n_agents, window_size)
            else:
                raise e
        
        # 重塑为 [B, N, N, W]
        pred_edge = pred_edge.reshape(batch_size, n_agents, n_agents, window_size)
        
        # 转置为 [B, W, N, N]
        edge_weights = pred_edge.permute(0, 3, 1, 2)
        
        return edge_weights

    def _select_attack_targets(self, edge_weights, initial_agent_ids, mask, future_steps=5):
        """
        基于边权重选择攻击目标
        edge_weights: [B, W, N, N]
        initial_agent_ids: [B] - 当前初始攻击者
        mask: [B, W]
        future_steps: 未来观察的步数
        返回: selected_agents [B, W, 2] - 每个时刻选择的两个攻击目标
        """
        B, W, N, _ = edge_weights.shape
        device = edge_weights.device
        
        selected_agents = th.zeros(B, W, 2, dtype=th.long, device=device)
        
        for b in range(B):
            init_agent = initial_agent_ids[b].item()
            
            # 选择初始攻击目标（未来 future_steps 内边权最大）
            if W >= future_steps:
                future_edges = edge_weights[b, :future_steps, init_agent, :]  # [future_steps, N]
                future_edge_sum = future_edges.sum(dim=0)  # [N]
                future_edge_sum[init_agent] = -float('inf')  # 排除自己
                initial_target = future_edge_sum.argmax()
            else:
                # 如果窗口不够，用全部
                future_edge_sum = edge_weights[b, :, init_agent, :].sum(dim=0)
                future_edge_sum[init_agent] = -float('inf')
                initial_target = future_edge_sum.argmax()
            
            # 每个时刻选择边权最大的两个 agent
            for t in range(W):
                if mask[b, t] > 0:
                    edges_t = edge_weights[b, t, init_agent, :]  # [N]
                    edges_t_sorted = edges_t.argsort(descending=True)
                    
                    # 选择前两个（排除自己）
                    candidates = []
                    for agent_id in edges_t_sorted:
                        if agent_id != init_agent and len(candidates) < 2:
                            candidates.append(agent_id)
                    
                    if len(candidates) >= 2:
                        selected_agents[b, t, 0] = candidates[0]
                        selected_agents[b, t, 1] = candidates[1]
                    elif len(candidates) == 1:
                        selected_agents[b, t, 0] = candidates[0]
                        selected_agents[b, t, 1] = candidates[0] # 重复
        
        return selected_agents

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        ######################################################
        # 1) TD 学习
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # [B, T+1, N, A]

        mac_out_temp = mac_out.clone().detach()

        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # [B, T, N]
        indi_q = chosen_action_qvals.clone().detach()  # per-agent Q（监督给 ODE 用）

        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # [B, T, N, A]

        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        if self.args.double_q:
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])     # [B, T, 1]
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])     # [B, T, 1]

        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals
        td_error = (chosen_action_qvals - targets.detach())
        mask_exp = mask.expand_as(td_error)
        masked_td_error = td_error * mask_exp
        loss = (masked_td_error ** 2).sum() / mask_exp.sum()

        ######################################################
        # 2) ODE 未来预测训练
        bs = batch["state"].shape[0]
        obs = batch["obs"][:, :-1]
        actions_onehot = batch["actions_onehot"][:, :-1]

        # 采样滑窗
        W = self.args.trans_input_len
        future_steps = self.args.attack_duration
        max_si = max(0, indi_q.shape[1] - W - future_steps)
        
        sis_list = []
        for _ in range(bs):
            si = random.randint(0, max_si) if max_si > 0 else 0
            sis_list.append(th.tensor(si, device=indi_q.device))

        obs_slided = self.sliding_windows(obs, sis_list, window_size=W)
        act1h_slided = self.sliding_windows(actions_onehot, sis_list, window_size=W)
        indi_q_slided = self.sliding_windows(indi_q, sis_list, window_size=W)

        # 采样未来真实 Q
        indi_q_future_list = []
        for b in range(bs):
            si = sis_list[b].item()
            if si + W + future_steps <= indi_q.shape[1]:
                indi_q_future_list.append(indi_q[b, si + W:si + W + future_steps])
            else:
                available = indi_q.shape[1] - (si + W)
                if available > 0:
                    pad = th.zeros(future_steps - available, self.args.n_agents, device=indi_q.device)
                    indi_q_future_list.append(th.cat([indi_q[b, si + W:], pad], dim=0))
                else:
                    indi_q_future_list.append(th.zeros(future_steps, self.args.n_agents, device=indi_q.device))
        
        indi_q_future_gt = th.stack(indi_q_future_list, dim=0)

        # 节点特征
        node_feat_slided = th.cat([obs_slided, act1h_slided], dim=-1)
        t_dim = 1
        treatment_zero = th.zeros(bs, W, self.args.n_agents, t_dim, device=indi_q.device)
        
        # 构造初始边权
        initial_edge_weights = self._construct_initial_edge_weights_from_q(indi_q_slided, self.args.n_agents)
        
        # 构建图批次
        encoder_batch, decoder_batch = self._construct_graph_batch(
            node_feat_slided, treatment_zero, bs, W, self.args.n_agents,
            initial_edge_weights=initial_edge_weights
        )
        
        # Encoder
        z0_mu, z0_std = self.timeseries_ode_model.encoder_z0(
            encoder_batch.x,
            encoder_batch.edge_weight,
            encoder_batch.edge_index,
            encoder_batch.pos,
            encoder_batch.edge_time,
            encoder_batch.batch,
            encoder_batch.y
        )
        z0 = z0_mu
        K_N = bs * self.args.n_agents
        
        # 关键修复6：treatment 形状 [N, T, K]
        max_T = W + future_steps
        treatment_dummy = th.zeros(self.args.n_agents, max_T, t_dim, device=z0.device)  # ✅ [N, T, K]
        self.timeseries_ode_model.set_treatments(treatment_dummy)
        
        # Policy: dtype=long，结束时间=T-1
        policy_starting_ending_points = th.zeros(self.args.n_agents, t_dim, 2, device=z0.device, dtype=th.long)
        policy_starting_ending_points[:, :, 0] = 0
        policy_starting_ending_points[:, :, 1] = max_T - 1
        self.timeseries_ode_model.set_policy_starting_ending_points(policy_starting_ending_points)
        
        # 自回归 rollout
        pred_q_future_list = []
        current_state = z0
        
        for step in range(future_steps):
            t_start = step / future_steps
            t_end = (step + 1) / future_steps
            time_next = th.tensor([t_start, t_end], device=z0.device, dtype=th.float)
            
            time_absolute_step = th.arange(max_T, device=z0.device, dtype=th.long).unsqueeze(0).repeat(K_N, 1)
            
            sol_y, K_N_output, _ = self.timeseries_ode_model.diffeq_solver(
                first_point=current_state,
                time_steps_to_predict=time_next,
                times_absolute=time_absolute_step,
                w_node_to_edge_initial=self.timeseries_ode_model.w_node_to_edge_initial
            )
            
            node_states = sol_y[:K_N, -1, :]
            pred_q_step = self.timeseries_ode_model.decoder_node(node_states)
            pred_q_future_list.append(pred_q_step.squeeze(-1).reshape(bs, self.args.n_agents))
            current_state = node_states
        
        pred_q_future = th.stack(pred_q_future_list, dim=1)
        loss_ode = F.mse_loss(pred_q_future, indi_q_future_gt)
        
        # ✅ 新增：计算额外的监控指标
        with th.no_grad():
            pred_mean = pred_q_future.mean().item()
            pred_std = pred_q_future.std().item()
            gt_mean = indi_q_future_gt.mean().item()
            gt_std = indi_q_future_gt.std().item()
            mae = (pred_q_future - indi_q_future_gt).abs().mean().item()
        
        # 训练 ODE
        self.optimizer_influence.zero_grad()
        loss_ode.backward()
        grad_norm_ode = th.nn.utils.clip_grad_norm_(self.timeseries_ode_model.parameters(), self.args.grad_norm_clip)
        self.optimizer_influence.step()

        ######################################################
        # 3) 优化 TD
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        ######################################################
        # 4) target 网络 & 日志
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            mask_elems = mask_exp.sum().item()
            
            # ✅ 原有日志（保持不变）
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask_exp).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask_exp).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("loss_ode_future", loss_ode.item(), t_env)

            # ✅ 新增：ODE 详细监控
            self.logger.log_stat("ode/loss_mse", loss_ode.item(), t_env)
            self.logger.log_stat("ode/loss_mae", mae, t_env)
            self.logger.log_stat("ode/grad_norm", grad_norm_ode, t_env)
            self.logger.log_stat("ode/pred_q_mean", pred_mean, t_env)
            self.logger.log_stat("ode/pred_q_std", pred_std, t_env)
            self.logger.log_stat("ode/gt_q_mean", gt_mean, t_env)
            self.logger.log_stat("ode/gt_q_std", gt_std, t_env)

            # ✅ wandb 日志（原有 + 新增）
            wandb.log({
                # 原有 TD 日志
                "loss_td": loss.item(),
                "grad_norm": grad_norm,
                "td_error_abs": (masked_td_error.abs().sum().item()/mask_elems),
                "q_taken_mean": (chosen_action_qvals * mask_exp).sum().item()/(mask_elems * self.args.n_agents),
                "target_mean": (targets * mask_exp).sum().item()/(mask_elems * self.args.n_agents),
                
                # ✅ 新增 ODE 监控
                "ode/loss_mse": loss_ode.item(),
                "ode/loss_mae": mae,
                "ode/grad_norm": grad_norm_ode,
                "ode/pred_q_mean": pred_mean,
                "ode/pred_q_std": pred_std,
                "ode/gt_q_mean": gt_mean,
                "ode/gt_q_std": gt_std,
                "ode/pred_gt_diff": abs(pred_mean - gt_mean),  # 预测与真实的均值差距
            }, step=t_env)

            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
        # ODE
        self.timeseries_ode_model.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))
        # 你已在外层加了保存 ODE 的逻辑，这里保留
        th.save(self.timeseries_ode_model.state_dict(), "{}/timeseries_ode_model.th".format(path))
        th.save(self.optimizer_influence.state_dict(), "{}/optimizer_influence.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
        # 如需一并恢复 ODE，可解开下面两行
        # self.timeseries_ode_model.load_state_dict(th.load("{}/timeseries_ode_model.th".format(path), map_location=lambda storage, loc: storage))
        # self.optimizer_influence.load_state_dict(th.load("{}/optimizer_influence.th".format(path), map_location=lambda storage, loc: storage))
