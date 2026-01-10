import torch
import torch.nn as nn
# ✅ 使用普通 odeint，避免 ReverseFunc 嵌套导致的 RecursionError
from torchdiffeq import odeint
import numpy as np
import lib.utils as utils
import torch.nn.functional as F
from scipy.linalg import block_diag
from torch_scatter import scatter_add
import scipy.sparse as sp
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.inits import glorot


class TemporalEncoding(nn.Module):
    def __init__(self, d_hid):
        super(TemporalEncoding, self).__init__()
        self.d_hid = d_hid
        div = torch.FloatTensor([1 / np.power(10000, 2 * (hid_j // 2) / self.d_hid) for hid_j in range(self.d_hid)])  # [D]
        div = div.view(1, -1)
        self.div_term = nn.Parameter(div, requires_grad=False)

    def forward(self, t):
        """
        t: [n, 1] or [n]
        return: [n, D]
        """
        t = t.view(-1, 1)
        position_term = torch.matmul(t, self.div_term)  # [n, D]
        position_term[:, 0::2] = torch.sin(position_term[:, 0::2])
        position_term[:, 1::2] = torch.cos(position_term[:, 1::2])
        return position_term


def compute_edge_initials(first_point_enc, num_atoms, w_node_to_edge_initial):
    """
    first_point_enc: [K*N, D]
    return: [K*N*N, D_edge]
    """
    node_feature_num = first_point_enc.shape[1]

    fully_connected = np.ones([num_atoms, num_atoms])
    rel_send = np.array(utils.encode_onehot(np.where(fully_connected)[0]), dtype=np.float32)  # (N*N, N)
    rel_rec  = np.array(utils.encode_onehot(np.where(fully_connected)[1]), dtype=np.float32)  # (N*N, N)

    rel_send = torch.FloatTensor(rel_send).to(first_point_enc.device)
    rel_rec  = torch.FloatTensor(rel_rec ).to(first_point_enc.device)

    first_point_enc = first_point_enc.view(-1, num_atoms, node_feature_num)  # [K, N, D]

    senders   = torch.matmul(rel_send, first_point_enc)   # [K, N*N, D]
    receivers = torch.matmul(rel_rec,  first_point_enc)   # [K, N*N, D]

    edge_initials = torch.cat([senders, receivers], dim=-1)          # [K, N*N, 2D]
    edge_initials = F.gelu(w_node_to_edge_initial(edge_initials))    # [K, N*N, D_edge]
    edge_initials = edge_initials.view(-1, edge_initials.shape[2])   # [K*N*N, D_edge]

    return edge_initials


class DiffeqSolver(nn.Module):
    def __init__(self, ode_func, method, args,
                 odeint_rtol=1e-3, odeint_atol=1e-4, device=torch.device("cpu")):
        super(DiffeqSolver, self).__init__()

        self.ode_method = method
        self.device = device
        self.ode_func = ode_func
        self.args = args
        self.num_atoms = args.num_atoms

        self.odeint_rtol = odeint_rtol
        self.odeint_atol = odeint_atol

    def forward(self, first_point, time_steps_to_predict, times_absolute, w_node_to_edge_initial):
        """
        first_point: [K*N, d]
        time_steps_to_predict: [T] （torchdiffeq 的时间网格，实值也可）
        times_absolute: [K*N, T] 绝对时间索引矩阵（或离散时间步索引），用于 treatment 对齐
        """
        # Node ODE Function
        n_traj, feature_node = first_point.size(0), first_point.size(1)  # [K*N, d]

        # Augment if needed
        if self.args.augment_dim > 0:
            aug_node = torch.zeros(first_point.shape[0], self.args.augment_dim, device=self.device)  # [K*N, D_aug]
            first_point = torch.cat([first_point, aug_node], dim=1)  # [K*N, d + D_aug]
            feature_node += self.args.augment_dim

        # Edge initialization: h_ij = f([u_i, u_j])
        edge_initials = compute_edge_initials(first_point, self.num_atoms, w_node_to_edge_initial)  # [K*N*N, D_edge]
        assert not torch.isnan(edge_initials).any(), "edge_initials contains NaN."

        node_edge_initial = torch.cat([first_point, edge_initials], dim=0)  # [K*N + K*N*N, *]
        K_N = int(node_edge_initial.shape[0] // (self.num_atoms + 1))  # = K*N
        K = K_N // self.num_atoms

        # 配置 ODEFunc
        self.ode_func.set_index_and_graph(K_N, K, time_steps_to_predict)
        self.ode_func.set_t_treatments(times_absolute)

        node_initial = node_edge_initial[:K_N, :]
        self.ode_func.set_initial_z0(node_initial)

        # 求解
        self.ode_func.nfe = 0
        self.ode_func.t_index = 0

        pred_y = odeint(self.ode_func, node_edge_initial, time_steps_to_predict,
                        rtol=self.odeint_rtol, atol=self.odeint_atol, method=self.ode_method)  # [T, K*N+K*N*N, d']
        pred_y = pred_y.permute(1, 0, 2)  # [K*N+K*N*N, T, d']

        # treatment 展开：从已设置的 t_treatments 推断维度
        treatment_dim = self.ode_func.t_treatments.shape[-1]
        T = time_steps_to_predict.shape[0]
        treatment_rep = torch.zeros(T, K_N, treatment_dim, device=self.device)

        if int(self.args.use_attention) == 1:
            for t in range(T):
                treatment_rep[t, :, :] = self.ode_func.calculate_merged_treatment(
                    self.ode_func.t_treatments[:, t, :], t, self.ode_func.w_treatment)
        else:
            for t in range(T):
                treatment_rep[t, :, :] = self.ode_func.t_treatments[:, t, :]

        treatment_rep = treatment_rep.permute(1, 0, 2)  # [K*N, T, treatment_dim]

        assert pred_y.size(0) == K_N * (self.num_atoms + 1), "pred_y size mismatch."

        if self.args.augment_dim > 0:
            pred_y = pred_y[:, :, :-self.args.augment_dim]

        return pred_y, K_N, treatment_rep

    def set_treatments(self, treatments):
        self.ode_func.set_treatments(treatments)

    def set_policy_starting_ending_points(self, policy_starting_ending_points):
        self.ode_func.set_policy_starting_ending_points(policy_starting_ending_points)


class CoupledODEFunc(nn.Module):
    def __init__(self, args, num_treatment, node_ode_func_net, edge_ode_func_net, num_atom, dropout,
                 device=torch.device("cpu")):
        """
        node_ode_func_net: 模型，输入 (cat_node_attributes, edge_value, node_z0)，输出 grad_node
        edge_ode_func_net: 模型，输入 (cat_node_attributes, edge_attributes, num_atom)，输出 (grad_edge, edge_value)
        """
        super(CoupledODEFunc, self).__init__()

        self.args = args
        self.num_treatment = num_treatment
        self.device = device
        self.node_ode_func_net = node_ode_func_net
        self.edge_ode_func_net = edge_ode_func_net
        self.num_atom = num_atom
        self.nfe = 0
        self.t_index = 0
        self.dropout = nn.Dropout(dropout)

        self.w_treatment = nn.Parameter(
            torch.FloatTensor(self.num_treatment, self.args.treatment_dim)
        )
        glorot(self.w_treatment)

        self.w_treatment_attention = nn.Linear(self.args.treatment_dim, self.args.treatment_dim)
        utils.init_network_weights(self.w_treatment_attention)

        self.temporal_net = TemporalEncoding(self.args.treatment_dim)

    # ---------- Helpers ----------

    def get_time_index(self, t_local):
        """
        给定 torchdiffeq 当前 t_local（标量张量），在 self.time_steps_to_predict 中找到 <= t_local 的最后一个索引；
        若没有（t_local 比最小值还小），返回 0。
        """
        # self.time_steps_to_predict: [T]
        # 构造 mask
        mask = (self.time_steps_to_predict <= t_local + 1e-12)
        idx = torch.nonzero(mask, as_tuple=False)
        if idx.numel() == 0:
            return 0
        return int(idx[-1].item())

    def set_index_and_graph(self, K_N, K, time_steps_to_predict):
        """
        K_N: K*N
        K: 批内样本数
        """
        self.K_N = K_N
        self.K = int(K)
        self.K_N_N = self.K_N * self.num_atom
        self.nfe = 0
        self.time_steps_to_predict = time_steps_to_predict

        # 构造块对角的全连接图（每个样本自身 N 节点）
        edge_each = np.ones((self.num_atom, self.num_atom))
        edge_whole = block_diag(*([edge_each] * self.K))  # [K*N, K*N]
        edge_index, _ = utils.convert_sparse(edge_whole)  # (2, E)
        self.edge_index = torch.LongTensor(edge_index).to(self.device)  # [2, E]

    def set_initial_z0(self, node_z0):
        self.node_z0 = node_z0

    def normalize_graph(self, edge_weight, num_nodes):
        """
        非对称图的按行归一化。
        edge_weight: [K, N*N]，对应 self.edge_index 的有序展开（与 convert_sparse 约定一致）
        num_nodes: K*N
        return: [K, N*N]
        """
        assert not torch.isnan(edge_weight).any(), "edge_weight contains NaN."
        assert torch.all(edge_weight >= 0), "edge_weight has negatives."

        edge_weight_flatten = edge_weight.reshape(-1)  # [K*N*N]
        row, col = self.edge_index[0], self.edge_index[1]  # [E], [E]

        deg = scatter_add(edge_weight_flatten, row, dim=0, dim_size=num_nodes)  # [K*N]
        deg_inv = deg.pow(-1)
        deg_inv.masked_fill_(torch.isinf(deg_inv), 0.0)

        if torch.isnan(deg_inv).any():
            # 额外检查（一般不会走到）
            assert torch.sum(deg == 0) == 0, "row degree zero exists."
            assert torch.sum(deg < 0) == 0, "row degree negative exists."

        edge_weight_normalized = deg_inv[row] * edge_weight_flatten  # [E]
        assert torch.all(edge_weight_normalized >= 0) and torch.all(edge_weight_normalized <= 1), \
            "normalized weight out of [0,1]."

        edge_weight_normalized = edge_weight_normalized.view(self.K, -1)  # [K, N*N]
        assert not torch.isnan(edge_weight_normalized).any(), "normalized weights NaN."
        return edge_weight_normalized

    def set_t_treatments(self, times_absolute):
        """
        根据绝对时间索引矩阵构造每个 (k,n) 的 treatment 时间序列。
        times_absolute: [K*N, T] 绝对时间索引（或离散 step 索引）
        依赖 self.treatments: [N, T_max, t_dim]
        """
        device = times_absolute.device
        K_N_input, T2 = times_absolute.shape
        t_dim = self.treatments.shape[2]

        # 与 set_index 保持一致
        if K_N_input != self.K_N:
            # 容错：使用 self.K_N 的前缀
            K_N = self.K_N
            times_absolute = times_absolute[:K_N] if K_N_input >= self.K_N else \
                torch.cat([times_absolute,
                           torch.zeros(self.K_N - K_N_input, T2, dtype=times_absolute.dtype, device=device)], dim=0)
        else:
            K_N = K_N_input

        # clamp 到合法范围
        T_max = self.treatments.shape[1]
        times_idx = times_absolute.long().clamp_(min=0, max=T_max - 1)  # [K*N, T]

        t_treatments = torch.zeros((K_N, T2, t_dim), device=device)
        for idx in range(K_N):
            agent_id = idx % self.num_atom  # 映射到 N
            t_treatments[idx, :, :] = self.treatments[agent_id, times_idx[idx], :]

        self.t_treatments = t_treatments  # [K*N, T, t_dim]
        self.time_absolute = times_idx    # [K*N, T]，保存已 clamp 的索引

    def set_treatments(self, treatments):
        """
        treatments: [N, T_max, t_dim]
        """
        self.treatments = treatments

    def set_policy_starting_ending_points(self, policy_starting_ending_points):
        """
        policy_starting_ending_points: [t_dim, 2], 各 treatment 的 [start, end]（按节点维展开取 index）
        """
        self.policy_starting_ending_points = policy_starting_ending_points

    def weather_multiple_treatment(self, t_all_index):
        """
        t_all_index: nonzero(t_treatment) 的索引对
        返回每条记录是否在一个含多个 treatment 的组中（同一个 kn 上 >1 条）
        """
        _, inverse_indices, counts = torch.unique(t_all_index[:, 0], return_inverse=True, return_counts=True)
        counts = counts - 1
        counts_indexes_nonzero = torch.nonzero(counts, as_tuple=False)
        returned_bool = torch.where(torch.isin(inverse_indices, counts_indexes_nonzero.view(-1)), 1, 0).to(t_all_index.device)
        return returned_bool

    def one_hot_encode(self, input_matrix, num_classes):
        """
        input_matrix: [k, 1] (类别索引)
        return: [k, K]
        """
        one_hot = torch.zeros((input_matrix.shape[0], num_classes), device=input_matrix.device)
        one_hot[torch.arange(input_matrix.shape[0], device=input_matrix.device), input_matrix.flatten()] = 1.0
        return one_hot

    def calculate_merged_treatment(self, t_treatment, t_current, w_treatment):
        """
        将时刻 t_current 的 treatment（多条/多类）按 attention 合并到每个 kn 上。
        t_treatment: [K*N, num_treatment] 或 [K*N, 1, num_treatment]
        return: [K*N, d]  (d = treatment_dim 或 num_treatment 取决于 use_onehot)
        """
        # 统一维度
        if t_treatment.ndim == 3:
            t_treatment = t_treatment.squeeze(1)

        # 全零：直接返回零向量
        if torch.sum(t_treatment != 0) == 0:
            d = self.args.treatment_dim if int(self.args.use_onehot) == 0 else int(self.num_treatment)
            return torch.zeros((t_treatment.shape[0], d), dtype=t_treatment.dtype, device=t_treatment.device)

        # 非零索引（kn, treatment_id）
        t_all_index = torch.nonzero(t_treatment, as_tuple=False).to(t_treatment.device)

        if int(self.args.use_onehot) == 1:
            t_treatment_embeddings_original = self.one_hot_encode(t_all_index[:, 1].view(-1, 1), self.num_treatment)

        # lookup learned embedding
        t_treatment_embeddings = torch.index_select(w_treatment, 0, t_all_index[:, 1])  # [m, d]
        # 计算每条的起始时刻差值
        num_k_list = t_all_index[:, 0] // self.num_atom
        num_n_list = (t_all_index[:, 0] - num_k_list * self.num_atom)
        t_start_treatment_list = torch.index_select(self.policy_starting_ending_points, 0, num_n_list)[:, :, 0]
        t_start_treatment_list = t_start_treatment_list[
            torch.arange(t_treatment_embeddings.shape[0], device=t_treatment.device), t_all_index[:, 1].squeeze()
        ]

        delta_t_treament = self.time_absolute[t_all_index[:, 0], t_current] - t_start_treatment_list

        # 是否只在多 treatment 组中加入时间编码
        if int(self.args.mask_single_treatment) == 1:
            treatment_group_bool = self.weather_multiple_treatment(t_all_index)
        else:
            treatment_group_bool = torch.ones_like(t_all_index[:, 0], device=t_all_index.device)

        # 时间编码 + 组掩码
        t_treatment_embeddings = t_treatment_embeddings + \
            self.temporal_net(delta_t_treament.to(torch.float).to(t_treatment.device)) * treatment_group_bool.view(-1, 1)

        # 组 attention
        attention_vector_group = F.gelu(self.w_treatment_attention(
            global_mean_pool(t_treatment_embeddings, t_all_index[:, 0])
        ))
        attention_vector_group_expanded = torch.index_select(attention_vector_group, 0, t_all_index[:, 0])

        attention_scores = torch.sigmoid(
            torch.squeeze(torch.bmm(attention_vector_group_expanded.unsqueeze(1),
                                    t_treatment_embeddings.unsqueeze(2)), dim=2)
        ).view(-1, 1)

        if int(self.args.use_onehot) == 0:
            weighted = torch.where(
                treatment_group_bool.view(-1, 1) == 1, attention_scores, torch.ones_like(attention_scores)
            ) * t_treatment_embeddings
        else:
            weighted = torch.where(
                treatment_group_bool.view(-1, 1) == 1, attention_scores, torch.ones_like(attention_scores)
            ) * t_treatment_embeddings_original

        # 聚合回 kn
        merged = global_mean_pool(weighted, t_all_index[:, 0])  # [unique_kn, d]
        t_treatment_back = torch.zeros((t_treatment.shape[0], merged.shape[1]),
                                       dtype=t_treatment.dtype, device=t_treatment.device)
        kn_indexes = torch.unique(t_all_index[:, 0])
        t_treatment_back[kn_indexes] = torch.index_select(merged, 0, kn_indexes)
        return t_treatment_back

    # ---------- ODE forward ----------

    def forward(self, t_local, z, backwards=False):
        """
        z: [K*N + K*N*N, d_z]
        返回 dz/dt
        """
        self.nfe += 1
        t_index = self.get_time_index(t_local)

        node_attributes = z[:self.K_N, :]
        edge_attributes = z[self.K_N:, :]

        assert not torch.isnan(node_attributes).any(), "node_attributes NaN."
        assert not torch.isnan(edge_attributes).any(), "edge_attributes NaN."

        # 计算当前时刻的 treatment 表征，并拼到节点特征后
        if int(self.args.use_attention) == 1:
            treatment_cur = self.calculate_merged_treatment(
                self.t_treatments[:, t_index, :], t_index, self.w_treatment
            )  # [K*N, d_t]
            cat_node_attributes = torch.cat((node_attributes, treatment_cur), dim=1)
        else:
            treatment_t = self.t_treatments[:, t_index, :]  # [K*N, d_t]
            cat_node_attributes = torch.cat((node_attributes, treatment_t), dim=1)

        # 计算边与节点的梯度
        grad_edge, edge_value = self.edge_ode_func_net(cat_node_attributes, edge_attributes, self.num_atom)
        edge_value = self.normalize_graph(edge_value, self.K_N)
        assert not torch.isnan(edge_value).any(), "edge_value NaN after normalize."

        grad_node = self.node_ode_func_net(cat_node_attributes, edge_value, self.node_z0)
        assert not torch.isnan(grad_node).any(), "grad_node NaN."
        assert not torch.isinf(grad_edge).any(), "grad_edge Inf."

        grad = torch.cat([grad_node, grad_edge], dim=0)
        grad = self.dropout(grad)
        return grad

    def set_time_base(self, input_t_base):
        self.input_t_base = input_t_base.long()
