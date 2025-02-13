
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import warnings
from itertools import accumulate

class EnhancedLSTM(torch.nn.Module):
    """
    A wrapper for different recurrent dropout implementations, which
    pytorch currently doesn't support nativly.

    Uses multilayer, bidirectional lstms with dropout between layers
    and time steps in a variational manner.

    "allen" reimplements a lstm with hidden to hidden dropout, thus disabling
    CUDNN. Can only be used in bidirectional mode.
    `A Theoretically Grounded Application of Dropout in Recurrent Neural Networks
        <https://arxiv.org/abs/1512.05287>`

    "drop_connect" uses default implemetation, but monkey patches the hidden to hidden
    weight matrices instead.
    `Regularizing and Optimizing LSTM Language Models
        <https://arxiv.org/abs/1708.02182>`

    "native" ignores dropout and uses the default implementation.
    """

    def __init__(self,
                 lstm_type,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 ff_dropout: float = 0.0,
                 recurrent_dropout: float = 0.0,
                 bidirectional=True) -> None:
        super().__init__()

        self.lstm_type = lstm_type

        if lstm_type == "drop_connect":
            self.provider = WeightDropLSTM(
                input_size,
                hidden_size,
                num_layers,
                ff_dropout,
                recurrent_dropout,
                bidirectional=bidirectional)
        elif lstm_type == "native":
            self.provider = torch.nn.LSTM(
                input_size,
                hidden_size,
                num_layers=num_layers,
                dropout=0,
                bidirectional=bidirectional,
                batch_first=True)
        else:
            raise Exception(lstm_type + " is an invalid lstm type")

    # Expects unpacked inputs in format (batch, seq, features)
    def forward(self, inputs, hidden, lengths):
        seq_len = inputs.shape[1]
        if self.lstm_type in ["allen", "native"]:
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                inputs, lengths, batch_first=True)

            output, _ = self.provider(packed, hidden)

            output, _ = torch.nn.utils.rnn.pad_packed_sequence(
                output, batch_first=True)

            return output
        elif self.lstm_type == "drop_connect":
            return self.provider(inputs, lengths, seq_len)


class WeightDropLSTM(torch.nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 ff_dropout: float = 0.0,
                 recurrent_dropout: float = 0.0,
                 bidirectional=True) -> None:
        super().__init__()

        self.locked_dropout = LockedDropout()
        self.lstms = [
            torch.nn.LSTM(
                input_size
                if l == 0 else hidden_size * (1 + int(bidirectional)),
                hidden_size,
                num_layers=1,
                dropout=0,
                bidirectional=bidirectional,
                batch_first=True) for l in range(num_layers)
        ]
        if recurrent_dropout:
            self.lstms = [
                WeightDrop(lstm, ['weight_hh_l0'], dropout=recurrent_dropout)
                for lstm in self.lstms
            ]

        self.lstms = torch.nn.ModuleList(self.lstms)
        self.ff_dropout = ff_dropout
        self.num_layers = num_layers

    def forward(self, input, lengths, seq_len):
        """Expects input in format (batch, seq, features)"""
        output = input
        for lstm in self.lstms:
            output = self.locked_dropout(
                output, batch_first=True, p=self.ff_dropout)
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                output, lengths, batch_first=True, enforce_sorted=False)
            output, _ = lstm(packed, None)
            output, _ = torch.nn.utils.rnn.pad_packed_sequence(
                output, batch_first=True, total_length=seq_len)

        return output

class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, batch_first=False, p=0.5):
        if not self.training or not p:
            return x
        mask_shape = (x.size(0), 1, x.size(2)) if batch_first else (1,
                                                                    x.size(1),
                                                                    x.size(2))

        mask = x.data.new(*mask_shape).bernoulli_(1 - p).div_(1 - p)
        return mask * x



class WeightDrop(torch.nn.Module):
    def __init__(self, module, weights, dropout=0):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        if hasattr(module, "bidirectional") and module.bidirectional:
            self.weights.extend(
                [weight + "_reverse" for weight in self.weights])

        self.dropout = dropout
        for name_w in self.weights:
            w = getattr(self.module, name_w)
            self.register_parameter(name_w + '_raw', Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self, name_w + '_raw')

            w = None
            mask = torch.ones(1, raw_w.size(1))
            if raw_w.is_cuda: mask = mask.to(raw_w.device)
            mask = torch.nn.functional.dropout(
                mask, p=self.dropout, training=self.training)
            w = mask.expand_as(raw_w) * raw_w
            self.module._parameters[name_w] = w

    def forward(self, *args):
        self._setweights()
        with warnings.catch_warnings():
            # Ignore lack of flattening warning
            warnings.simplefilter("ignore")
            return self.module.forward(*args)


class Biaffine(nn.Module):
    r"""
    Biaffine layer for first-order scoring :cite:`dozat-etal-2017-biaffine`.
    This function has a tensor of weights :math:`W` and bias terms if needed.
    The score :math:`s(x, y)` of the vector pair :math:`(x, y)` is computed as :math:`x^T W y / d^s`,
    where `d` and `s` are vector dimension and scaling factor respectively.
    :math:`x` and :math:`y` can be concatenated with bias terms.
    Args:
        n_in (int):
            The size of the input feature.
        n_out (int):
            The number of output channels.
        scale (float):
            Factor to scale the scores. Default: 0.
        bias_x (bool):
            If ``True``, adds a bias term for tensor :math:`x`. Default: ``True``.
        bias_y (bool):
            If ``True``, adds a bias term for tensor :math:`y`. Default: ``True``.
    """

    def __init__(self, n_in, n_out=1, scale=0, bias_x=True, bias_y=True):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.scale = scale
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(n_out, n_in+bias_x, n_in+bias_y))

        self.reset_parameters()

    def __repr__(self):
        s = f"n_in={self.n_in}"
        if self.n_out > 1:
            s += f", n_out={self.n_out}"
        if self.scale != 0:
            s += f", scale={self.scale}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return f"{self.__class__.__name__}({s})"

    def reset_parameters(self):
        nn.init.zeros_(self.weight)

    def forward(self, x, y):
        r"""
        Args:
            x (torch.Tensor): ``[batch_size, seq_len, n_in]``.
            y (torch.Tensor): ``[batch_size, seq_len, n_in]``.
        Returns:
            ~torch.Tensor:
                A scoring tensor of shape ``[batch_size, n_out, seq_len, seq_len]``.
                If ``n_out=1``, the dimension for ``n_out`` will be squeezed automatically.
        """

        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        # remove dim 1 if n_out == 1
        s = s.squeeze(1) / self.n_in ** self.scale

        return s


class RoEmbedding(nn.Module):
    def __init__(self, cfg, input_size):
        super().__init__()
        self.cfg = cfg
        # 第一个2代表，切分成两块；第二个2代表，一共有2个标签（正+负）
        self.dense = nn.Linear(input_size, cfg.inner_dim * 2 * 2)

    def custom_sinusoidal_position_embedding(self, token_index, pos_type=1):
        """
        See RoPE paper: https://arxiv.org/abs/2104.09864
        """
        output_dim = self.cfg.inner_dim
        position_ids = token_index.unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float).to(self.cfg.device)
        if pos_type == 0:
            indices = torch.pow(10000, -2 * indices / output_dim)
        else:
            indices = torch.pow(15, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((1, *([1]*len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (1, len(token_index), output_dim))
        embeddings = embeddings.squeeze(0)
        return embeddings
    
    def get_ro_embedding(self, qw: torch.Tensor, kw: torch.Tensor, token_index):
        """_summary_
        Parameters
        ----------
        qw : torch.Tensor, (seq_len, class_nums, hidden_size)
        kw : torch.Tensor, (seq_len, class_nums, hidden_size)
        """

        x, y = token_index, token_index

        x_pos_emb = self.custom_sinusoidal_position_embedding(x)
        y_pos_emb = self.custom_sinusoidal_position_embedding(y)

        x_pos_emb = x_pos_emb.unsqueeze(0)
        y_pos_emb = y_pos_emb.unsqueeze(0)

        x_cos_pos = x_pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
        x_sin_pos = x_pos_emb[...,  None, ::2].repeat_interleave(2, dim=-1)
        cur_qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
        cur_qw2 = cur_qw2.reshape(qw.shape)
        cur_qw = qw * x_cos_pos + cur_qw2 * x_sin_pos

        y_cos_pos = y_pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
        y_sin_pos = y_pos_emb[...,  None, ::2].repeat_interleave(2, dim=-1)
        cur_kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
        cur_kw2 = cur_kw2.reshape(kw.shape)
        cur_kw = kw * y_cos_pos + cur_kw2 * y_sin_pos

        pred_logits = torch.einsum('bmhd,bnhd->bmnh', cur_qw, cur_kw).contiguous()
        return pred_logits


    def classify_matrix(self, sequence_outputs, input_labels, masks, similarity):

        outputs = self.dense(sequence_outputs)
        outputs = torch.split(outputs, self.cfg.inner_dim * 2, dim=-1)
        outputs = torch.stack(outputs, dim=-2)
        q_utterance, k_utterance = torch.split(outputs, self.cfg.inner_dim, dim=-1)

        # q_utterance: batch_size, seq_len, class_nums, inner_dim

        token_index = torch.arange(0, sequence_outputs.shape[1]).to(self.cfg.device)

        # ro_logits = []
        ro_logits = self.get_ro_embedding(q_utterance, k_utterance, token_index)

        pred_logits = ro_logits

        pred_logits = pred_logits * similarity

        # criterion = nn.CrossEntropyLoss(sequence_outputs.new_tensor([1.0] + [self.cfg.loss_weight[mat_name]] * (nums - 1)))
        criterion = nn.CrossEntropyLoss(sequence_outputs.new_tensor([1.0] + [1.0]))

        active_loss = masks.view(-1) == 1
        active_logits = pred_logits.view(-1, pred_logits.shape[-1])[active_loss]
        active_labels = input_labels.view(-1)[active_loss]
        loss = criterion(active_logits, active_labels)

        return loss, pred_logits 


class SupConLoss(nn.Module):

    def __init__(self, temperature=0.5, scale_by_temperature=True):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, features, labels=None, mask=None):
        """
        输入:
            features: 输入样本的特征，尺寸为 [batch_size, hidden_dim].
            labels: 每个样本的ground truth标签，尺寸是[batch_size].
            mask: 用于对比学习的mask，尺寸为 [batch_size, batch_size], 如果样本i和j属于同一个label，那么mask_{i,j}=1
        输出:
            loss值
        """
        # device = (torch.device('cuda')
                #   if features.is_cuda
                #   else torch.device('cpu'))
        device = features.device
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        # 关于labels参数
        if labels is not None and mask is not None:  # labels和mask不能同时定义值，因为如果有label，那么mask是需要根据Label得到的
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None: # 如果没有labels，也没有mask，就是无监督学习，mask是对角线为1的矩阵，表示(i,i)属于同一类
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None: # 如果给出了labels, mask根据label得到，两个样本i,j的label相等时，mask_{i,j}=1
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        '''
        示例:
        labels:
            tensor([[1.],
                    [2.],
                    [1.],
                    [1.]])
        mask:  # 两个样本i,j的label相等时，mask_{i,j}=1
            tensor([[1., 0., 1., 1.],
                    [0., 1., 0., 0.],
                    [1., 0., 1., 1.],
                    [1., 0., 1., 1.]])
        '''
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)  # 计算两两样本间点乘相似度
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)
        '''
        logits是anchor_dot_contrast减去每一行的最大值得到的最终相似度
        示例: logits: torch.size([4,4])
        logits:
            tensor([[ 0.0000, -0.0471, -0.3352, -0.2156],
                    [-1.2576,  0.0000, -0.3367, -0.0725],
                    [-1.3500, -0.1409, -0.1420,  0.0000],
                    [-1.4312, -0.0776, -0.2009,  0.0000]])
        '''
        # 构建mask
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size).to(mask.device)
        positives_mask = mask * logits_mask
        negatives_mask = 1. - mask
        '''
        但是对于计算Loss而言，(i,i)位置表示样本本身的相似度，对Loss是没用的，所以要mask掉
        # 第ind行第ind位置填充为0
        得到logits_mask:
            tensor([[0., 1., 1., 1.],
                    [1., 0., 1., 1.],
                    [1., 1., 0., 1.],
                    [1., 1., 1., 0.]])
        positives_mask:
        tensor([[0., 0., 1., 1.],
                [0., 0., 0., 0.],
                [1., 0., 0., 1.],
                [1., 0., 1., 0.]])
        negatives_mask:
        tensor([[0., 1., 0., 0.],
                [1., 0., 1., 1.],
                [0., 1., 0., 0.],
                [0., 1., 0., 0.]])
        '''
        num_positives_per_row  = torch.sum(positives_mask , axis=1) # 除了自己之外，正样本的个数  [2 0 2 2]
        denominator = torch.sum(
        exp_logits * negatives_mask, axis=1, keepdims=True) + torch.sum(
            exp_logits * positives_mask, axis=1, keepdims=True)

        log_probs = logits - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")


        log_probs = torch.sum(
            log_probs*positives_mask , axis=1)[num_positives_per_row > 0] / num_positives_per_row[num_positives_per_row > 0]
        '''
        计算正样本平均的log-likelihood
        考虑到一个类别可能只有一个样本，就没有正样本了 比如我们labels的第二个类别 labels[1,2,1,1]
        所以这里只计算正样本个数>0的
        '''
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")
        # loss
        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature
        loss = loss.mean()
        return loss

# 创建异构图
def build_hgraph(utterance_num=None):
    if utterance_num is None:
        utterance_num = 2
        #0是all，1是sub
        # edges = torch.tensor([ [[0, 0], [0, 1]], [[0, 0], [1, 0]], [[0, 0], [0, 1]]])
    # else:
    aa_edges0 = torch.Tensor([[i, i + 1] for i in range(utterance_num - 1)])
    aa_edges1 = torch.Tensor([[i + 1, i] for i in range(utterance_num - 1)])
    aa_edges2 = torch.Tensor([[i, i] for i in range(utterance_num)])
    aa_edges = torch.cat((aa_edges0, aa_edges1, aa_edges2), dim=0)

    as_edges = torch.cat((
            torch.tensor([[i, 0 * utterance_num + i] for i in range(utterance_num)]),
            torch.tensor([[i, 1 * utterance_num + i] for i in range(utterance_num)]),
            torch.tensor([[i, 2 * utterance_num + i] for i in range(utterance_num)]),
    ), dim=0)
    sa_edges = as_edges[:, [1, 0]]
    edges = [aa_edges, as_edges, sa_edges]

    # print('nn', edges[0].shape)
    g = dgl.heterograph({
        ('all', 'dependency0', 'all'): edges[0].tolist(),
        ('sub', 'dependency1', 'all'): edges[2].tolist(),
        ('all', 'dependency2', 'sub'): edges[1].tolist(),
    })

# To print nodes and edges, you have to specify the node/edge type
    # for ntype in g.ntypes:
        # print(f"Nodes of type '{ntype}': {g.nodes(ntype)}")

    # for etype in g.etypes:
        # src, dst = g.edges(etype=etype)
        # print(f"Edges of type '{etype}': {src} -> {dst}")
    return g
# build_hgraph(2)




class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None and len(mask.shape) == 3:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


class FusionGate(nn.Module):
    def __init__(self, hid_size):
        super(FusionGate, self).__init__()
        self.fuse_weight = nn.Parameter(torch.Tensor(hid_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.fuse_weight)

    def forward(self, a, b):
        # Compute fusion coefficients
        fusion_coef = torch.sigmoid(self.fuse_weight)
        # Fuse tensors a and b
        fused_tensor = fusion_coef * a + (1 - fusion_coef) * b
        return fused_tensor


class NewFusionGate(nn.Module):
    def __init__(self, hid_size):
        super(NewFusionGate, self).__init__()
        self.fuse = nn.Linear(hid_size * 2, hid_size)

    def forward(self, a, b):
        # Concatenate a and b along the last dimension
        concat_ab = torch.cat([a, b], dim=-1)
        # Apply the linear layer
        fusion_coef = torch.sigmoid(self.fuse(concat_ab))
        # Fuse tensors a and b
        fused_tensor = fusion_coef * a + (1 - fusion_coef) * b
        return fused_tensor
