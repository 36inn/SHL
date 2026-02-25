import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class HGTConv_ope(nn.Module):
    def __init__(self, in_dim_list, out_dim, n_heads, dropout=0.,
                 use_norm=False, use_RTE=True):
        super(HGTConv_ope, self).__init__()
        self.in_dim_src = in_dim_list[0]
        self.in_dim_dst = in_dim_list[1]
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.d_k    = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.use_norm = use_norm
        self.use_RTE = use_RTE
        self.att = None
        self.leaky_relu = nn.LeakyReLU(0.1)

        self.k_linear = nn.Linear(self.in_dim_src, out_dim)
        self.k_linear_dst = nn.Linear(self.in_dim_dst, out_dim)
        self.k_linear_edge=nn.Linear(1, out_dim)
        #self.att_edge=  nn.Parameter(torch.rand(size=(1, n_heads, self.d_k), dtype=torch.float))
        self.q_linear = nn.Linear(self.in_dim_dst, out_dim)
        self.v_linear = nn.Linear(self.in_dim_src, out_dim)
        self.v_linear_dst = nn.Linear(self.in_dim_dst, out_dim)


        self.a_linear = nn.Linear(out_dim, out_dim)
        self.norms    = nn.LayerNorm(out_dim)
        '''
            TODO: make relation_pri smaller, as not all <st, rt, tt> pair exist in meta relation list.
        '''
        self.relation_pri = nn.Parameter(torch.ones(2, self.n_heads))
        #self.relation_att = nn.Parameter(torch.Tensor(2, n_heads, self.d_k, self.d_k))
        #self.relation_msg = nn.Parameter(torch.Tensor(2, n_heads, self.d_k, self.d_k))
        self.relation_att = torch.ones(2, n_heads, self.d_k, self.d_k)
        self.relation_msg = torch.ones(2, n_heads, self.d_k, self.d_k)
        self.skip = nn.Parameter(torch.ones(1))
        self.drop = nn.Dropout(dropout)
        self.mid_linear = nn.Linear(out_dim, out_dim )
        self.out_linear = nn.Linear(out_dim , out_dim)
        self.out_norm = nn.LayerNorm(out_dim)

        # if self.use_RTE:
        # self.emb = RelTemporalEncoding(in_dim)

        self.reset_parameters()

    def reset_parameters(self):

        nn.init.xavier_normal_(self.k_linear.weight)
        nn.init.xavier_normal_(self.k_linear_dst.weight)
        nn.init.xavier_normal_(self.k_linear_edge.weight)
        nn.init.xavier_normal_(self.q_linear.weight)
        nn.init.xavier_normal_(self.v_linear.weight)
        nn.init.xavier_normal_(self.v_linear_dst.weight)
        nn.init.xavier_normal_(self.a_linear.weight)
            # nn.init.xavier_normal_(self.norms[i].weight)
        nn.init.xavier_normal_(self.relation_att)  # init.xavier_uniform_(self.relation_att)
        nn.init.xavier_normal_(self.relation_msg)

    def forward(self, **kwargs):
        ope_ma_adj_batch = kwargs.get('ope_ma_adj_batch')
        ope_ope_adj_batch = kwargs.get('ope_ope_adj_batch')
        batch_idxes = kwargs.get('batch_idxes')
        feat = kwargs.get('feat')

        h_src,h_dst,h_edge=feat[1],feat[0],feat[2]
        h_edge=h_edge.to(torch.float32)
        '''
            Create Attention and Message tensor beforehand.
        '''
        batch_size,n,m = ope_ma_adj_batch[batch_idxes, :, :].size()
        #res_att = torch.zeros(batch_size, n,  m, self.n_heads).to(ope_ma_adj_batch.device)
        #res_msg = torch.zeros(batch_size, n,  m, self.n_heads, self.d_k).to(ope_ma_adj_batch.device)
        '''
        # Step 1: Heterogeneous Mutual Attention
        '''
        m_o_q_mat = self.q_linear(h_dst).reshape(batch_size, n, self.n_heads,self.d_k)
        m_o_k_mat = self.k_linear(h_src).view(batch_size, m, self.n_heads, self.d_k)

        X = m_o_k_mat.permute(0, 2, 1, 3)
        Y = self.relation_att[0].expand(batch_size, -1, -1, -1)  # shape：batch_size,  num_head, d_k , d_k
        m_o_k_mat = torch.einsum('abcd,abdd->abcd', X, Y).permute(0, 2, 1, 3)
        m_o_att=(m_o_q_mat.unsqueeze(-3) * m_o_k_mat.unsqueeze(-4)).sum(dim=-1)* self.relation_pri[0] / (math.sqrt(self.d_k))



        edge_mat=self.k_linear_edge(h_edge.unsqueeze(-1)).view(batch_size, n,m, self.n_heads, self.d_k)
        ey = self.relation_att[0].unsqueeze(0).expand(batch_size, n, -1, -1,-1)
        edge_mat = torch.einsum('abcde,abdee->abcde', edge_mat, ey)
        ee=(m_o_q_mat.unsqueeze(-3)*edge_mat).sum(dim=-1)* self.relation_pri[0] / (math.sqrt(self.d_k))
        #ee=(edge_mat*self.att_edge).sum(dim=-1)* self.relation_pri[0]

        m_att = ope_ma_adj_batch[batch_idxes].unsqueeze(-1) * m_o_att  + ee
        m_att = self.leaky_relu(m_att)



        o_o_q_mat = m_o_q_mat
        o_o_k_mat = self.k_linear_dst(h_dst).view(batch_size, n, self.n_heads, self.d_k)
        X1 = o_o_k_mat.permute(0, 2, 1, 3)
        Y1 = self.relation_att[1].expand(batch_size, -1, -1, -1)
        o_o_k_mat = torch.einsum('abcd,abdd->abcd', X1, Y1).permute(0, 2, 1, 3)
        o_o_att = (o_o_q_mat.unsqueeze(-3) * o_o_k_mat.unsqueeze(-4)).sum(dim=-1)* self.relation_pri[1] / (math.sqrt(self.d_k))

        o_att = ope_ope_adj_batch[batch_idxes].unsqueeze(-1) * o_o_att
        o_att = self.leaky_relu(o_att)

        mask0= (ope_ma_adj_batch[batch_idxes].unsqueeze(-1)==1).expand_as(m_att)
        mask1= (ope_ope_adj_batch[batch_idxes].unsqueeze(-1)==1).expand_as(o_att)
        mask= torch.cat((mask0,mask1),dim=-2)
        att = torch.cat((m_att, o_att),dim=-2)
        att[~mask]=float('-1e9')
        #all_inf=torch.all(torch.isneginf(att), dim=-2)
        #indices = torch.nonzero(all_inf, as_tuple=True)

        att = F.softmax(att, dim=-2)
        '''
        #Step 2: Heterogeneous Message Passing
        '''
        m_v_mat = self.v_linear(h_src).view(batch_size, m, self.n_heads, self.d_k)

        x = m_v_mat.permute(0, 2, 1, 3)
        y = self.relation_msg[0].expand(batch_size, -1, -1,-1)
        C = torch.einsum('abcd,abdd->abcd', x, y).permute(0, 2, 1, 3)
        m_res_msg =  edge_mat + C.unsqueeze(-4)


        o_v_mat = self.v_linear_dst(h_dst).view(batch_size, n, self.n_heads, self.d_k)
        c =torch.einsum('abcd,abdd->abcd',
                               (o_v_mat.permute(0, 2, 1, 3)), (self.relation_msg[1].expand(batch_size, -1, -1,-1))).permute(0, 2, 1, 3)
        o_res_msg =c.unsqueeze(-4)+c.unsqueeze(-3)


        res_msg = torch.cat((m_res_msg, o_res_msg),dim=-3)

        a   = (res_msg * att.unsqueeze(-1)).sum(dim=-3)
        res  = a.view(batch_size, n,self.out_dim)


        '''
        Step 3: Target-specific Aggregation
        x = W[node_type] * gelu(Agg(x)) + x
        '''

        aggr_out = F.gelu(res)
        trans_out = self.drop(self.a_linear(aggr_out))
        alpha = torch.sigmoid(self.skip)
        if self.use_norm:
            out = self.norms(trans_out * alpha + (m_o_q_mat.view(batch_size, n,self.out_dim)) * (1 - alpha))
        else:
            out = trans_out * alpha + (m_o_q_mat.view(batch_size, n,self.out_dim)) * (1 - alpha)


        '''
        trans_out=self.drop(self.a_linear(res))+ (m_o_q_mat.view(batch_size, n,self.out_dim))
        if self.use_norm:
            trans_out=self.norms(trans_out)
        trans_out = self.drop(self.out_linear(F.gelu(self.mid_linear(trans_out)))) + trans_out
        out=self.out_norm(trans_out)
        '''
        return  out


