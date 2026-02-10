import torch
import numpy as np
from torch import nn
import math
# from utils.functions import extract_h_pick_and_delivery as get_hpick_hdelivery
from utils.functions import split_embeddings

PD_counts = None

class SkipConnection(nn.Module):

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input):
        return input + self.module(input)


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        # pickup
        self.W1_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        
        self.W2_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))

        self.W3_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))

        self.W4_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        
        

        if embed_dim is not None:
            self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """

        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """

        # print(f'\n mha count: {PD_counts} \n')
        if h is None:
            h = q  # compute self-attention
        


        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()

        global PD_counts

        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)
        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)
        
        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        h_pick, h_delivery = split_embeddings(h, PD_counts)
        
        max_n_pick_= max([count[0] for count in PD_counts])
        max_n_delivery_ = max([count[1] for count in PD_counts])

        max_n_pick = h_pick.size()[1]
        max_n_delivery = h_delivery.size()[1]

        assert max_n_pick == max_n_pick_, "max_n_pick should be equal to the maximum number of pickups in the batch"
        assert max_n_delivery == max_n_delivery_, "max_n_delivery should be equal to the maximum number of deliveries in the batch"

        # Adjust shapes for all instances in the batch
        shp_q_allpick = (self.n_heads, batch_size, h_pick.size()[1], -1)
        shp_q_alldelivery = (self.n_heads, batch_size, h_delivery.size()[1], -1)
        shp_allpick = (self.n_heads, batch_size, h_pick.size()[1], -1)
        shp_alldelivery = (self.n_heads, batch_size, h_delivery.size()[1], -1)

        pick_flat = h_pick.contiguous().view(-1, input_dim)

        delivery_flat = h_delivery.contiguous().view(-1, input_dim)
        

        
        Q_pick_allpick = torch.matmul(pick_flat, self.W1_query).view(shp_q_allpick)
        K_allpick = torch.matmul(pick_flat, self.W_key).view(shp_allpick)
        V_allpick = torch.matmul(pick_flat, self.W_val).view(shp_allpick)

        compatibility_pick_allpick = self.norm_factor * torch.matmul(Q_pick_allpick, K_allpick.transpose(2, 3))
        if (compatibility_pick_allpick == 0).any():
            compatibility_pick_allpick.masked_fill_(compatibility_pick_allpick == 0, -torch.inf)

        Q_pick_alldelivery = torch.matmul(pick_flat, self.W2_query).view(shp_q_allpick)
        K_alldelivery = torch.matmul(delivery_flat, self.W_key).view(shp_alldelivery)
        V_alldelivery = torch.matmul(delivery_flat, self.W_val).view(shp_alldelivery)
        
        compatibility_pick_alldelivery = self.norm_factor * torch.matmul(Q_pick_alldelivery, K_alldelivery.transpose(2, 3))
        if (compatibility_pick_alldelivery == 0).any():
            compatibility_pick_alldelivery.masked_fill_(compatibility_pick_alldelivery == 0, -torch.inf)
        
        Q_delivery_allpickup = torch.matmul(delivery_flat, self.W3_query).view(shp_q_alldelivery)
        K_allpickup2 = torch.matmul(pick_flat, self.W_key).view(shp_allpick)
        V_allpickup2 = torch.matmul(pick_flat, self.W_val).view(shp_allpick)
        
        compatibility_delivery_allpick = self.norm_factor * torch.matmul(Q_delivery_allpickup, K_allpickup2.transpose(2, 3))
        if (compatibility_delivery_allpick == 0).any():
            compatibility_delivery_allpick.masked_fill_(compatibility_delivery_allpick == 0, -torch.inf)

        
        Q_delivery_alldelivery = torch.matmul(delivery_flat, self.W4_query).view(shp_q_alldelivery)
        K_alldelivery2 = torch.matmul(delivery_flat, self.W_key).view(shp_alldelivery)
        V_alldelivery2 = torch.matmul(delivery_flat, self.W_val).view(shp_alldelivery)
        
        compatibility_delivery_alldelivery = self.norm_factor * torch.matmul(Q_delivery_alldelivery, K_alldelivery2.transpose(2, 3))
        if (compatibility_delivery_alldelivery == 0).any():
            compatibility_delivery_alldelivery.masked_fill_(compatibility_delivery_alldelivery == 0, -torch.inf)

       
        compatibility_additional_allpick = torch.full((self.n_heads, batch_size, graph_size, max_n_pick), fill_value=-torch.inf, device=compatibility.device)
        
        compatibility_additional_alldelivery = torch.full((self.n_heads, batch_size, graph_size, max_n_delivery), fill_value=-torch.inf, device=compatibility.device)
        compatibility_additional_allpick2 = torch.full((self.n_heads, batch_size, graph_size, max_n_pick), fill_value=-torch.inf, device=compatibility.device)

        compatibility_additional_alldelivery2 = torch.full((self.n_heads, batch_size, graph_size, max_n_delivery), fill_value=-torch.inf, device=compatibility.device)
        
        compatibility_additional_allpick[:, :, 1:max_n_pick+1, :] = compatibility_pick_allpick
        compatibility_additional_alldelivery[:, :, 1:max_n_pick+1, :] = compatibility_pick_alldelivery
        
        compatibility_additional_alldelivery2[:, :, graph_size-max_n_delivery:, :] = compatibility_delivery_alldelivery
        compatibility_additional_allpick2[:, :, graph_size-max_n_delivery:, :] = compatibility_delivery_allpick

        compatibility = torch.cat([compatibility, compatibility_additional_allpick, compatibility_additional_alldelivery,
                                 compatibility_additional_alldelivery2, compatibility_additional_allpick2], dim=-1)
       
        
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -np.inf

        attn = torch.softmax(compatibility, dim=-1)  # [n_heads, batch_size, n_query, graph_size+1+n_pick*2] (graph_size include depot)
        
        
        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc


        heads = torch.matmul(attn[:, :, :, :graph_size], V)  # V: (self.n_heads, batch_size, graph_size, val_size)
        
        
        heads = heads + torch.matmul(
            attn[:, :, :, graph_size:graph_size+max_n_pick].view(self.n_heads, batch_size, graph_size, max_n_pick), 
            V_allpick)

        heads = heads + torch.matmul(
            attn[:, :, :, graph_size + max_n_pick : graph_size + max_n_pick +max_n_delivery].view(self.n_heads, batch_size, graph_size, max_n_delivery), 
            V_alldelivery)

        heads = heads +  torch.matmul(
                    attn[:, :, :, graph_size + max_n_pick +max_n_delivery : graph_size + max_n_pick + 2 * max_n_delivery].view(self.n_heads, batch_size, graph_size, max_n_delivery),
                    V_alldelivery2)
        
        heads = heads +  torch.matmul(
                    attn[:, :, :,  graph_size + max_n_pick + 2 * max_n_delivery : graph_size + 2 * max_n_pick + 2 * max_n_delivery].view(self.n_heads, batch_size, graph_size, max_n_pick),
                    V_allpickup2)


        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)
        

        return out


class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim, affine=True)

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):

        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input


class MultiHeadAttentionLayer(nn.Sequential):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden=512,
            normalization='batch',
    ):
        super(MultiHeadAttentionLayer, self).__init__(
            SkipConnection(
                MultiHeadAttention(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim
                )
            ),
            Normalization(embed_dim, normalization),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim, normalization)
        )


class GraphAttentionEncoder(nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            n_layers,
            node_dim=None,
            normalization='batch',
            feed_forward_hidden=512
    ):
        super(GraphAttentionEncoder, self).__init__()

        self.init_embed = nn.Linear(node_dim, embed_dim) if node_dim is not None else None

        self.layers = nn.Sequential(*(
            MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization)
            for _ in range(n_layers)
        ))

    def forward(self, x, count, mask=None):
        
        global PD_counts
        PD_counts = count
        
        assert mask is None, "TODO mask not yet supported!"

        # Batch multiply to get initial embeddings of nodes
        h = self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:2], -1) if self.init_embed is not None else x
        h = self.layers(h)

        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )
