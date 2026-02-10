import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import math
from typing import NamedTuple
from utils.tensor_functions import compute_in_batches, generate_binary_latent_codes

from nets.graph_encoder import GraphAttentionEncoder

from torch.nn import DataParallel
from utils.beam_search import CachedLookup
from utils.functions import sample_many, count_demands, concatenate_embeddings, split_features
from torch.nn.parallel import DistributedDataParallel


def set_decode_type(model, decode_type):

    if isinstance(model, (DataParallel, DistributedDataParallel)):
        model = model.module

    if hasattr(model, "set_decode_type"):
        model.set_decode_type(decode_type)
    else:
        print(f"Available methods: {dir(model)}")
        raise AttributeError(f"{type(model)} object has no attribute 'set_decode_type'")


class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):
            return AttentionModelFixed(
                node_embeddings=self.node_embeddings[key],
                context_node_projected=self.context_node_projected[key],
                glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
                glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
                logit_key=self.logit_key[key]
            )
        return tuple.__getitem__(key)


class AttentionModel(nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 problem,
                 n_encode_layers=2,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization='batch',
                 n_heads=8,
                 checkpoint_encoder=False,
                 shrink_size=None,
                 latent_dim=3):
        super(AttentionModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0

        self.tanh_clipping = tanh_clipping

        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        self.problem = problem
        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size

        self.latent_dim = latent_dim

        
        step_context_dim = embedding_dim + 1
        node_dim = 3
        self.init_embed_depot = nn.Linear(2, embedding_dim)
        self.W_placeholder = nn.Parameter(torch.Tensor(2 * embedding_dim))
        self.W_placeholder.data.uniform_(-1, 1)  # Placeholder should be in range of activations
        self.init_embed_pick = nn.Linear(node_dim, embedding_dim)
        self.init_embed_delivery = nn.Linear(node_dim, embedding_dim)

        if self.latent_dim is not None:
            self.num_trajectories = 2 ** self.latent_dim
            self.latent_embedding_layer = Polynet(embedding_dim, self.latent_dim, hidden_dim)
            self.latent_codes = generate_binary_latent_codes(self.latent_dim)

    

        self.init_embed = nn.Linear(node_dim, embedding_dim)

        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization
        )

        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_step_context = nn.Linear(step_context_dim, embedding_dim, bias=False)
        assert embedding_dim % n_heads == 0
        # Note n_heads * val_dim == embedding_dim so input to project_out is embedding_dim
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp

    def forward(self, input, return_pi=False, return_kl=False):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :param return_pi: whether to return the output sequences, this is optional as it is not compatible with
        using DataParallel as the results may be of different lengths on different GPUs
        :return:
        """
        latent_codes = self.latent_codes.repeat(input['loc'].size(0), 1).to(input['loc'].device)

        repeated_input = {k: v.repeat_interleave(self.num_trajectories, dim=0) for k, v in input.items()}

        if self.checkpoint_encoder and self.training:  # Only checkpoint if we need gradients
            embeddings, _ = checkpoint(self.embedder, self._init_embed(input), count_demands(input['demand']))
        else:
            embeddings, _ = self.embedder(self._init_embed(input), count_demands(input['demand']))

        repeated_embeddings = embeddings.repeat_interleave(self.num_trajectories, dim=0)

        _log_p, pi = self._inner(repeated_input, repeated_embeddings, latent_codes)

        cost, _ = self.problem.get_costs(repeated_input, pi)
        

        if self.training:
            ll = self._calc_log_likelihood(_log_p, pi, mask=None)

            if return_pi:
                return cost, ll, pi
            
            return cost, ll
        
        cost = cost.view(input['loc'].size(0), self.num_trajectories)
        pi = pi.view(input['loc'].size(0), self.num_trajectories, pi.size(-1))
        _log_p = _log_p.view(input['loc'].size(0), self.num_trajectories, _log_p.size(1), _log_p.size(2))

        min_cost, best_idx = cost.min(dim=1)

        best_pi = pi[torch.arange(input['loc'].size(0)), best_idx]
        best_log_p = _log_p[torch.arange(input['loc'].size(0)), best_idx]

        ll = self._calc_log_likelihood(best_log_p, best_pi, mask=None)

        if return_pi:
            return min_cost, ll, best_pi
    
        return min_cost, ll

        

    def beam_search(self, *args, **kwargs):
        return self.problem.beam_search(*args, **kwargs, model=self)

    def precompute_fixed(self, input):
        embeddings, _ = self.embedder(self._init_embed(input), count_demands(input['demand']))
        # Use a CachedLookup such that if we repeatedly index this object with the same index we only need to do
        # the lookup once... this is the case if all elements in the batch have maximum batch size
        return CachedLookup(self._precompute(embeddings))

    def propose_expansions(self, beam, fixed, expand_size=None, normalize=False, max_calc_batch_size=4096):
        # First dim = batch_size * cur_beam_size
        log_p_topk, ind_topk = compute_in_batches(
            lambda b: self._get_log_p_topk(fixed[b.ids], b.state, k=expand_size, normalize=normalize),
            max_calc_batch_size, beam, n=beam.size()
        )

        assert log_p_topk.size(1) == 1, "Can only have single step"
        # This will broadcast, calculate log_p (score) of expansions
        score_expand = beam.score[:, None] + log_p_topk[:, 0, :]

        # We flatten the action as we need to filter and this cannot be done in 2d
        flat_action = ind_topk.view(-1)
        flat_score = score_expand.view(-1)
        flat_feas = flat_score > -1e10  # != -math.inf triggers

        # Parent is row idx of ind_topk, can be found by enumerating elements and dividing by number of columns
        flat_parent = torch.arange(flat_action.size(-1), out=flat_action.new()) / ind_topk.size(-1)

        # Filter infeasible
        feas_ind_2d = torch.nonzero(flat_feas)

        if len(feas_ind_2d) == 0:
            # Too bad, no feasible expansions at all :(
            return None, None, None

        feas_ind = feas_ind_2d[:, 0]

        return flat_parent[feas_ind], flat_action[feas_ind], flat_score[feas_ind]

    def _calc_log_likelihood(self, _log_p, a, mask):

        # Get log_p corresponding to selected actions
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        if mask is not None:
            log_p[mask] = 0

        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        return log_p.sum(1)

    def _init_embed(self, input):
        embed_depot = self.init_embed_depot(input['depot'])[:, None, :]
        feature_all = torch.cat([input['loc'], input['demand'].unsqueeze(-1)], -1)
        features_pick, features_delivery = split_features(feature_all)
        embed_pick = self.init_embed_pick(features_pick)
        embed_delivery = self.init_embed_delivery(features_delivery)
        embeding_all = concatenate_embeddings(embed_pick, embed_delivery, features_pick, features_delivery)

        return torch.cat( (embed_depot, embeding_all), 1 )


    def _inner(self, input, embeddings, latent_codes=None):

        outputs = []
        sequences = []

        state = self.problem.make_state(input)

        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        fixed = self._precompute(embeddings)

        batch_size = state.ids.size(0)

        # Perform decoding steps
        i = 0
        while not (self.shrink_size is None and state.all_finished()):

            if self.shrink_size is not None:
                unfinished = torch.nonzero(state.get_finished() == 0)
                if len(unfinished) == 0:
                    break
                unfinished = unfinished[:, 0]
                # Check if we can shrink by at least shrink_size and if this leaves at least 16
                # (otherwise batch norm will not work well and it is inefficient anyway)
                if 16 <= len(unfinished) <= state.ids.size(0) - self.shrink_size:
                    # Filter states
                    state = state[unfinished]
                    fixed = fixed[unfinished]

            log_p, mask = self._get_log_p(fixed, state, latent_codes)

            # Select the indices of the next nodes in the sequences, result (batch_size) long
            selected = self._select_node(log_p.exp()[:, 0, :], mask[:, 0, :])  # Squeeze out steps dimension

            state = state.update(selected)

            # Now make log_p, selected desired output size by 'unshrinking'
            if self.shrink_size is not None and state.ids.size(0) < batch_size:
                log_p_, selected_ = log_p, selected
                log_p = log_p_.new_zeros(batch_size, *log_p_.size()[1:])
                selected = selected_.new_zeros(batch_size)

                log_p[state.ids[:, 0]] = log_p_
                selected[state.ids[:, 0]] = selected_

            # Collect output of step
            outputs.append(log_p[:, 0, :])
            sequences.append(selected)

            i += 1
        # Collected lists, return Tensor
        return torch.stack(outputs, 1), torch.stack(sequences, 1)

    def sample_many(self, input, batch_rep=1, iter_rep=1):
        """
        :param input: (batch_size, graph_size, node_dim) input node features
        :return:
        """
        # Bit ugly but we need to pass the embeddings as well.
        # Making a tuple will not work with the problem.get_cost function

        latent_codes = self.latent_codes.repeat(input['loc'].size(0), 1).to(input['loc'].device)

        repeated_input = {k: v.repeat_interleave(self.num_trajectories, dim=0) for k, v in input.items()}

        embeddings = self.embedder(self._init_embed(input), count_demands(input['demand']))[0]

        repeated_embeddings = embeddings.repeat_interleave(self.num_trajectories, dim=0)

        # embeddings = self.embedder(self._init_embed(repeated_input), count_demands(repeated_input['demand']))[0]
        input_tuple = (repeated_input, repeated_embeddings, latent_codes)

        def run_inner(input_tuple):
            return self._inner(input_tuple[0], input_tuple[1], input_tuple[2])
        
        def compute_costs(input_tuple, pi):
            return self.problem.get_costs(input_tuple[0], pi)
        
        pis, costs = sample_many(
                run_inner,
                compute_costs,
                input_tuple,
                batch_rep,
                iter_rep
            )

        costs = costs.view(input['loc'].size(0), self.num_trajectories)
        pis = pis.view(input['loc'].size(0), self.num_trajectories, pis.size(-1))

        min_cost, best_idx = costs.min(dim=1)

        best_pi = pis[torch.arange(input['loc'].size(0)), best_idx]

        return best_pi, min_cost

    def _select_node(self, probs, mask):

        assert (probs == probs).all(), "Probs should not contain any nans"

        if self.decode_type == "greedy":
            _, selected = probs.max(1)
            assert not mask.gather(1, selected.unsqueeze(
                -1)).data.any(), "Decode greedy: infeasible action has maximum probability"

        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)

            # Check if sampling went OK, can go wrong due to bug on GPU
            # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print('Sampled bad values, resampling!')
                selected = probs.multinomial(1).squeeze(1)

        else:
            assert False, "Unknown decode type"
        return selected

    def _precompute(self, embeddings, num_steps=1):

        # The fixed context projection of the graph embedding is calculated only once for efficiency
        graph_embed = embeddings.mean(1)
        # fixed context = (batch_size, 1, embed_dim) to make broadcastable with parallel timesteps
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]

        # The projection of the node embeddings for the attention is calculated once up front
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)

        # No need to rearrange key for logit as there is a single head
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous()
        )
        return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)

    def _get_log_p_topk(self, fixed, state, k=None, normalize=True):
        log_p, _ = self._get_log_p(fixed, state, normalize=normalize)

        # Return topk
        if k is not None and k < log_p.size(-1):
            return log_p.topk(k, -1)

        # Return all, note different from torch.topk this does not give error if less than k elements along dim
        return (
            log_p,
            torch.arange(log_p.size(-1), device=log_p.device, dtype=torch.int64).repeat(log_p.size(0), 1)[:, None, :]
        )

    def _get_log_p(self, fixed, state, latent_codes, normalize=True):

        # Compute query = context node embedding
        query = fixed.context_node_projected + \
                self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings, state))

        # Compute keys and values for the nodes
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed, state)

        # Compute the mask
        mask = state.get_mask()

        # Compute logits (unnormalized log_p)
        log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask, latent_codes)

        if normalize:
            log_p = torch.log_softmax(log_p / self.temp, dim=-1)

        assert not torch.isnan(log_p).any()

        return log_p, mask

    def _get_parallel_step_context(self, embeddings, state, from_depot=False):
        """
        Returns the context per step, optionally for multiple steps at once (for efficient evaluation of the model)
        
        :param embeddings: (batch_size, graph_size, embed_dim)
        :param prev_a: (batch_size, num_steps)
        :param first_a: Only used when num_steps = 1, action of first step or None if first step
        :return: (batch_size, num_steps, context_dim)
        """

        current_node = state.get_current_node()
        batch_size, num_steps = current_node.size()

        if from_depot:
            # 1st dimension is node idx, but we do not squeeze it since we want to insert step dimension
            # i.e. we actually want embeddings[:, 0, :][:, None, :] which is equivalent
            return torch.cat(
                (
                    embeddings[:, 0:1, :].expand(batch_size, num_steps, embeddings.size(-1)),
                    # used capacity is 0 after visiting depot
                    state.veh_load[:, :, None]
                ),
                -1
            )
        else:
            return torch.cat(
                (
                    torch.gather(
                        embeddings,
                        1,
                        current_node.contiguous()
                            .view(batch_size, num_steps, 1)
                            .expand(batch_size, num_steps, embeddings.size(-1))
                    ).view(batch_size, num_steps, embeddings.size(-1)),
                    state.veh_load[:, :, None]
                ),
                -1
            )


    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask, latent_codes):

        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        # Compute the glimpse, rearrange dimensions so the dimensions are (n_heads, batch_size, num_steps, 1, key_size)
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_steps, graph_size)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        if self.mask_inner:
            assert self.mask_logits, "Cannot mask inner without masking logits"
            compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf

        # Batch matrix multiplication to compute heads (n_heads, batch_size, num_steps, val_size)
        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)

        # Project to get glimpse/updated context node embedding (batch_size, num_steps, embedding_dim)
        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size))

        # Now projecting the glimpse is not needed since this can be absorbed into project_out
        # final_Q = self.project_glimpse(glimpse)

        glimpse = self.latent_embedding_layer(glimpse.squeeze(2), latent_codes).unsqueeze(2)  if latent_codes is not None else glimpse

        final_Q = glimpse
        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        # logits = 'compatibility'
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))

        # From the logits compute the probabilities by clipping, masking and softmax
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[mask] = -math.inf

        return logits, glimpse.squeeze(-2)

    def _get_attention_node_data(self, fixed, state):
        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )
    

class PlynetLayer(nn.Module):
    def __init__(self, embedding_dim, latent_dim, hidden_dim):
        super(PlynetLayer, self).__init__()
        self.latend_linear0 = nn.Linear(latent_dim, embedding_dim)
        self.latend_linear1 = nn.Linear(latent_dim + embedding_dim , hidden_dim)
        self.latend_linear2 = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, embedding, latent_code):
        """
        embedding: [batch_size, num_nodes, embedding_dim]
        latent_code: [batch_size, latent_dim]
        """
        # Expand latent_code to match sequence length
        latent_code_expanded = latent_code.unsqueeze(1).expand(-1, embedding.size(1), -1)

        x = torch.cat((embedding, latent_code_expanded), dim=-1)  # [batch, num_nodes, embed+latent]
        
        x = self.latend_linear1(x)
        x = torch.nn.functional.relu(x)
        x = self.latend_linear2(x)
        
        return x

class Polynet(nn.Module):
    def __init__(self, embedding_dim, latent_dim, hidden_dim):
        super(Polynet, self).__init__()
        self.PN_layer = PlynetLayer(embedding_dim, latent_dim, hidden_dim)
        self.norm = nn.LayerNorm(embedding_dim)  # Optional but recommended

    def forward(self, mha_embedding, latent_code):
        """
        mha_embedding: [batch_size, num_nodes, embedding_dim]
        latent_code: [batch_size, latent_dim]
        """
        latent_embedding = self.PN_layer(mha_embedding, latent_code)
        out = mha_embedding + latent_embedding  # Skip connection
        # out = self.norm(out)  # Optional normalization
        return out