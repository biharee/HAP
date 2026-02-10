import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter


class StatePDP(NamedTuple):
    # Fixed input
    coords: torch.Tensor  # Depot + loc
    demand: torch.Tensor
    ids: torch.Tensor

    # State
    prev_a: torch.Tensor
    used_capacity: torch.Tensor
    veh_load: torch.Tensor  # Tracks current vehicle load
    visited_: torch.Tensor
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    i: torch.Tensor

    VEHICLE_CAPACITY = 1.0  # Hardcoded or could be set individually

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.demand.size(-1))

    @property
    def dist(self):
        return (self.coords[:, :, None, :] - self.coords[:, None, :, :]).norm(p=2, dim=-1)

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)
        return self._replace(
            ids=self.ids[key],
            prev_a=self.prev_a[key],
            used_capacity=self.used_capacity[key],
            veh_load=self.veh_load[key],  # Update veh_load
            visited_=self.visited_[key],
            lengths=self.lengths[key],
            cur_coord=self.cur_coord[key],
        )

    def initialize(input, visited_dtype=torch.uint8):

        depot = input['depot']
        loc = input['loc']
        demand = input['demand']


        veh_load = -torch.min(-demand.sum(dim=-1, keepdim=True), torch.zeros(demand.size(0), 1, device=demand.device))
        
    
        batch_size, n_loc, _ = loc.size()
        return StatePDP(
            coords=torch.cat((depot[:, None, :], loc), -2),
            demand=demand,
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device),
            used_capacity=demand.new_zeros(batch_size, 1),
            veh_load=veh_load,  # Initialized with demand of depot before setting it to zero
            visited_=torch.zeros(
                batch_size, 1, n_loc + 1, dtype=visited_dtype, device=loc.device
            ),
            lengths=torch.zeros(batch_size, 1, device=loc.device),
            cur_coord=depot[:, None, :],
            i=torch.zeros(1, dtype=torch.int64, device=loc.device)
        )
    
    def get_final_cost(self):

        assert self.all_finished()

        return self.lengths + (self.coords[self.ids, 0, :] - self.cur_coord).norm(p=2, dim=-1)

    def update(self, selected):
        
        assert self.i.size(0) == 1, "Can only update if state represents single step"

        selected = selected[:, None]  # Add dimension for step
        prev_a = selected
        n_loc = self.demand.size(-1)

        cur_coord = self.coords[self.ids, selected]
        lengths = self.lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1)

        selected_demand = self.demand[self.ids, torch.clamp(prev_a - 1, 0, n_loc - 1)]

        used_capacity = (self.used_capacity - selected_demand) * (prev_a != 0).float()

        veh_load = self.veh_load - selected_demand
 
        if self.visited_.dtype == torch.uint8:
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        else:
            visited_ = mask_long_scatter(self.visited_, prev_a - 1)

        return self._replace(
            prev_a=prev_a, used_capacity=used_capacity, veh_load=veh_load,
            visited_=visited_, lengths=lengths, cur_coord=cur_coord, i=self.i + 1
        )
    
    def all_finished(self):
        return self.i.item() >= self.demand.size(-1) and self.visited.all()

    def get_finished(self):
        return self.visited.sum(-1) == self.visited.size(-1)

    def get_current_node(self):
        return self.prev_a

    def get_mask(self):
        """
        Gets a (batch_size, n_loc + 1) mask with the feasible actions (0 = depot), depends on already visited and
        remaining capacity. 0 = feasible, 1 = infeasible
        Forbids to visit depot twice in a row, unless all nodes have been visited
        :return:
        """

        if self.visited_.dtype == torch.uint8:
            visited_loc = self.visited_[:, :, 1:]
        else:
            visited_loc = mask_long2bool(self.visited_, n=self.demand.size(-1))

        # For demand steps_dim is inserted by indexing with id, for used_capacity insert node dim for broadcasting
        # exceeds_cap = (self.demand[self.ids, :] + self.used_capacity[:, :, None] > self.VEHICLE_CAPACITY)
        # # Nodes that cannot be visited are already visited or too much demand to be served now
        # mask_loc = visited_loc.to(exceeds_cap.dtype) | exceeds_cap

        

        demand_exceeds_load = (self.demand[self.ids, :] >= self.veh_load[:, :, None] + 1e-4 )
        mask_loc = visited_loc.to(demand_exceeds_load.dtype) | demand_exceeds_load


        all_nodes_visited = visited_loc.all(dim=-1).bool()
        mask_depot =  (~ all_nodes_visited)

        return torch.cat((mask_depot[:, :, None], mask_loc), -1)



    def construct_solutions(self, actions):
        return actions