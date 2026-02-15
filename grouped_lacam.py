import numpy as np
from collections import deque, defaultdict
from typing import List, Tuple, Set, Dict, Optional
import time
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import os

Vertex        = Tuple[int, int]
Configuration = Tuple[Vertex, ...]
AgentID       = int

class Graph:
    def __init__(self, grid_map: np.ndarray):
        self.grid_map           = grid_map
        self.height, self.width = grid_map.shape
        self.goal_distances: Dict[Vertex, Dict[Vertex, int]] = {}

    def neighbors(self, v: Vertex) -> List[Vertex]:
        row, col = v
        result = []
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            nr, nc = row+dr, col+dc
            if 0 <= nr < self.height and 0 <= nc < self.width and self.grid_map[nr, nc] == 0:
                result.append((nr, nc))
        return result

    def get_distance(self, start: Vertex, goal: Vertex) -> int:
        if start == goal:
            return 0
        if goal not in self.goal_distances:
            self._bfs_from(goal)
        return self.goal_distances[goal].get(start, float('inf'))

    def _bfs_from(self, goal: Vertex):
        dist = {goal: 0}
        q    = deque([goal])
        while q:
            cur = q.popleft()
            for nb in self.neighbors(cur):
                if nb not in dist:
                    dist[nb] = dist[cur] + 1
                    q.append(nb)
        self.goal_distances[goal] = dist


class GroupTracker:
    def __init__(self, num_agents: int):
        self.num_agents = num_agents
        self.parent = list(range(num_agents))
        self.rank = [0] * num_agents
        self.size = [1] * num_agents
        self.gid = [-1] * num_agents
        self.originalgid = [-1] * num_agents
    
    def find(self, i: int) -> int:
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
    
    def connect(self, k: int, j: int, gid: int):
        rootK = self.find(k)
        rootJ = self.find(j)
        
        if rootK == rootJ:
            return
        
        if self.rank[rootK] < self.rank[rootJ]:
            self.parent[rootK] = rootJ
            self.size[rootJ] += self.size[rootK]
            self.gid[rootJ] = gid
        elif self.rank[rootK] > self.rank[rootJ]:
            self.parent[rootJ] = rootK
            self.size[rootK] += self.size[rootJ]
            self.gid[rootK] = gid
        else:
            self.parent[rootJ] = rootK
            self.size[rootK] += self.size[rootJ]
            self.rank[rootK] += 1
            self.gid[rootK] = gid
    
    def populateGroupTracker(self, groups: List['Group']):
        for aGroup in groups:
            if len(aGroup.agents) == 0:
                continue
            
            root = min(aGroup.agents)
            self.gid[root] = aGroup.gid
            
            for agent in aGroup.agents:
                if agent != root:
                    self.connect(root, agent, aGroup.gid)
                self.originalgid[agent] = aGroup.gid
    
    def findGID(self, i: int) -> int:
        return self.gid[self.find(i)]
    
    def setGID(self, i: int, gid: int):
        self.gid[self.find(i)] = gid
    
    def getGroup(self, gid: int) -> Set[int]:
        groupOfAgents = set()
        for i in range(self.num_agents):
            if self.gid[self.find(i)] == gid:
                groupOfAgents.add(i)
        return groupOfAgents
    
    def getGroupedWith(self, group: 'Group') -> Set[int]:
        if len(group.agents) == 0:
            return set()
        gid = self.findGID(min(group.agents))
        return self.getGroup(gid)


class PIBT:
    def __init__(self, graph: Graph, goals: List[Vertex], verbose: bool = False):
        self.graph      = graph
        self.num_agents = len(goals)
        self.goals      = np.array(goals, dtype=np.int32)
        self.priorities = np.zeros(self.num_agents)
        self.verbose    = verbose

        self.current_locs     : np.ndarray
        self.next_locs        : np.ndarray
        self.occupied_current = np.full((graph.height, graph.width), -1, dtype=np.int32)
        self.occupied_next    = np.full((graph.height, graph.width), -1, dtype=np.int32)

        self.constrained_agents: Set[int] = set()
        self.group_tracker: Optional[GroupTracker] = None

    def plan_one_step(self, config: Configuration,
                      constraints: List[Tuple[AgentID, Vertex]],
                      group_tracker: Optional[GroupTracker] = None
                      ) -> Optional[Configuration]:
        
        self.group_tracker = group_tracker
        
        cdict = {aid: v for aid, v in constraints}
        self.constrained_agents = set(cdict.keys())

        self.current_locs = np.array(config, dtype=np.int32)
        self.next_locs    = np.full((self.num_agents, 2), -1, dtype=np.int32)

        for aid, vtx in cdict.items():
            self.next_locs[aid] = np.array(vtx, dtype=np.int32)

        self.occupied_current.fill(-1)
        self.occupied_next.fill(-1)
        for i in range(self.num_agents):
            r, c = self.current_locs[i]
            self.occupied_current[r, c] = i
        for i in range(self.num_agents):
            if self.next_locs[i, 0] != -1:
                r, c = self.next_locs[i]
                self.occupied_next[r, c] = i

        for i in range(self.num_agents):
            if not np.array_equal(self.current_locs[i], self.goals[i]):
                self.priorities[i] += 1
            else:
                self.priorities[i] = i * 0.001

        order = np.argsort(-self.priorities)

        for aid in order:
            aid = int(aid)
            if aid in self.constrained_agents:
                continue
            if self.next_locs[aid, 0] == -1:
                self._pibt_recursive(aid, None)

        locs = [(int(self.next_locs[i][0]), int(self.next_locs[i][1]))
                for i in range(self.num_agents)]

        for aid, req in cdict.items():
            if locs[aid] != req:
                return None
        if len(set(locs)) != self.num_agents:
            return None
        for i in range(self.num_agents):
            for j in range(i+1, self.num_agents):
                ci = (int(self.current_locs[i][0]), int(self.current_locs[i][1]))
                cj = (int(self.current_locs[j][0]), int(self.current_locs[j][1]))
                ni, nj = locs[i], locs[j]
                if ci == nj and cj == ni:
                    print(f"⚠️ SWAP DETECTED: Agent {i} and {j} - REJECTING config")
                    return None

        return tuple((int(r), int(c)) for r, c in locs)

    def _pibt_recursive(self, agent_id: AgentID,
                        blocked_by: Optional[AgentID]) -> bool:
        
        if agent_id in self.constrained_agents:
            return True

        cur  = tuple(self.current_locs[agent_id])
        goal = tuple(self.goals[agent_id])

        cands = self.graph.neighbors(cur) + [cur]
        cands.sort(key=lambda v: (
            self.graph.get_distance(v, goal),
            self.occupied_current[v[0], v[1]] != -1
        ))

        for cand in cands:
            # Vertex conflict
            if self.occupied_next[cand[0], cand[1]] != -1:
                occupant = int(self.occupied_next[cand[0], cand[1]])
                if self.group_tracker is not None:
                    gid = self.group_tracker.findGID(agent_id)
                    self.group_tracker.connect(agent_id, occupant, gid)
                continue
            
            # Edge conflict
            if blocked_by is not None:
                if tuple(self.current_locs[blocked_by]) == cand:
                    if self.group_tracker is not None:
                        gid = self.group_tracker.findGID(agent_id)
                        self.group_tracker.connect(agent_id, blocked_by, gid)
                    continue

            self.next_locs[agent_id] = np.array(cand, dtype=np.int32)
            self.occupied_next[cand[0], cand[1]] = agent_id

            occupant = int(self.occupied_current[cand[0], cand[1]])
            if (occupant != -1 and
                occupant != agent_id and
                self.next_locs[occupant, 0] == -1 and
                occupant not in self.constrained_agents):

                if not self._pibt_recursive(occupant, agent_id):
                    self.next_locs[agent_id] = np.array([-1, -1], dtype=np.int32)
                    self.occupied_next[cand[0], cand[1]] = -1
                    
                    if self.group_tracker is not None:
                        gid = self.group_tracker.findGID(agent_id)
                        self.group_tracker.connect(agent_id, occupant, gid)
                    
                    continue

            return True

        self.next_locs[agent_id] = self.current_locs[agent_id].copy()
        self.occupied_next[cur[0], cur[1]] = agent_id
        return False


class Group:
    
    _next_gid = 0
    
    def __init__(self, agents: Set[AgentID], positions: Configuration, 
                 graph: Graph, agent_order: List[AgentID]):
        self.agents = frozenset(agents)
        self.positions = tuple(positions[a] for a in sorted(agents))
        self.graph = graph
        self.gid = Group._next_gid
        Group._next_gid += 1
        
        self.constraints: deque = deque()
        self.avoidSuccessors: Set[Configuration] = set([self.positions])
        self.retryConstraint = True
        self.tryFirstParent = False
        self.parents: List[Tuple[int, Configuration]] = []
        self.parentsTraversed: Set[int] = set()
        
        self.order = [a for a in agent_order if a in agents]
        self._using_parent = False 
        self._initialize_constraints()
    
    def _initialize_constraints(self):
        if not self.order:
            return
        
        aid = self.order[0]
        agent_list = sorted(self.agents)
        if aid not in agent_list:
            return
        
        pos = self.positions[agent_list.index(aid)]
        neighbors = self.graph.neighbors(pos) + [pos]
        for nb in neighbors:
            self.constraints.append(ConstraintNode(None, aid, nb))
    
    def getNextConstraints(self) -> Optional[List[Tuple[AgentID, Vertex]]]:
        if not self.retryConstraint:
            self.updateConstraints()
        
        if self.constraints:
            return self.constraints[0].get_constraints()
        
        if self.parents:
            self._using_parent = True
            return self._convertToConstraints(self.parents[-1])
        
        return None
    
    def _convertToConstraints(self, parent: Tuple[int, Configuration]) -> List[Tuple[AgentID, Vertex]]:
        timestep, config = parent
        constraints = []
        for aid in self.agents:
            constraints.append((aid, config[aid]))
        return constraints
    
    def updateConstraints(self):
        if self.constraints:
            prevConstraint = self.constraints.popleft()
            new_constraints = self._createMore(prevConstraint)
            self.constraints.extend(new_constraints)
        elif self.parents:
            if self.tryFirstParent:
                self.tryFirstParent = False
            else:
                self.parents.pop()
    
    def _createMore(self, node: 'ConstraintNode') -> List['ConstraintNode']:
        depth = node.depth()
        
        if depth >= len(self.order):
            return []
        
        aid = self.order[depth]
        agent_list = sorted(self.agents)
        if aid not in agent_list:
            return []
        
        pos = self.positions[agent_list.index(aid)]
        neighbors = self.graph.neighbors(pos) + [pos]
        
        children = []
        for nb in neighbors:
            children.append(ConstraintNode(node, aid, nb))
        
        return children
    
    def addParent(self, config: Configuration, timestep: int):
        if timestep not in self.parentsTraversed:
            self.parents.append((timestep, config))
            group_positions = tuple(config[a] for a in sorted(self.agents))
            self.avoidSuccessors.add(group_positions)
    
    def backtrackedEdge(self, config: Configuration, timestep: int):
        self.parentsTraversed.add(timestep)
    
    def constraintsNotDone(self) -> bool:
        return len(self.constraints) > 0 or len(self.parents) > 0


class ConstraintNode:
    __slots__ = ('parent', 'who', 'where')

    def __init__(self, parent: Optional['ConstraintNode'],
                 who:    Optional[AgentID],
                 where:  Optional[Vertex]):
        self.parent = parent
        self.who    = who
        self.where  = where

    def depth(self) -> int:
        d, cur = 0, self.parent
        while cur is not None:
            d  += 1
            cur = cur.parent
        return d

    def get_constraints(self) -> List[Tuple[AgentID, Vertex]]:
        out, cur = [], self
        while cur is not None and cur.who is not None:
            out.append((cur.who, cur.where))
            cur = cur.parent
        return out


class GroupDatabase:
    
    def __init__(self):
        self.groups: Dict[Tuple[frozenset, tuple], Group] = {}
        self.creation_log: List[Dict] = []
    
    def getOrCreate(self, agents: Set[AgentID], config: Configuration,
                    graph: Graph, agent_order: List[AgentID]) -> Group:
        positions = tuple(config[a] for a in sorted(agents))
        key = (frozenset(agents), positions)
        
        if key not in self.groups:
            group = Group(agents, config, graph, agent_order)
            self.groups[key] = group
            self.creation_log.append({
                'agents': set(agents),
                'positions': set(positions),
                'created_at': len(self.creation_log)
            })
        
        return self.groups[key]
    
    def findApplicableGroups(self, config: Configuration) -> List[Group]:
        applicable = []
        
        for (agents, positions), group in self.groups.items():
            config_positions = tuple(config[a] for a in sorted(agents))
            
            if config_positions == positions:
                applicable.append(group)
        
        return applicable


class HighLevelNode:
    def __init__(self, config: Configuration, global_order: List[AgentID],
                 parent: Optional['HighLevelNode'],
                 graph: Graph, num_agents: int, timestep: int):
        self.config = config
        self.parent = parent
        self.graph = graph
        self.num_agents = num_agents
        self.global_order = global_order
        self.timestep = timestep
        
        self.global_tree: deque = deque()
        self.global_tree.append(ConstraintNode(None, None, None))
        
        self.path_from_root: List[Configuration] = []
        if parent is not None:
            self.path_from_root = parent.path_from_root + [parent.config]
        else:
            self.path_from_root = []

    def global_tree_exhausted(self) -> bool:
        return len(self.global_tree) == 0

    def pop_global_constraint(self) -> Optional[List[Tuple[AgentID, Vertex]]]:
        while self.global_tree:
            node = self.global_tree.popleft()
            if node.depth() < self.num_agents:
                idx = node.depth()
                aid = self.global_order[idx]
                vtx = self.config[aid]
                nbs = self.graph.neighbors(vtx) + [vtx]
                for nb in nbs:
                    self.global_tree.append(ConstraintNode(node, aid, nb))
                continue
            return node.get_constraints()
        return None


class GroupedLaCAM:
    def __init__(self, graph: Graph, starts: List[Vertex], goals: List[Vertex],
                 verbose: bool = True):
        self.graph = graph
        self.starts = tuple(starts)
        self.goals = tuple(goals)
        self.num_agents = len(starts)
        self.verbose = verbose
        self.pibt = PIBT(graph, goals, verbose=verbose)
        
        self.group_db = GroupDatabase()
        self.visited_configs: Set[Configuration] = set()
        
        self.timestep_log: List[Dict] = []
        self.all_explored_configs: List[Configuration] = []

    def solve(self, node_limit: int = 500_000,
              time_limit: float = 60.0) -> Optional[List[List[Vertex]]]:

        t0 = time.time()
        open_list: List[HighLevelNode] = []
        explored: Dict[Configuration, HighLevelNode] = {}

        init_order = self._get_initial_order()
        init_node = HighLevelNode(self.starts, init_order, None,
                                   self.graph, self.num_agents, timestep=0)
        open_list.append(init_node)
        explored[self.starts] = init_node
        nodes_gen, nodes_exp = 1, 0

        self.all_explored_configs.append(self.starts)

        while open_list:
            if time.time() - t0 > time_limit:
                if self.verbose:
                    print(f"Timeout ({time_limit}s). exp={nodes_exp} gen={nodes_gen}")
                return None
            if nodes_gen >= node_limit:
                if self.verbose:
                    print(f"Node limit. exp={nodes_exp} gen={nodes_gen}")
                return None

            node = open_list[-1]

            if node.config == self.goals:
                if self.verbose:
                    print(f"✓ Found! exp={nodes_exp} gen={nodes_gen}")
                    
                    messy_paths = self._configs_to_paths(self.all_explored_configs)
                    if messy_paths:
                        print(f"\n{'='*70}")
                        print("MESSY SEARCH PATH (all explored configs):")
                        print(f"{'='*70}")
                        for i, p in enumerate(messy_paths):
                            print(f"  Agent {i}: {p}")
                        print(f"Messy path length: {len(messy_paths[0])} timesteps")
                    
                    self._print_summary()
                
                clean_solution = self._backtrack(node)

                if self.verbose:
                    print(f"\n{'='*70}")
                    print("VALIDATING SOLUTION:")
                    print(f"{'='*70}")
                    is_valid = self.validate_solution(clean_solution)
                    if not is_valid:
                        print("WARNING: Solution has conflicts!")
                
                os.makedirs('data/logs', exist_ok=True)
                
                if messy_paths:
                    animate_mapf_solution(
                        self.graph.grid_map, messy_paths, 
                        list(self.starts), list(self.goals),
                        save_path='data/logs/search_messy.gif'
                    )
                
                animate_mapf_solution(
                    self.graph.grid_map, clean_solution,
                    list(self.starts), list(self.goals),
                    save_path='data/logs/solution_clean.gif'
                )
                
                return clean_solution

            if node.global_tree_exhausted():
                open_list.pop()
                continue

            nodes_exp += 1

            global_con = node.pop_global_constraint()
            if global_con is None:
                open_list.pop()
                continue

            new_config = self._get_next_config(node, global_con)
            if new_config is None:
                continue
            
            if new_config is not None:
                self.all_explored_configs.append(new_config)

            if new_config in explored:
                existing = explored[new_config]
                if existing not in open_list:
                    open_list.append(existing)
                continue

            new_order = self._get_order(new_config)
            new_node = HighLevelNode(new_config, new_order, node,
                                      self.graph, self.num_agents, 
                                      timestep=node.timestep + 1)
            open_list.append(new_node)
            explored[new_config] = new_node
            nodes_gen += 1

            if self.verbose and nodes_gen % 1000 == 0:
                print(f"  … {nodes_gen} nodes")

        if self.verbose:
            print(f"Open empty. exp={nodes_exp}")
        return None

    def _get_next_config(self, node: HighLevelNode,
                         global_con: List[Tuple[AgentID, Vertex]]
                         ) -> Optional[Configuration]:
        
        config_repeated = node.config in self.visited_configs
        
        timestep_info = {
            'timestep': node.timestep,
            'config': node.config,
            'repeated': config_repeated,
            'groups_detected': [],
            'applicable_groups': [],
            'parents_added': []
        }
        
        if not config_repeated:
            # FIRST VISIT: Run PIBT without constraints, detect groups
            self.visited_configs.add(node.config)

            if self.verbose:
                print(f"\n[T{node.timestep}] First visit - running PIBT")

            GT = GroupTracker(self.num_agents)
            for i in range(self.num_agents):
                GT.gid[i] = i

            config = self.pibt.plan_one_step(node.config, [], GT)

            if config is None:
                return None
            
            groups_detected = self._extract_groups_from_gt(GT, node.config)

            for agents in groups_detected:
                self.group_db.getOrCreate(agents, node.config, self.graph, node.global_order)
                timestep_info['groups_detected'].append(sorted(agents))

            if self.verbose and groups_detected:
                print(f"  Detected {len(groups_detected)} groups: {[sorted(g) for g in groups_detected]}")
            
            self.timestep_log.append(timestep_info)
            return config
        
        else:
            # REVISIT: Apply LaCAM + Group Constraints
            if self.verbose:
                print(f"\n[T{node.timestep}] LOOP DETECTED - applying group constraints")
            
            LaCAMConstraints = global_con
            Groups = self.group_db.findApplicableGroups(node.config)
            
            if self.verbose:
                print(f"  Found {len(Groups)} applicable groups")
            
            for group in Groups:
                timestep_info['applicable_groups'].append(sorted(group.agents))

                if node.parent is not None:
                    group.addParent(node.parent.config, node.parent.timestep)
                    timestep_info['parents_added'].append({
                        'group': sorted(group.agents),
                        'parent_timestep': node.parent.timestep
                    })
            
            groupQueue: deque = deque(Groups)
            GT = GroupTracker(self.num_agents)
            GT.populateGroupTracker(Groups)
            
            # Build partial config as we go
            partial_config = list(node.config)
            
            while groupQueue:
                group = groupQueue.popleft()
                
                if self.verbose:
                    print(f"  Planning group {sorted(group.agents)}")
                
                # Plan group and update partial_config directly
                # success = self._plan_group_incremental(
                    # group, tuple(partial_config), GT, LaCAMConstraints, partial_config
                # )

                success = self._plan_group_incremental(
                    group, node.config, GT, LaCAMConstraints, partial_config
                )
                
                if not success:
                    if self.verbose:
                        print(f"  Group {sorted(group.agents)} planning failed")
                    return None
                
                if hasattr(group, '_using_parent') and group._using_parent:
                    timestep_info.setdefault('backtracking_used', []).append(sorted(group.agents))
                    group._using_parent = False
            
            if self.verbose:
                print(f"  All groups planned successfully")
            
            self.timestep_log.append(timestep_info)
            return tuple(partial_config)
        
    def _extract_groups_from_gt(self, GT: GroupTracker, config: Configuration) -> List[Set[int]]:
        groups = []
        visited = set()
        
        for i in range(self.num_agents):
            if i in visited:
                continue
            
            root = GT.find(i)
            component = set()
            
            for j in range(self.num_agents):
                if GT.find(j) == root:
                    component.add(j)
                    visited.add(j)
            
            if len(component) > 1:
                groups.append(component)
        
        return groups

    def _plan_group_incremental(self, group: Group, current_config: Configuration,
                                GT: GroupTracker, LaCAMConstraints: List[Tuple[AgentID, Vertex]],
                                partial_config: List[Vertex]) -> bool:
        
        lacam_dict = {aid: v for aid, v in LaCAMConstraints}
        
        while group.constraintsNotDone():
            # Algorithm line 2: Set group's part of config to Null
            for aid in group.agents:
                partial_config[aid] = None
            
            constraints = group.getNextConstraints()
            
            if constraints is None:
                if self.verbose:
                    print(f"    No more constraints for group")
                break
            
            conflict = any(
                aid in lacam_dict and lacam_dict[aid] != v
                for aid, v in constraints
            )
            
            if conflict:
                if self.verbose:
                    print(f"    Conflict with LaCAM - using LaCAM constraints")
                merged_constraints = list(LaCAMConstraints)
                # for aid in group.agents:
                    # if aid in lacam_dict:
                        # partial_config[aid] = lacam_dict[aid]
                # return True
            
            else:
                merged_constraints = list(LaCAMConstraints) + [
                (aid, v) for aid, v in constraints 
                if aid not in lacam_dict
            ]
            
            base_config = tuple(
                partial_config[i] if partial_config[i] is not None else current_config[i]
                for i in range(self.num_agents)
            )
            
            success_config = self.pibt.plan_one_step(base_config, merged_constraints, GT)
            
            if success_config is not None:
                for aid in group.agents:
                    partial_config[aid] = success_config[aid]
                
                group.retryConstraint = False
                if self.verbose:
                    print(f"    Group planned successfully")
                return True
            
            if self.verbose:
                print(f"    Constraints failed, trying next")
            group.updateConstraints()
        
        if self.verbose:
            print(f"    Group constraints exhausted")
        return False

    def _get_initial_order(self) -> List[AgentID]:
        dists = [self.graph.get_distance(self.starts[i], self.goals[i])
                 for i in range(self.num_agents)]
        return list(np.argsort(dists)[::-1])

    def _get_order(self, config: Configuration) -> List[AgentID]:
        not_goal, at_goal = [], []
        for i in range(self.num_agents):
            (at_goal if config[i] == self.goals[i] else not_goal).append(i)
        return not_goal + at_goal

    def _backtrack(self, goal_node: HighLevelNode) -> List[List[Vertex]]:
        configs: List[Configuration] = []
        cur = goal_node
        while cur is not None:
            configs.append(cur.config)
            cur = cur.parent
        configs.reverse()
        paths = [[] for _ in range(self.num_agents)]
        for cfg in configs:
            for aid in range(self.num_agents):
                vtx = cfg[aid]
                paths[aid].append((int(vtx[0]), int(vtx[1])))
        return paths
    
    def _configs_to_paths(self, configs: List[Configuration]) -> List[List[Vertex]]:
        if not configs:
            return None
        
        paths = [[] for _ in range(self.num_agents)]
        for cfg in configs:
            for aid in range(self.num_agents):
                vtx = cfg[aid]
                paths[aid].append((int(vtx[0]), int(vtx[1])))
        return paths
    
    def validate_solution(self, solution: List[List[Vertex]]) -> bool:
        for t in range(len(solution[0]) - 1):
            positions_t = [solution[a][t] for a in range(self.num_agents)]
            positions_next = [solution[a][t+1] for a in range(self.num_agents)]
            
            # Check vertex conflicts
            if len(set(positions_next)) != self.num_agents:
                position_counts = {}
                for i, pos in enumerate(positions_next):
                    if pos not in position_counts:
                        position_counts[pos] = []
                    position_counts[pos].append(i)
                for pos, agents in position_counts.items():
                    if len(agents) > 1:
                        print(f"Vertex conflict at t={t+1}: Agents {agents} at {pos}")
                return False
            
            # Check swap conflicts
            for i in range(self.num_agents):
                for j in range(i+1, self.num_agents):
                    if positions_t[i] == positions_next[j] and positions_t[j] == positions_next[i]:
                        print(f"Swap conflict at t={t}: Agents {i} and {j}")
                        return False
        
        print("✅ Solution is valid - no conflicts!")
        return True

    def _print_summary(self):
        print("\n" + "="*100)
        print("SEARCH SUMMARY")
        print("="*100)
        
        self._print_database_table()
        self._print_timestep_table()
        self._print_parent_tracking_table()
        self._print_statistics_table()
    
    def _print_database_table(self):
        print("\n" + "="*100)
        print("TABLE 1: GROUP DATABASE")
        print("="*100)
        
        if not self.group_db.creation_log:
            print("No groups stored in database.")
            return
        
        print(f"{'Group ID':<10} {'Agents':<20} {'Positions':<60}")
        print("-" * 100)
        
        for i, entry in enumerate(self.group_db.creation_log, 1):
            agents_str = str(sorted(entry['agents']))
            positions_str = str(sorted(entry['positions']))
            if len(positions_str) > 58:
                positions_str = positions_str[:55] + "..."
            print(f"{i:<10} {agents_str:<20} {positions_str:<60}")
        
        print("="*100)
    
    def _print_timestep_table(self):
        print("\n" + "="*120)
        print("TABLE 2: TIMESTEP LOG")
        print("="*120)
        
        if not self.timestep_log:
            print("No timestep log available.")
            return
        
        print(f"{'T':<4} {'Repeated':<10} {'Groups Detected':<20} {'Applicable Groups':<20} {'Parents Added':<30}")
        print("-" * 120)
        
        for entry in self.timestep_log:
            t = entry['timestep']
            repeated = "YES" if entry['repeated'] else "NO"
            groups_detected = str(entry.get('groups_detected', []))
            if len(groups_detected) > 18:
                groups_detected = groups_detected[:15] + "..."
            applicable = str(entry.get('applicable_groups', []))
            if len(applicable) > 18:
                applicable = applicable[:15] + "..."
            
            parents_added = entry.get('parents_added', [])
            if parents_added:
                parent_str = f"{len(parents_added)} groups"
            else:
                parent_str = "-"
            
            print(f"{t:<4} {repeated:<10} {groups_detected:<20} {applicable:<20} {parent_str:<30}")
        
        print("="*120)
    
    def _print_parent_tracking_table(self):
        print("\n" + "="*120)
        print("TABLE 3: PARENT TRACKING & BACKTRACKING")
        print("="*120)
        
        parent_added_events = []
        backtrack_used_events = []
        
        for entry in self.timestep_log:
            for parent_info in entry.get('parents_added', []):
                parent_added_events.append({
                    'timestep': entry['timestep'],
                    'group': parent_info['group'],
                    'parent_timestep': parent_info['parent_timestep']
                })
            
            for group in entry.get('backtracking_used', []):
                backtrack_used_events.append({
                    'timestep': entry['timestep'],
                    'group': group
                })
        
        if not parent_added_events and not backtrack_used_events:
            print("No parent tracking events (groups never needed backtracking).")
            print("="*120)
            return
        
        if parent_added_events:
            print("\nParents Added (for potential backtracking):")
            print(f"{'Timestep':<10} {'Group':<20} {'Parent From T':<20}")
            print("-" * 60)
            for event in parent_added_events:
                print(f"{event['timestep']:<10} {str(event['group']):<20} {event['parent_timestep']:<20}")
        
        if backtrack_used_events:
            print("\n" + "="*60)
            print("Backtracking Actually Used:")
            print(f"{'Timestep':<10} {'Group':<20}")
            print("-" * 60)
            for event in backtrack_used_events:
                print(f"{event['timestep']:<10} {str(event['group']):<20}")
        else:
            print("\n" + "="*60)
            print("Backtracking Never Used (constraint tree was sufficient)")
        
        print("="*120)
    
    def _print_statistics_table(self):
        print("\n" + "="*100)
        print("TABLE 4: SUMMARY STATISTICS")
        print("="*100)
        
        total_timesteps = len(self.timestep_log)
        total_loops = sum(1 for e in self.timestep_log if e['repeated'])
        total_groups_detected = len(self.group_db.creation_log)
        
        print(f"{'Metric':<40} {'Value':<20}")
        print("-" * 100)
        
        stats = [
            ("Total timesteps processed", total_timesteps),
            ("Loops detected (config repeated)", total_loops),
            ("Unique groups stored in database", total_groups_detected),
        ]
        
        for metric, value in stats:
            print(f"{metric:<40} {value:<20}")
        
        print("="*100)


def animate_mapf_solution(grid_map: np.ndarray, solution: List[List[Vertex]], 
                         starts: List[Vertex], goals: List[Vertex],
                         save_path: str = 'mapf_solution.gif'):
    
    if not solution:
        print("No solution to animate")
        return
    
    timesteps = len(solution[0])
    num_agents = len(solution)
    height, width = grid_map.shape
    
    colors = plt.cm.tab10(range(num_agents))
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    def update(frame):
        ax.clear()
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(-0.5, height - 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_xticks(range(width))
        ax.set_yticks(range(height))
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Grouped LaCAM Solution - Timestep {frame}/{timesteps-1}", 
                    fontsize=16, fontweight='bold')
        
        for i in range(height):
            for j in range(width):
                if grid_map[i, j] == 1:
                    ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, 
                                              color='black', alpha=0.7))
        
        for agent_id, goal in enumerate(goals):
            gr, gc = goal
            ax.add_patch(patches.Circle((gc, gr), 0.25, 
                                       facecolor=colors[agent_id], alpha=0.2, 
                                       edgecolor=colors[agent_id], linewidth=2, linestyle='--'))
        
        for agent_id in range(num_agents):
            if frame > 0:
                path = solution[agent_id][:frame+1]
                rows = [p[0] for p in path]
                cols = [p[1] for p in path]
                ax.plot(cols, rows, '-', color=colors[agent_id], alpha=0.4, linewidth=2)
        
        for agent_id in range(num_agents):
            r, c = solution[agent_id][frame]
            ax.add_patch(patches.Circle((c, r), 0.35,
                                       facecolor=colors[agent_id], 
                                       edgecolor='black', linewidth=2))
            ax.text(c, r, str(agent_id), ha='center', va='center', 
                   fontsize=10, color='white', fontweight='bold')
        
        return []
    
    anim = animation.FuncAnimation(fig, update, frames=timesteps, 
                                  interval=500, repeat=True, blit=False)
    
    anim.save(save_path, writer='pillow', fps=2)
    plt.close()
    
    print(f"Animation saved to {save_path}")


def calculate_costs(solution, goals):
    if not solution:
        return float('inf'), float('inf')
    soc = sum(
        sum(1 for t in range(len(solution[a])) if solution[a][t] != goals[a])
        for a in range(len(solution))
    )
    return soc, len(solution[0]) - 1


def _print_map(grid, starts, goals):
    for r in range(grid.shape[0]):
        print("    " + "".join("# " if grid[r,c] else ". " for c in range(grid.shape[1])))
    for i, (s, g) in enumerate(zip(starts, goals)):
        print(f"    Agent {i}: {s} → {g}")


def test_corridor_swap():
    print("\n" + "="*70)
    print("TEST: Corridor Swap (2 agents)")
    print("="*70)
    grid   = np.array([[0,0,0],[1,0,1]], dtype=int)
    graph  = Graph(grid)
    starts = [(0,0),(0,2)]
    goals  = [(0,2),(0,0)]
    _print_map(grid, starts, goals)

    sol = GroupedLaCAM(graph, starts, goals, verbose=True).solve(time_limit=10.0)
    if sol:
        soc, ms = calculate_costs(sol, goals)
        print(f"\n{'='*70}")
        print("CLEAN SOLUTION PATH:")
        print(f"{'='*70}")
        print(f"SOC={soc}  Makespan={ms}")
        for i, p in enumerate(sol): 
            print(f"  Agent {i}: {p}")
        
        print(f"\nSaved: data/logs/search_messy.gif (all explored)")
        print(f"Saved: data/logs/solution_clean.gif (final solution)")
    else:
        print("\nFAILED")
    return sol is not None


def test_open_swap():
    print("\n" + "="*70)
    print("TEST: Open Swap (2 agents, 3×3)")
    print("="*70)
    grid   = np.zeros((3,3), dtype=int)
    graph  = Graph(grid)
    starts = [(0,0),(0,2)]
    goals  = [(0,2),(0,0)]
    _print_map(grid, starts, goals)

    sol = GroupedLaCAM(graph, starts, goals, verbose=True).solve(time_limit=10.0)
    if sol:
        soc, ms = calculate_costs(sol, goals)
        print(f"\n{'='*70}")
        print("CLEAN SOLUTION PATH:")
        print(f"{'='*70}")
        print(f"SOC={soc}  Makespan={ms}")
        for i, p in enumerate(sol): 
            print(f"  Agent {i}: {p}")
        
        print(f"\nSaved: data/logs/search_messy.gif (all explored)")
        print(f"Saved: data/logs/solution_clean.gif (final solution)")
    else:
        print("\nFAILED")
    return sol is not None


def test_two_pairs():
    print("\n" + "="*70)
    print("TEST: Two Pairs (4 agents, 5×5)")
    print("="*70)
    grid   = np.zeros((5,5), dtype=int)
    graph  = Graph(grid)
    starts = [(0,0),(0,4),(4,0),(4,4)]
    goals  = [(0,4),(0,0),(4,4),(4,0)]
    _print_map(grid, starts, goals)

    sol = GroupedLaCAM(graph, starts, goals, verbose=True).solve(time_limit=60.0)
    if sol:
        soc, ms = calculate_costs(sol, goals)
        print(f"\n{'='*70}")
        print("CLEAN SOLUTION PATH:")
        print(f"{'='*70}")
        print(f"SOC={soc}  Makespan={ms}")
        for i, p in enumerate(sol): 
            print(f"  Agent {i}: {p}")
        
        print(f"\nSaved: data/logs/search_messy.gif (all explored)")
        print(f"Saved: data/logs/solution_clean.gif (final solution)")
    else:
        print("\nFAILED")
    return sol is not None

def test_four_by_seven_swap():
    print("\n" + "="*70)
    print("TEST: 4x7 Grid with 3 Swapping Pairs (6 agents)")
    print("="*70)
    
    grid = np.zeros((4, 7), dtype=int)
    
    # Left column obstacles (col 0, rows 1-3)
    grid[1, 0] = 1
    grid[2, 0] = 1
    grid[3, 0] = 1
    
    # Column 2 obstacles (rows 1-3)
    grid[1, 2] = 1
    grid[2, 2] = 1
    grid[3, 2] = 1
    
    # Column 4 obstacles (rows 1-3)
    grid[1, 4] = 1
    grid[2, 4] = 1
    grid[3, 4] = 1
    
    # Right column obstacles (col 6, rows 1-3)
    grid[1, 6] = 1
    grid[2, 6] = 1
    grid[3, 6] = 1
    
    graph = Graph(grid)
    
    starts = [
        (1, 1),  # Agent 1
        (2, 1),  # Agent 2
        (1, 3),  # Agent 3
        (2, 3),  # Agent 4
        (1, 5),  # Agent 5
        (2, 5)   # Agent 6
    ]
    
    goals = [
        (2, 1),  # Agent 1 → down
        (1, 1),  # Agent 2 → up
        (2, 3),  # Agent 3 → down
        (1, 3),  # Agent 4 → up
        (2, 5),  # Agent 5 → down
        (1, 5)   # Agent 6 → up
    ]
    
    _print_map(grid, starts, goals)
    
    sol = GroupedLaCAM(graph, starts, goals, verbose=True).solve(time_limit=60.0)
    if sol:
        soc, ms = calculate_costs(sol, goals)
        print(f"\n{'='*70}")
        print("CLEAN SOLUTION PATH:")
        print(f"{'='*70}")
        print(f"SOC={soc}  Makespan={ms}")
        for i, p in enumerate(sol): 
            print(f"  Agent {i}: {p}")
        
        print(f"\nSaved: data/logs/search_messy.gif (all explored)")
        print(f"Saved: data/logs/solution_clean.gif (final solution)")
    else:
        print("\nFAILED")
    return sol is not None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grouped LaCAM (Fully Algorithm-Compliant)")
    parser.add_argument('--test', choices=['corridor','open','pairs','four_seven','all'],
                        default='corridor')
    args = parser.parse_args()

    print("\n" + "="*70)
    print("GROUPED LaCAM — FULLY ALGORITHM-COMPLIANT IMPLEMENTATION")
    print("="*70)

    results = {}
    if args.test in ('corridor','all'): results['corridor'] = test_corridor_swap()
    if args.test in ('open',   'all'):  results['open']     = test_open_swap()
    if args.test in ('pairs',  'all'):  results['pairs']    = test_two_pairs()
    if args.test in ('four_seven','all'): results['four_seven'] = test_four_by_seven_swap()

    print("\n" + "="*70)
    print("RESULTS")
    for name, ok in results.items():
        print(f"  {name:12s} {'✓ PASS' if ok else 'FAIL'}")
    print("="*70)
