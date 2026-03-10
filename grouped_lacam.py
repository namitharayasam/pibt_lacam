import numpy as np
from collections import deque, defaultdict
from typing import List, Tuple, Set, Dict, Optional
import time
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import os
import csv

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
            if self.occupied_next[cand[0], cand[1]] != -1:
                occupant = int(self.occupied_next[cand[0], cand[1]])
                if self.group_tracker is not None:
                    gid = self.group_tracker.findGID(agent_id)
                    self.group_tracker.connect(agent_id, occupant, gid)
                continue

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
        self.retryConstraint = True
        self.visit_count = 0  # only apply constraints when visit_count >= 1

        self.order = [a for a in agent_order if a in agents]
        self._initialize_constraints()

    def _initialize_constraints(self):
        self.constraints.clear()
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
        if not self.constraints:
            self._initialize_constraints()
        if self.constraints:
            return self.constraints[0].get_constraints()
        return None

    def updateConstraints(self):
        if self.constraints:
            prevConstraint = self.constraints.popleft()
            new_constraints = self._createMore(prevConstraint)
            self.constraints.extend(new_constraints)
        else:
            self._initialize_constraints()

    def _createMore(self, node: 'ConstraintNode') -> List['ConstraintNode']:
        depth = node.depth()
        next_depth = depth + 1
        if next_depth >= len(self.order):
            return []
        aid = self.order[next_depth]
        agent_list = sorted(self.agents)
        if aid not in agent_list:
            return []
        pos = self.positions[agent_list.index(aid)]
        neighbors = self.graph.neighbors(pos) + [pos]
        children = []
        for nb in neighbors:
            children.append(ConstraintNode(node, aid, nb))
        return children

    def constraintsNotDone(self) -> bool:
        return True


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
            out.append((int(cur.who), cur.where))
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
        else:
            self.groups[key].visit_count += 1
        return self.groups[key]

    def findApplicableGroups(self, config: Configuration) -> List[Group]:
        applicable = []
        for (agents, positions), group in self.groups.items():
            config_positions = tuple(config[a] for a in sorted(agents))
            if config_positions == positions:
                if group.visit_count >= 2:  # only apply constraints second time onwards
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
        self.expansion_count = 0


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
        if not self.global_tree:
            return None
        node = self.global_tree.popleft()
        if node.depth() < self.num_agents:
            idx = node.depth()
            aid = self.global_order[idx]
            vtx = self.config[aid]
            nbs = self.graph.neighbors(vtx) + [vtx]
            np.random.shuffle(nbs)
            for nb in nbs:
                self.global_tree.append(ConstraintNode(node, aid, nb))
        return node.get_constraints()


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

        self.search_snapshots: List[List[Configuration]] = []
        self.timestep_log: List[Dict] = []

        self.nodes_exp = 0
        self.nodes_gen = 0

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
        cycle_hits = 0

        while open_list:
            if time.time() - t0 > time_limit:
                if self.verbose:
                    print(f"Timeout ({time_limit}s). exp={nodes_exp} gen={nodes_gen}")
                self._save_search_gif()
                return None
            if nodes_gen >= node_limit:
                if self.verbose:
                    print(f"Node limit. exp={nodes_exp} gen={nodes_gen}")
                self._save_search_gif()
                return None

            node = open_list[-1]

            if len(self.search_snapshots) < 50:
                self.search_snapshots.append(node.path_from_root + [node.config])
            elif nodes_exp % 100 == 0:
                self.search_snapshots.append(node.path_from_root + [node.config])

            if node.config == self.goals:
                if self.verbose:
                    print(f"✓ Found! exp={nodes_exp} gen={nodes_gen}")
                    self._print_summary()

                clean_solution = self._backtrack(node)

                if self.verbose:
                    print(f"\n{'='*70}")
                    print("VALIDATING SOLUTION:")
                    print(f"{'='*70}")
                    is_valid = self.validate_solution(clean_solution)
                    soc, ms = calculate_costs(clean_solution, list(self.goals))
                    print(f"SOC={soc}  Makespan={ms}")
                    elapsed = time.time() - t0
                    print(f"Time: {elapsed:.2f}s")

                    if not is_valid:
                        print("WARNING: Solution has conflicts!")

                os.makedirs('data/logs', exist_ok=True)

                self.nodes_exp = nodes_exp
                self.nodes_gen = nodes_gen

                return clean_solution

            if node.global_tree_exhausted():
                open_list.pop()
                continue

            nodes_exp += 1

            # LaCAM constraint from this node's tree
            global_con = node.pop_global_constraint()
            if global_con is None:
                open_list.pop()
                continue

            node.expansion_count += 1
            if node.expansion_count > 2 * self.num_agents:
                new_config = self.pibt.plan_one_step(node.config, global_con, None)
                if new_config is None or new_config in explored:
                    open_list.pop()
                    continue
                new_order = self._get_order(new_config)
                new_node = HighLevelNode(new_config, new_order, node,
                                        self.graph, self.num_agents,
                                        timestep=node.timestep + 1)
                open_list.append(new_node)
                explored[new_config] = new_node
                nodes_gen += 1
                continue

            new_config = self._get_next_config(node, global_con)

            if new_config is None:
                continue

            else:
                frozen = [i for i in range(self.num_agents) if new_config[i] == node.config[i] and node.config[i] != self.goals[i]]
                if self.verbose and frozen:
                    print(f"  [FREEZE DETECT] T={node.timestep} Frozen agents: {frozen}")
                    print(f"    Their positions: {[node.config[i] for i in frozen]}")
                    print(f"    Their goals:     {[self.goals[i] for i in frozen]}")

            if new_config in explored:
                cycle_hits += 1
                new_config = self.pibt.plan_one_step(node.config, global_con, None)
                if new_config is None or new_config in explored:
                    continue

            new_order = self._get_order(new_config)
            new_node = HighLevelNode(new_config, new_order, node,
                                    self.graph, self.num_agents,
                                    timestep=node.timestep + 1)
            open_list.append(new_node)
            explored[new_config] = new_node
            nodes_gen += 1

            if self.verbose and nodes_gen % 1000 == 0:
                print(f"  … {nodes_gen} nodes, explored: {len(explored)}, cycle_hits: {cycle_hits}")

        if self.verbose:
            print(f"Open empty. exp={nodes_exp}")
        self._save_search_gif()
        return None


    def _get_next_config(self, node: HighLevelNode,
                         global_con: List[Tuple[AgentID, Vertex]]
                         ) -> Optional[Configuration]:

        if self.verbose:
            print(f"\n[T{node.timestep}] Planning from config {node.config}")

        GT = GroupTracker(self.num_agents)
        for i in range(self.num_agents):
            GT.gid[i] = i

        base_config = self.pibt.plan_one_step(node.config, [], GT)

        if base_config is None:
            return None

        groups_detected = self._extract_groups_from_gt(GT, node.config)
        for agents in groups_detected:
            self.group_db.getOrCreate(agents, node.config, self.graph, node.global_order)
            if self.verbose:
                print(f"  Detected/stored group at start: {sorted(agents)}")

            self.group_db.getOrCreate(agents, base_config, self.graph, node.global_order)
            if self.verbose:
                print(f"  Detected/stored group at end: {sorted(agents)}")

        applicable_groups = self.group_db.findApplicableGroups(node.config)

        if not applicable_groups:
            if self.verbose:
                print(f"  No applicable groups - using base LaCAM result")
            return self.pibt.plan_one_step(node.config, global_con, None)

        if self.verbose:
            print(f"  Found {len(applicable_groups)} applicable groups - trying group constraints")

        return self._try_with_group_constraints(node, global_con, applicable_groups)

    def _try_with_group_constraints(self, node: HighLevelNode,
                                    global_con: List[Tuple[AgentID, Vertex]],
                                    groups: List[Group]
                                    ) -> Optional[Configuration]:

        active_groups = [g for g in groups if g.constraintsNotDone()]

        if not active_groups:
            return self.pibt.plan_one_step(node.config, global_con, None)

        groupQueue: deque = deque(active_groups)
        GT = GroupTracker(self.num_agents)
        GT.populateGroupTracker(active_groups)
        partial_config = list(node.config)
        planned_next = {}

        # TRY CASE 1: G + L together
        all_groups_success = True
        while groupQueue:
            group = groupQueue.popleft()
            if self.verbose:
                print(f"    Planning group {sorted(group.agents)}")
            result = self._plan_group_incremental( group, node.config, GT, global_con, partial_config, planned_next)
            if result is None:
                continue
            elif not result:
                all_groups_success = False
                break

        if all_groups_success:
            final_config = tuple(partial_config)

            if len(set(final_config)) != self.num_agents:
                if self.verbose:
                    print(f"    ❌ Vertex conflict in group result")
                all_groups_success = False

            if all_groups_success:
                for i in range(self.num_agents):
                    for j in range(i+1, self.num_agents):
                        if (node.config[i] == final_config[j] and
                                node.config[j] == final_config[i]):
                            if self.verbose:
                                print(f"    ❌ Swap conflict between agents {i} and {j}")
                            all_groups_success = False
                            break
                    if not all_groups_success:
                        break

            if all_groups_success:
                if self.verbose:
                    print(f"    ✅ CASE 1: G + L both worked")
                return final_config

        # G + L failed - test blame
        if self.verbose:
            print(f"    ❌ G + L failed - testing blame...")

        # TEST: L alone
        lacam_only_result = self.pibt.plan_one_step(node.config, global_con, None)

        if lacam_only_result is not None:
            # CASE 3: G fails, L works → advance G, use L
            if self.verbose:
                print(f"    ✅ CASE 3: L works alone → advancing G constraints")
            for group in active_groups:
                group.updateConstraints()
            return lacam_only_result

        # TEST: G alone
        group_only_result = self._try_groups_only(node.config, active_groups)

        if group_only_result is not None:
            # CASE 2: G works, L fails → return None, let solver try next L
            if self.verbose:
                print(f"    ⚠️ CASE 2: G works but L failed → returning None for next L")
            return None

        # CASE 4: Both fail
        if self.verbose:
            print(f"    ❌ CASE 4: Both G and L failed → returning None")
        return None

    def _try_groups_only(self, config: Configuration, groups: List[Group]) -> Optional[Configuration]:
        planned_next = {}
        groupQueue: deque = deque(groups)
        GT = GroupTracker(self.num_agents)
        GT.populateGroupTracker(groups)
        partial_config = list(config)

        while groupQueue:
            group = groupQueue.popleft()
            result = self._plan_group_incremental(group, config, GT, [], partial_config, planned_next)
            if result is None:
                continue
            elif not result:
                return None

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
                                partial_config: List[Vertex], planned_next) -> Optional[bool]:

        if not group.constraintsNotDone():
            return None

        lacam_dict = {aid: v for aid, v in LaCAMConstraints}

        max_attempts = len(group.constraints)
        attempts = 0

        while group.constraintsNotDone():
            if attempts >= max_attempts:
                break
            attempts += 1
            if self.verbose:
                print(f"      Trying constraint, tree size: {len(group.constraints)}")
            for aid in group.agents:
                partial_config[aid] = None

            constraints = group.getNextConstraints()

            if constraints is None:
                if self.verbose:
                    print(f"      No more constraints for group")
                break

            # PARTIAL OVERRIDE: identify conflicting agents
            conflicting_agents = set()
            non_conflicting_constraints = []
            for aid, v in constraints:
                if int(aid) in lacam_dict and lacam_dict[int(aid)] != v:
                    conflicting_agents.add(int(aid))
                else:
                    non_conflicting_constraints.append((int(aid), v))

            if conflicting_agents:
                if self.verbose:
                    print(f"      Partial override: agents {conflicting_agents} → using LaCAM")
                merged_constraints = list(LaCAMConstraints) + [
                    (aid, v) for aid, v in non_conflicting_constraints
                    if aid not in lacam_dict
                ]
            else:
                merged_constraints = list(LaCAMConstraints) + [
                    (aid, v) for aid, v in constraints
                    if aid not in lacam_dict
                ]

            base_config = tuple(
                partial_config[i] if partial_config[i] is not None else current_config[i]
                for i in range(self.num_agents)
            )

            already_planned = [
                (i, partial_config[i])
                for i in range(self.num_agents)
                if i in planned_next and i not in group.agents and partial_config[i] is not None
            ]

            already_planned_dict = dict(already_planned)
            final_constraints = already_planned + [
                (aid, v) for aid, v in merged_constraints
                if aid not in already_planned_dict
            ]

            if self.verbose:
                print(f"      base_config: {base_config}")
                print(f"      final_constraints: {final_constraints}")

            success_config = self.pibt.plan_one_step(base_config, final_constraints, GT)

            if self.verbose:
                print(f"      result: {success_config}")

            if success_config is not None:
                for i in range(self.num_agents):
                    partial_config[i] = success_config[i]
                    planned_next[i] = success_config[i]
                group.retryConstraint = False
                if self.verbose:
                    print(f"      Group planned successfully")
                return True

            if self.verbose:
                print(f"      Constraints failed, trying next")
            group.updateConstraints()

        if self.verbose:
            print(f"      Group constraints exhausted during planning")
        return None

    def _get_initial_order(self) -> List[AgentID]:
        dists = [self.graph.get_distance(self.starts[i], self.goals[i])
                 for i in range(self.num_agents)]
        return [int(x) for x in np.argsort(dists)[::-1]]

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

    def validate_solution(self, solution: List[List[Vertex]]) -> bool:
        for t in range(len(solution[0]) - 1):
            positions_t    = [solution[a][t]   for a in range(self.num_agents)]
            positions_next = [solution[a][t+1] for a in range(self.num_agents)]

            if len(set(positions_next)) != self.num_agents:
                position_counts = {}
                for i, pos in enumerate(positions_next):
                    position_counts.setdefault(pos, []).append(i)
                for pos, agents in position_counts.items():
                    if len(agents) > 1:
                        print(f"Vertex conflict at t={t+1}: Agents {agents} at {pos}")
                return False

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
            agents_str    = str(sorted(entry['agents']))
            positions_str = str(sorted(entry['positions']))
            if len(positions_str) > 58:
                positions_str = positions_str[:55] + "..."
            print(f"{i:<10} {agents_str:<20} {positions_str:<60}")
        print("="*100)

    def _print_statistics_table(self):
        print("\n" + "="*100)
        print("TABLE 2: SUMMARY STATISTICS")
        print("="*100)
        total_groups_detected = len(self.group_db.creation_log)
        print(f"{'Metric':<40} {'Value':<20}")
        print("-" * 100)
        stats = [("Unique groups stored in database", total_groups_detected)]
        for metric, value in stats:
            print(f"{metric:<40} {value:<20}")
        print("="*100)

    def _save_search_gif(self):
        os.makedirs('data/logs', exist_ok=True)
        animate_mapf_search(
            self.graph.grid_map, self.search_snapshots,
            list(self.starts), list(self.goals),
            save_path='data/logs/search_messy.gif'
        )


def save_to_csv(filepath, map_name, num_agents, time, soc, makespan, nodes_exp, nodes_gen, success):
    file_exists = os.path.exists(filepath)
    with open(filepath, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['map', 'agents', 'time', 'soc', 'makespan', 'nodes_exp', 'nodes_gen', 'success'])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'map': map_name,
            'agents': num_agents,
            'time': round(time, 4),
            'soc': soc,
            'makespan': makespan,
            'nodes_exp': nodes_exp,
            'nodes_gen': nodes_gen,
            'success': success
        })

# File I/O helpers

def load_map_file(map_path: str) -> np.ndarray:
    with open(map_path, 'r') as f:
        lines = f.readlines()
    header = {}
    map_start_idx = 0
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('type'):
            header['type'] = line.split()[1]
        elif line.startswith('height'):
            header['height'] = int(line.split()[1])
        elif line.startswith('width'):
            header['width'] = int(line.split()[1])
        elif line.startswith('map'):
            map_start_idx = i + 1
            break
    grid_map = np.zeros((header['height'], header['width']), dtype=int)
    for i in range(header['height']):
        for j, ch in enumerate(lines[map_start_idx + i].strip()):
            if ch in ['@', 'O', 'T', 'W']:
                grid_map[i, j] = 1
    return grid_map


def load_scenario_file(scen_path: str, num_agents: int = None) -> Tuple[str, List[Vertex], List[Vertex]]:
    with open(scen_path, 'r') as f:
        lines = f.readlines()
    starts, goals, map_name = [], [], None
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) < 9:
            continue
        if map_name is None:
            map_name = parts[1]
        starts.append((int(parts[5]), int(parts[4])))  # (row, col)
        goals.append( (int(parts[7]), int(parts[6])))  # (row, col)
        if num_agents is not None and len(starts) >= num_agents:
            break
    return map_name, starts, goals


def save_solution_to_file(solution: List[List[Vertex]], filepath: str,
                          map_path: str = None, goals: List[Vertex] = None):
    if solution is None:
        print(f"No solution to save to {filepath}")
        return
    with open(filepath, 'w') as f:
        if map_path:
            f.write(f"Map_name: {map_path}\n")
        f.write(f"Num_agents: {len(solution)}\n")
        if goals:
            for i, g in enumerate(goals):
                f.write(f"Goal_for_agent {i}: ({g[0]},{g[1]})\n")
        for i, path in enumerate(solution):
            f.write(f"Agent {i}: " + '->'.join(f"({v[0]},{v[1]})" for v in path) + '->\n')
    print(f"Solution saved to: {filepath}")


# Animation helpers

def animate_mapf_search(grid_map: np.ndarray, search_snapshots: List[List[Configuration]],
                        starts: List[Vertex], goals: List[Vertex],
                        save_path: str = 'search_messy.gif'):
    if not search_snapshots:
        print("No search snapshots to animate")
        return

    num_agents = len(starts)
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

        snapshot = search_snapshots[frame]
        ax.set_title(f"Search Exploration - Snapshot {frame+1}/{len(search_snapshots)} "
                     f"(depth {len(snapshot)})", fontsize=16, fontweight='bold')

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

        per_agent_paths = [[] for _ in range(num_agents)]
        for cfg in snapshot:
            for aid in range(num_agents):
                per_agent_paths[aid].append(cfg[aid])

        for agent_id in range(num_agents):
            path = per_agent_paths[agent_id]
            if len(path) > 1:
                rows = [p[0] for p in path]
                cols = [p[1] for p in path]
                ax.plot(cols, rows, '-', color=colors[agent_id], alpha=0.4, linewidth=2)

        last_config = snapshot[-1]
        for agent_id in range(num_agents):
            r, c = last_config[agent_id]
            ax.add_patch(patches.Circle((c, r), 0.35,
                                        facecolor=colors[agent_id],
                                        edgecolor='black', linewidth=2))
            ax.text(c, r, str(agent_id), ha='center', va='center',
                    fontsize=10, color='white', fontweight='bold')

        return []

    anim = animation.FuncAnimation(fig, update, frames=len(search_snapshots),
                                   interval=500, repeat=True, blit=False)
    anim.save(save_path, writer='pillow', fps=2)
    plt.close()
    print(f"Search animation saved to {save_path}")


def animate_mapf_solution(grid_map: np.ndarray, solution: List[List[Vertex]],
                          starts: List[Vertex], goals: List[Vertex],
                          save_path: str = 'mapf_solution.gif'):
    if not solution:
        print("No solution to animate")
        return
    timesteps  = len(solution[0])
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
            at_goal = (solution[agent_id][frame] == goals[agent_id])
            color = '#90EE90' if at_goal else '#AEC6CF'
            ax.add_patch(patches.Circle((c, r), 0.35, facecolor=color, edgecolor='black', linewidth=2))
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


def test_benchmark_scenario(map_path: str, scen_path: str, num_agents: int = 10,
                            save_solution: bool = True):
    os.makedirs('data/logs', exist_ok=True)

    print(f"\nTEST: Benchmark Scenario")
    print(f"Map:     {map_path}")
    print(f"Scenario:{scen_path}")
    print(f"Agents:  {num_agents}")

    grid_map = load_map_file(map_path)
    graph = Graph(grid_map)
    print(f"Grid size: {grid_map.shape[0]}x{grid_map.shape[1]}")

    _, starts, goals = load_scenario_file(scen_path, num_agents)
    print(f"Loaded {len(starts)} agents")
    print(f"First agent: start={starts[0]}, goal={goals[0]}")

    print("\nRunning GroupedLaCAM...")
    t0 = time.time()
    solver = GroupedLaCAM(graph, starts, goals, verbose=True)
    solution = solver.solve(node_limit=500_000, time_limit=600.0)
    elapsed = time.time() - t0

    if solution:
        soc, makespan = calculate_costs(solution, goals)
        print(f"\nGroupedLaCAM SUCCESS!")
        print(f"  Time:           {elapsed:.4f}s")
        print(f"  Timesteps:      {len(solution[0])}")
        print(f"  Sum-of-costs:   {soc}")
        print(f"  Makespan:       {makespan}")
        print(f"  Nodes generated:{solver.nodes_gen}")
        print(f"  Nodes expanded: {solver.nodes_exp}")

        save_to_csv('results.csv', map_path, num_agents, elapsed, soc, makespan, solver.nodes_exp, solver.nodes_gen, True)

        if save_solution:
            save_solution_to_file(solution, 'data/logs/grouped_lacam_solution.txt',
                                  map_path=map_path, goals=goals)

        # Always save solution GIF
        animate_mapf_solution(grid_map, solution, starts, goals,
                              save_path='data/logs/benchmark_solution.gif')
    else:
        print("\nGroupedLaCAM FAILED")
        save_to_csv('results.csv', map_path, num_agents, elapsed, 0, 0,
                solver.nodes_exp, solver.nodes_gen, False)
        # Save partial search snapshots so there is still something to inspect
        animate_mapf_search(grid_map, solver.search_snapshots, starts, goals,
                            save_path='data/logs/benchmark_search_failure.gif')

    return solution


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
        print(f"\nSOC={soc}  Makespan={ms}")
        for i, p in enumerate(sol):
            print(f"  Agent {i}: {p}")
    else:
        print("\nFAILED")
    return sol is not None


def test_abc_example():
    print("\n" + "="*70)
    print("TEST: A-B-C Example")
    print("="*70)
    grid = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1]
    ], dtype=int)
    graph  = Graph(grid)
    starts = [(0,0),(1,3),(1,4)]
    goals  = [(0,4),(1,4),(1,3)]
    _print_map(grid, starts, goals)
    sol = GroupedLaCAM(graph, starts, goals, verbose=True).solve(time_limit=30.0)
    if sol:
        soc, ms = calculate_costs(sol, goals)
        print(f"\nSOC={soc}  Makespan={ms}")
        for i, p in enumerate(sol):
            print(f"  Agent {i}: {p}")
    else:
        print("\nFAILED")
    return sol is not None


def test_four_by_seven_swap():
    print("\n" + "="*70)
    print("TEST: 4x7 Grid with 3 Swapping Pairs (6 agents)")
    print("="*70)
    grid = np.zeros((4, 7), dtype=int)
    grid[1,0]=1; grid[2,0]=1; grid[3,0]=1
    grid[1,2]=1; grid[2,2]=1; grid[3,2]=1
    grid[1,4]=1; grid[2,4]=1; grid[3,4]=1
    grid[1,6]=1; grid[2,6]=1; grid[3,6]=1
    graph  = Graph(grid)
    starts = [(1,1),(2,1),(1,3),(2,3),(1,5),(2,5)]
    goals  = [(2,1),(1,1),(2,3),(1,3),(2,5),(1,5)]
    _print_map(grid, starts, goals)
    sol = GroupedLaCAM(graph, starts, goals, verbose=True).solve(time_limit=600.0)
    if sol:
        soc, ms = calculate_costs(sol, goals)
        print(f"\nSOC={soc}  Makespan={ms}")
        for i, p in enumerate(sol):
            print(f"  Agent {i}: {p}")
    else:
        print("\nFAILED")
    return sol is not None


def test_custom_3x8():
    print("\n" + "="*70)
    print("TEST: 3x8 Grid with obstacles (3 agents)")
    print("="*70)
    grid = np.zeros((3, 8), dtype=int)
    grid[1,2]=1; grid[1,3]=1; grid[1,4]=1
    grid[1,5]=1; grid[2,5]=1
    graph  = Graph(grid)
    starts = [(0,0),(2,3),(2,4)]
    goals  = [(0,7),(2,4),(2,3)]
    _print_map(grid, starts, goals)
    sol = GroupedLaCAM(graph, starts, goals, verbose=True).solve(time_limit=30.0)
    if sol:
        soc, ms = calculate_costs(sol, goals)
        print(f"\nSOC={soc}  Makespan={ms}")
        for i, p in enumerate(sol):
            print(f"  Agent {i}: {p}")
    else:
        print("\nFAILED")
    return sol is not None



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grouped LaCAM")
    parser.add_argument('--test', choices=['corridor', 'abc', 'four_seven', '3x8', 'all'])
    parser.add_argument('--map',    type=str, help='Path to .map file')
    parser.add_argument('--scen',   type=str, help='Path to .scen file')
    parser.add_argument('--agents', type=int, default=10, help='Number of agents (default: 10)')
    parser.add_argument('--save',   action='store_true', help='Save solution to text file')
    args = parser.parse_args()

    print("\n" + "="*70)
    print("GROUPED LaCAM — PHASE 3: GROUPS + LaCAM")
    print("="*70)

    if args.map and args.scen:
        test_benchmark_scenario(args.map, args.scen, args.agents, save_solution=args.save)
    elif args.test:
        results = {}
        if args.test in ('corridor',    'all'): results['corridor']   = test_corridor_swap()
        if args.test in ('abc',         'all'): results['abc']        = test_abc_example()
        if args.test in ('four_seven',  'all'): results['four_seven'] = test_four_by_seven_swap()
        if args.test in ('3x8',         'all'): results['3x8']        = test_custom_3x8()

        print("\n" + "="*70)
        print("RESULTS")
        for name, ok in results.items():
            print(f"  {name:12s} {'✓ PASS' if ok else 'FAIL'}")
        print("="*70)
    else:
        parser.print_help()
