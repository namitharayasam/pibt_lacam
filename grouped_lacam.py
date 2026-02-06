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


class PIBT:
    def __init__(self, graph: Graph, goals: List[Vertex]):
        self.graph      = graph
        self.num_agents = len(goals)
        self.goals      = np.array(goals, dtype=np.int32)
        self.priorities = np.zeros(self.num_agents)

        self.current_locs     : np.ndarray
        self.next_locs        : np.ndarray
        self.occupied_current = np.full((graph.height, graph.width), -1, dtype=np.int32)
        self.occupied_next    = np.full((graph.height, graph.width), -1, dtype=np.int32)

        self.constrained_agents: Set[int]             = set()
        self.desired           : Dict[int, Vertex]    = {}
        self.failed_agents     : Set[int]             = set()
        self.bump_chains       : Dict[int, List[int]] = {}

    
    def plan_one_step(self, config: Configuration,
                      constraints: List[Tuple[AgentID, Vertex]]
                      ) -> Optional[Configuration]:

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

        self.bump_chains   = {}
        self.failed_agents = set()
        self.desired       = {}

        for i in range(self.num_agents):
            d = self._get_desired_position(i)
            self.desired[i] = (int(d[0]), int(d[1]))

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
                self._pibt_recursive(aid, None, [aid])

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

        for i in range(self.num_agents):
            if i not in self.constrained_agents and locs[i] != self.desired[i]:
                self.failed_agents.add(i)

        desired_groups: Dict[Vertex, List[int]] = defaultdict(list)
        for aid in range(self.num_agents):
            if aid in self.constrained_agents:
                continue
            d = self.desired[aid]
            desired_groups[(int(d[0]), int(d[1]))].append(aid)

        for _vtx, agents in desired_groups.items():
            if len(agents) < 2:
                continue
            if not any(a in self.failed_agents for a in agents):
                continue
            root = agents[0]
            if root not in self.bump_chains:
                self.bump_chains[root] = agents

        return tuple((int(r), int(c)) for r, c in locs)

    def _get_desired_position(self, agent_id: int) -> Vertex:
        cur  = tuple(self.current_locs[agent_id])
        goal = tuple(self.goals[agent_id])
        cands = self.graph.neighbors(cur) + [cur]
        cands.sort(key=lambda v: (
            self.graph.get_distance(v, goal),
            self.occupied_current[v[0], v[1]] != -1
        ))
        return cands[0]

    def _pibt_recursive(self, agent_id: AgentID,
                        blocked_by: Optional[AgentID],
                        chain: List[int]) -> bool:
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
                continue
            if blocked_by is not None:
                if tuple(self.current_locs[blocked_by]) == cand:
                    continue

            self.next_locs[agent_id] = np.array(cand, dtype=np.int32)
            self.occupied_next[cand[0], cand[1]] = agent_id

            occupant = int(self.occupied_current[cand[0], cand[1]])
            if (occupant != -1 and
                occupant != agent_id and
                self.next_locs[occupant, 0] == -1 and
                occupant not in self.constrained_agents):

                extended = chain + [occupant]
                if not self._pibt_recursive(occupant, agent_id, extended):
                    self.bump_chains[chain[0]] = extended
                    self.next_locs[agent_id] = np.array([-1, -1], dtype=np.int32)
                    self.occupied_next[cand[0], cand[1]] = -1
                    continue

            return True

        self.next_locs[agent_id] = self.current_locs[agent_id].copy()
        self.occupied_next[cur[0], cur[1]] = agent_id
        return False

    def detect_groups(self) -> List[Set[int]]:
        conflict: Dict[int, Set[int]] = defaultdict(set)

        for _root, chain in self.bump_chains.items():
            if len(chain) < 2:
                continue
            if not any(a in self.failed_agents for a in chain):
                continue
            for i in range(len(chain)):
                for j in range(i+1, len(chain)):
                    conflict[chain[i]].add(chain[j])
                    conflict[chain[j]].add(chain[i])

        visited: Set[int] = set()
        groups:  List[Set[int]] = []

        for agent in range(self.num_agents):
            if agent in visited or agent not in conflict:
                visited.add(agent)
                continue
            component: Set[int] = set()
            stack = [agent]
            while stack:
                cur = stack.pop()
                if cur in visited:
                    continue
                visited.add(cur)
                component.add(cur)
                for nb in conflict[cur]:
                    if nb not in visited:
                        stack.append(nb)
            if len(component) > 1:
                groups.append(component)

        return groups


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


class GroupConstraintTree:
    def __init__(self, graph: Graph, group: Set[AgentID],
                 config: Configuration, agent_order: List[AgentID]):
        self.graph   = graph
        self.group   = group
        self.order   = [a for a in agent_order if a in group]
        self.config  = config
        self.tree    : deque = deque()
        self.exhausted       = False
        self.last_global_at_reset: Optional[List[Tuple[AgentID, Vertex]]] = None
        self._initialize()

    def _initialize(self):
        root = ConstraintNode(None, None, None)
        if not self.order:
            self.exhausted = True
            return
        aid = self.order[0]
        vtx = self.config[aid]
        for nb in self.graph.neighbors(vtx) + [vtx]:
            self.tree.append(ConstraintNode(root, aid, nb))

    def reset(self, current_global: List[Tuple[AgentID, Vertex]]):
        self.tree.clear()
        self.exhausted            = False
        self.last_global_at_reset = list(current_global)
        self._initialize()

    def pop_next_constraint(self) -> Optional[List[Tuple[AgentID, Vertex]]]:
        while self.tree:
            node = self.tree.popleft()
            if node.depth() < len(self.order):
                idx = node.depth()
                aid = self.order[idx]
                vtx = self.config[aid]
                for nb in self.graph.neighbors(vtx) + [vtx]:
                    self.tree.append(ConstraintNode(node, aid, nb))
                continue
            return node.get_constraints()

        self.exhausted = True
        return None


class HighLevelNode:
    def __init__(self, config: Configuration, global_order: List[AgentID],
                 parent: Optional['HighLevelNode'],
                 graph: Graph, num_agents: int):
        self.config       = config
        self.parent       = parent
        self.graph        = graph
        self.num_agents   = num_agents
        self.global_order = global_order
        self.global_tree: deque = deque()
        self.global_tree.append(ConstraintNode(None, None, None))
        self.group_trees: Dict[frozenset, GroupConstraintTree] = {}

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
                np.random.shuffle(nbs)
                for nb in nbs:
                    self.global_tree.append(ConstraintNode(node, aid, nb))
                continue
            return node.get_constraints()
        return None

    def get_or_create_group_tree(self, group: Set[AgentID],
                                 graph: Graph) -> GroupConstraintTree:
        key = frozenset(group)
        if key not in self.group_trees:
            self.group_trees[key] = GroupConstraintTree(
                graph, group, self.config, self.global_order)
        return self.group_trees[key]


class GroupedLaCAM:
    def __init__(self, graph: Graph, starts: List[Vertex], goals: List[Vertex],
                 verbose: bool = True):
        self.graph      = graph
        self.starts     = tuple(starts)
        self.goals      = tuple(goals)
        self.num_agents = len(starts)
        self.verbose    = verbose
        self.pibt       = PIBT(graph, goals)
        self.group_detections = []

    def solve(self, node_limit: int = 500_000,
              time_limit: float = 60.0) -> Optional[List[List[Vertex]]]:

        t0 = time.time()
        open_list: List[HighLevelNode]              = []
        explored:  Dict[Configuration, HighLevelNode] = {}

        init_order = self._get_initial_order()
        init_node  = HighLevelNode(self.starts, init_order, None,
                                   self.graph, self.num_agents)
        open_list.append(init_node)
        explored[self.starts] = init_node
        nodes_gen, nodes_exp = 1, 0

        while open_list:
            if time.time() - t0 > time_limit:
                if self.verbose:
                    print(f"✗ Timeout ({time_limit}s). exp={nodes_exp} gen={nodes_gen}")
                return None
            if nodes_gen >= node_limit:
                if self.verbose:
                    print(f"✗ Node limit. exp={nodes_exp} gen={nodes_gen}")
                return None

            node = open_list[-1]

            if node.config == self.goals:
                if self.verbose:
                    print(f"✓ Found! exp={nodes_exp} gen={nodes_gen}")
                    if self.group_detections:
                        self._print_group_summary()
                    else:
                        print("\n(No groups detected during search)")
                return self._backtrack(node)

            if node.global_tree_exhausted():
                open_list.pop()
                continue

            nodes_exp += 1

            global_con = node.pop_global_constraint()
            if global_con is None:
                open_list.pop()
                continue

            new_config = self._generate_config(node, global_con)
            if new_config is None:
                continue

            if new_config in explored:
                existing = explored[new_config]
                if existing not in open_list:
                    open_list.append(existing)
                continue

            new_order = self._get_order(new_config)
            new_node  = HighLevelNode(new_config, new_order, node,
                                      self.graph, self.num_agents)
            open_list.append(new_node)
            explored[new_config] = new_node
            nodes_gen += 1

            if self.verbose and nodes_gen % 5000 == 0:
                print(f"  … {nodes_gen} nodes")

        if self.verbose:
            print(f"✗ Open empty. exp={nodes_exp}")
        return None

    def _generate_config(self, node: HighLevelNode,
                         global_con: List[Tuple[AgentID, Vertex]]
                         ) -> Optional[Configuration]:
        
        config = self.pibt.plan_one_step(node.config, global_con)
        if config is None:
            return None

        groups = self.pibt.detect_groups()

        if groups and self.verbose:
            print(f"  [DEBUG] Groups detected at config {node.config}: {[set(g) for g in groups]}")

        if not groups:
            return config

        if self.verbose:
            self.group_detections.append({
                'config': node.config,
                'global_con': global_con,
                'groups': [set(g) for g in groups],
                'failed': self.pibt.failed_agents.copy(),
                'desired': self.pibt.desired.copy(),
                'actual': config
            })

        return self._try_group_constraints(node, global_con, groups)

    def _try_group_constraints(self, node: HighLevelNode,
                               global_con: List[Tuple[AgentID, Vertex]],
                               groups: List[Set[AgentID]]
                               ) -> Optional[Configuration]:
        MAX_ATTEMPTS = 200
        global_dict  = {aid: v for aid, v in global_con}

        cached: Dict[frozenset, Optional[List[Tuple[AgentID, Vertex]]]] = {
            frozenset(g): None for g in groups
        }

        for attempt in range(MAX_ATTEMPTS):
            merged       = list(global_con)
            skip_attempt = False

            for group in groups:
                key   = frozenset(group)
                gtree = node.get_or_create_group_tree(group, self.graph)

                if cached[key] is None:
                    if gtree.exhausted:
                        if set(gtree.last_global_at_reset) == set(global_con):
                            return None
                        gtree.reset(global_con)
                        if gtree.exhausted:
                            return None

                    g_con = gtree.pop_next_constraint()
                    if g_con is None:
                        skip_attempt = True
                        break

                    cached[key] = g_con

                g_con = cached[key]

                conflict = any(
                    aid in global_dict and global_dict[aid] != v
                    for aid, v in g_con
                )
                if conflict:
                    cached[key] = None
                    skip_attempt = True
                    break

                merged.extend(g_con)

            if skip_attempt:
                continue

            seen: Dict[AgentID, Vertex] = {}
            dup = False
            for aid, v in merged:
                if aid in seen:
                    if seen[aid] != v:
                        dup = True
                        break
                seen[aid] = v
            if dup:
                for g in groups:
                    if aid in g:
                        cached[frozenset(g)] = None
                        break
                continue

            new_config = self.pibt.plan_one_step(node.config, merged)
            if new_config is None:
                cached = {k: None for k in cached}
                continue

            return new_config

        return None

    def _print_group_summary(self):
        print("\n" + "="*100)
        print("GROUP DETECTION SUMMARY")
        print("="*100)
        
        for i, det in enumerate(self.group_detections, 1):
            print(f"\nEvent {i}:")
            print(f"  Config: {det['config']}")
            print(f"  Global Constraints: {det['global_con']}")
            print(f"  Groups Detected: {det['groups']}")
            
            for g in det['groups']:
                failed_in_group = g & det['failed']
                if failed_in_group:
                    print(f"    Group {g} failures:")
                    for a in failed_in_group:
                        desired = det['desired'][a]
                        actual = det['actual'][a]
                        print(f"      Agent {a}: wanted {desired}, got {actual}")
        
        print("\n" + "="*100)

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


def visualize_solution_static(grid_map: np.ndarray, solution: List[List[Vertex]], 
                              starts: List[Vertex], goals: List[Vertex],
                              save_path: str = 'mapf_solution.png'):
    
    if not solution:
        print("No solution to visualize")
        return
    
    num_agents = len(solution)
    height, width = grid_map.shape
    colors = plt.cm.tab10(range(num_agents))
    
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(-0.5, height - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xticks(range(width))
    ax.set_yticks(range(height))
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Grouped LaCAM Solution - All Paths", fontsize=16, fontweight='bold')
    
    for i in range(height):
        for j in range(width):
            if grid_map[i, j] == 1:
                ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, 
                                          color='black', alpha=0.7))
    
    for agent_id in range(num_agents):
        path = solution[agent_id]
        rows = [p[0] for p in path]
        cols = [p[1] for p in path]
        ax.plot(cols, rows, '-', color=colors[agent_id], alpha=0.6, linewidth=3, 
               label=f'Agent {agent_id}')
    
    for agent_id, start in enumerate(starts):
        sr, sc = start
        ax.add_patch(patches.Rectangle((sc-0.4, sr-0.4), 0.8, 0.8,
                                       facecolor=colors[agent_id], alpha=0.5,
                                       edgecolor='black', linewidth=2))
        ax.text(sc, sr, 'S'+str(agent_id), ha='center', va='center', 
               fontsize=8, color='black', fontweight='bold')
    
    for agent_id, goal in enumerate(goals):
        gr, gc = goal
        ax.add_patch(patches.Circle((gc, gr), 0.35,
                                   facecolor=colors[agent_id], 
                                   edgecolor='black', linewidth=2))
        ax.text(gc, gr, 'G'+str(agent_id), ha='center', va='center', 
               fontsize=8, color='white', fontweight='bold')
    
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Static visualization saved to {save_path}")


def load_map_file(path: str) -> np.ndarray:
    with open(path) as f:
        lines = f.readlines()
    header, map_start = {}, 0
    for i, line in enumerate(lines):
        tok = line.strip().split()
        if   tok[0] == 'height': header['height'] = int(tok[1])
        elif tok[0] == 'width':  header['width']  = int(tok[1])
        elif tok[0] == 'map':    map_start = i+1; break
    H, W = header['height'], header['width']
    grid = np.zeros((H, W), dtype=int)
    for i in range(H):
        for j, ch in enumerate(lines[map_start+i].strip()):
            if ch in '@OTW':
                grid[i, j] = 1
    return grid


def load_scenario_file(path: str, num_agents: int = None):
    with open(path) as f:
        lines = f.readlines()
    starts, goals, map_name = [], [], None
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) < 9:
            continue
        if map_name is None:
            map_name = parts[1]
        starts.append((int(parts[5]), int(parts[4])))
        goals.append( (int(parts[7]), int(parts[6])))
        if num_agents and len(starts) >= num_agents:
            break
    return map_name, starts, goals


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
        print(f"\n  ✓ SOC={soc}  Makespan={ms}")
        for i, p in enumerate(sol): print(f"    Agent {i}: {p}")
        
        os.makedirs('data/logs', exist_ok=True)
        visualize_solution_static(grid, sol, starts, goals, 
                                 save_path='data/logs/corridor_swap_solution.png')
        animate_mapf_solution(grid, sol, starts, goals,
                             save_path='data/logs/corridor_swap_solution.gif')
    else:
        print("\n  ✗ FAILED")
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
        print(f"\n  ✓ SOC={soc}  Makespan={ms}")
        for i, p in enumerate(sol): print(f"    Agent {i}: {p}")
        
        os.makedirs('data/logs', exist_ok=True)
        visualize_solution_static(grid, sol, starts, goals, 
                                 save_path='data/logs/open_swap_solution.png')
        animate_mapf_solution(grid, sol, starts, goals,
                             save_path='data/logs/open_swap_solution.gif')
    else:
        print("\n  ✗ FAILED")
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

    sol = GroupedLaCAM(graph, starts, goals, verbose=True).solve(time_limit=100000.0)
    if sol:
        soc, ms = calculate_costs(sol, goals)
        print(f"\n  ✓ SOC={soc}  Makespan={ms}")
        for i, p in enumerate(sol): print(f"    Agent {i}: {p}")
        
        os.makedirs('data/logs', exist_ok=True)
        visualize_solution_static(grid, sol, starts, goals, 
                                 save_path='data/logs/two_pairs_solution.png')
        animate_mapf_solution(grid, sol, starts, goals,
                             save_path='data/logs/two_pairs_solution.gif')
    else:
        print("\n  ✗ FAILED")
    return sol is not None


def test_three_agents_corridor():
    print("\n" + "="*70)
    print("TEST: Three Agents Corridor")
    print("="*70)
    grid   = np.array([
        [0,0,0,0,0],
        [1,0,0,1,1],
    ], dtype=int)
    graph  = Graph(grid)
    starts = [(0,0),(0,3),(0,4)]
    goals  = [(0,4),(0,0),(0,2)]
    _print_map(grid, starts, goals)

    sol = GroupedLaCAM(graph, starts, goals, verbose=True).solve(time_limit=30.0)
    if sol:
        soc, ms = calculate_costs(sol, goals)
        print(f"\n  ✓ SOC={soc}  Makespan={ms}")
        for i, p in enumerate(sol): print(f"    Agent {i}: {p}")
        
        os.makedirs('data/logs', exist_ok=True)
        visualize_solution_static(grid, sol, starts, goals, 
                                 save_path='data/logs/three_agents_solution.png')
        animate_mapf_solution(grid, sol, starts, goals,
                             save_path='data/logs/three_agents_solution.gif')
    else:
        print("\n  ✗ FAILED")
    return sol is not None


def test_benchmark(map_path, scen_path, num_agents=10):
    print("\n" + "="*70)
    print(f"TEST: Benchmark ({num_agents} agents)")
    print("="*70)
    grid  = load_map_file(map_path)
    graph = Graph(grid)
    _, starts, goals = load_scenario_file(scen_path, num_agents)
    print(f"    Map {grid.shape[0]}×{grid.shape[1]},  {len(starts)} agents")

    t0  = time.time()
    sol = GroupedLaCAM(graph, starts, goals, verbose=True).solve(time_limit=30.0)
    elapsed = time.time() - t0

    if sol:
        soc, ms = calculate_costs(sol, goals)
        print(f"\n  ✓ SOC={soc}  Makespan={ms}  Time={elapsed:.3f}s")
        
        os.makedirs('data/logs', exist_ok=True)
        visualize_solution_static(grid, sol, starts, goals, 
                                 save_path='data/logs/benchmark_solution.png')
        animate_mapf_solution(grid, sol, starts, goals,
                             save_path='data/logs/benchmark_solution.gif')
    else:
        print(f"\n  ✗ FAILED ({elapsed:.3f}s)")
    return sol is not None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grouped LaCAM")
    parser.add_argument('--map',    help='Map file')
    parser.add_argument('--scen',   help='Scenario file')
    parser.add_argument('--agents', type=int, default=10)
    parser.add_argument('--starts', help='Start positions as "r1,c1;r2,c2;..."')
    parser.add_argument('--goals',  help='Goal positions as "r1,c1;r2,c2;..."')
    parser.add_argument('--test',   choices=['corridor','open','pairs','three','all'],
                        default='all')
    args = parser.parse_args()

    print("\n" + "="*70)
    print("GROUPED LaCAM — Multi-Level Multi-Agent Pathfinding")
    print("="*70)

    if args.starts and args.goals:
        if not args.map:
            print("Error: --map required when using --starts/--goals")
        else:
            grid = load_map_file(args.map)
            graph = Graph(grid)
            
            starts = [tuple(map(int, s.split(','))) for s in args.starts.split(';')]
            goals = [tuple(map(int, g.split(','))) for g in args.goals.split(';')]
            
            print(f"Custom test: {len(starts)} agents")
            _print_map(grid, starts, goals)
            
            sol = GroupedLaCAM(graph, starts, goals, verbose=True).solve(time_limit=30.0)
            if sol:
                soc, ms = calculate_costs(sol, goals)
                print(f"\n  ✓ SOC={soc}  Makespan={ms}")
                for i, p in enumerate(sol): print(f"    Agent {i}: {p}")
                
                os.makedirs('data/logs', exist_ok=True)
                visualize_solution_static(grid, sol, starts, goals, 
                                         save_path='data/logs/custom_solution.png')
                animate_mapf_solution(grid, sol, starts, goals,
                                     save_path='data/logs/custom_solution.gif')
            else:
                print("\n  ✗ FAILED")
    elif args.map and args.scen:
        test_benchmark(args.map, args.scen, args.agents)
    else:
        results = {}
        if args.test in ('corridor','all'): results['corridor'] = test_corridor_swap()
        if args.test in ('open',   'all'):  results['open']     = test_open_swap()
        if args.test in ('pairs',  'all'):  results['pairs']    = test_two_pairs()
        if args.test in ('three',  'all'):  results['three']    = test_three_agents_corridor()

        print("\n" + "="*70)
        print("RESULTS")
        for name, ok in results.items():
            print(f"  {name:12s} {'✓ PASS' if ok else '✗ FAIL'}")
        print("="*70)
