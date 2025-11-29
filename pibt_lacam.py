import numpy as np
from collections import deque
from typing import List, Tuple, Set, Dict, Optional
import time
import sys
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors
import os

Vertex = Tuple[int, int]  # (row, col)
Configuration = Tuple[Vertex, ...]  # tuple of all agent locations
AgentID = int
Path = List[Vertex]

# GRAPH CLASS

class Graph:    
    def __init__(self, grid_map: np.ndarray):
        self.grid_map = grid_map
        self.height, self.width = grid_map.shape
        
        # Store backward distances from goals
        self.goal_distances = {}  # goal -> {position: distance}
    
    def neighbors(self, v: Vertex) -> List[Vertex]:
        row, col = v
        neighbors = []
        
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if (0 <= nr < self.height and 
                0 <= nc < self.width and 
                self.grid_map[nr, nc] == 0):
                neighbors.append((nr, nc))
        
        return neighbors
    
    def get_distance(self, start: Vertex, goal: Vertex) -> int:
        # Get distance using cached backward BFS from goal
        if start == goal:
            return 0
        
        # If we haven't computed distances for this goal yet, do it now
        if goal not in self.goal_distances:
            self._compute_all_distances_from_goal(goal)
        
        return self.goal_distances[goal].get(start, float('inf'))
    
    def _compute_all_distances_from_goal(self, goal: Vertex):
        # Backward BFS from goal to ALL reachable vertices, called once per goal, caches everything
        distances = {goal: 0}
        queue = deque([(goal, 0)])
        
        while queue:  # No early stopping
            current, dist = queue.popleft()
            
            for neighbor in self.neighbors(current):
                if neighbor not in distances:
                    distances[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))
        
        # Cache ALL distances for this goal
        self.goal_distances[goal] = distances


# PIBT IMPLEMENTATION

class PIBT:
    def __init__(self, graph: Graph, starts: List[Vertex], goals: List[Vertex]):
        self.graph = graph
        self.num_agents = len(starts)
        
        # Convert to NumPy arrays (N, 2)
        self.starts = np.array(starts, dtype=np.int32)
        self.goals = np.array(goals, dtype=np.int32)
        
        # No precomputation needed - distances computed lazily on first query
        
        # tie breaker
        self.priorities = np.array([i * 0.001 for i in range(self.num_agents)])
        
        # NumPy arrays for locations (N, 2)
        self.current_locs = self.starts.copy()
        self.next_locs = np.full((self.num_agents, 2), -1, dtype=np.int32)  # -1 = unassigned
        
        # Occupancy maps for fast conflict checking
        self.occupied_current = np.full((graph.height, graph.width), -1, dtype=np.int32)
        self.occupied_next = np.full((graph.height, graph.width), -1, dtype=np.int32)
    
    def _update_occupancy_current(self):
        # Update the occupancy map for current locations
        self.occupied_current.fill(-1)
        for agent_id in range(self.num_agents):
            r, c = self.current_locs[agent_id]
            self.occupied_current[r, c] = agent_id
    
    def _update_occupancy_next(self):
        # Update the occupancy map for next locations
        self.occupied_next.fill(-1)
        for agent_id in range(self.num_agents):
            if self.next_locs[agent_id, 0] != -1:  # if assigned
                r, c = self.next_locs[agent_id]
                self.occupied_next[r, c] = agent_id
    
    def solve(self, max_timesteps: int = 1000) -> Optional[List[List[Vertex]]]:
        solution = [[tuple(loc)] for loc in self.starts]
        
        for timestep in range(max_timesteps):
            # Check if all agents reached goals
            if np.all(np.all(self.current_locs == self.goals, axis=1)):
                return solution
            
            if not self.plan_one_timestep():
                return None  # failed
            
            # move agents
            self.current_locs = self.next_locs.copy()
            for i in range(self.num_agents):
                solution[i].append(tuple(self.current_locs[i]))
        
        return None 
    
    def plan_one_timestep(self) -> bool:
        # update priorities
        for i in range(self.num_agents):
            if not np.array_equal(self.current_locs[i], self.goals[i]):
                self.priorities[i] += 1
            else:
                self.priorities[i] = i * 0.001  # reset with tie-breaker
        
        # sort agents by priority
        agent_order = np.argsort(-self.priorities)
        
        # initialize next loc (N, 2) array with -1
        self.next_locs.fill(-1)
        self.occupied_next.fill(-1)
        self._update_occupancy_current()
        
        # plan for each agent
        for agent_id in agent_order:
            if self.next_locs[agent_id, 0] == -1:  # not assigned yet
                success = self._pibt_recursive(agent_id, None)
                if not success:
                    return False 
        
        return True
    
    def _pibt_recursive(self, agent_id: AgentID, blocked_by: Optional[AgentID]) -> bool:
        current_pos = tuple(self.current_locs[agent_id])
        goal = tuple(self.goals[agent_id])
        
        # Get candidate nodes
        candidates = self.graph.neighbors(current_pos) + [current_pos]
        
        # Sort by distance to goal using precomputed distances
        # Check occupancy using occupancy map
        candidates.sort(key=lambda v: (
            self.graph.get_distance(v, goal),
            self.occupied_current[v] != -1  # True if occupied (moves occupied to end)
        ))
        
        # Try each candidate
        for candidate in candidates:
            # Check vertex conflict using occupancy map
            if self.occupied_next[candidate] != -1:
                continue
            
            # Check swap conflict
            if blocked_by is not None:
                blocked_pos = tuple(self.current_locs[blocked_by])
                if blocked_pos == candidate:
                    continue
            
            # Reserve this location
            self.next_locs[agent_id] = np.array(candidate, dtype=np.int32)
            self.occupied_next[candidate] = agent_id
            
            # Check if another agent occupies this location using occupancy map
            conflicting_agent = self.occupied_current[candidate]
            
            if (conflicting_agent != -1 and 
                conflicting_agent != agent_id and 
                self.next_locs[conflicting_agent, 0] == -1):  # Not yet assigned
                
                # Priority inheritance - recursive call
                if not self._pibt_recursive(conflicting_agent, agent_id):
                    # Failed, unreserve and try next candidate
                    self.occupied_next[candidate] = -1
                    self.next_locs[agent_id] = np.array([-1, -1], dtype=np.int32)
                    continue
            
            # Success
            return True
        
        # No valid move found - stay in place
        self.next_locs[agent_id] = self.current_locs[agent_id].copy()
        self.occupied_next[current_pos] = agent_id
        return False


# LaCAM IMPLEMENTATION

class LaCAM:
    class Constraint:
        # low-level constraint node
        def __init__(self, parent: Optional['LaCAM.Constraint'], 
                     who: Optional[AgentID], where: Optional[Vertex]):
            self.parent = parent
            self.who = who  # which agent
            self.where = where  # must go to which vertex
        
        def depth(self) -> int:
            # Get depth in the tree
            if self.parent is None:
                return 0
            return 1 + self.parent.depth()
        
        def get_constraints(self) -> Dict[AgentID, Vertex]:
            # Extract all constraints from root to this node
            constraints = {}
            current = self
            while current is not None and current.who is not None:
                constraints[current.who] = current.where
                current = current.parent
            return constraints
    
    class HighLevelNode:
        # high-level search node
        def __init__(self, config: Configuration, order: List[AgentID], 
                     parent: Optional['LaCAM.HighLevelNode']):
            self.config = config
            self.order = order  # agent ordering for constraint generation
            self.parent = parent
            self.tree = deque()  # queue of constraints (BFS)
            
            # initialize with root constraint
            self.tree.append(LaCAM.Constraint(parent=None, who=None, where=None))
    
    def __init__(self, graph: Graph, starts: List[Vertex], goals: List[Vertex]):
        self.graph = graph
        self.starts = tuple(starts)
        self.goals = tuple(goals)
        self.num_agents = len(starts)
        
        # no precomputation needed - distances computed lazily on first query
        
        # PIBT-based configuration generator
        self.config_generator = PIBTConfigGenerator(graph, starts, goals)

    
    def solve(self, node_limit: int = 100000, time_limit: float = 30.0) -> Optional[List[List[Vertex]]]:
        start_time = time.time()
        
        # Initialize (Lines 1-3)
        open_list = []  # Stack (DFS)
        explored = {}  # Config -> HighLevelNode
        
        init_order = self._get_initial_order()
        init_node = LaCAM.HighLevelNode(self.starts, init_order, None)
        
        open_list.append(init_node)
        explored[self.starts] = init_node
        
        nodes_generated = 1
        nodes_expanded = 0
        
        # Main loop (Lines 4-19)
        while open_list and nodes_generated < node_limit:
            # Check time limit
            if time.time() - start_time > time_limit:
                print(f"LaCAM timeout. Nodes expanded: {nodes_expanded}, generated: {nodes_generated}")
                return None
            
            # Pop from stack (DFS) - Line 5
            node = open_list.pop()
            
            # Goal test (Line 6)
            if node.config == self.goals:
                print(f"LaCAM success! Depth: {self._get_depth(node)}, "
                      f"Nodes expanded: {nodes_expanded}, generated: {nodes_generated}")
                solution = self._backtrack(node)
                return solution
            
            # Check if node exhausted (Line 7)
            if not node.tree:
                continue
            
            # Get next constraint (Line 8)
            constraint = node.tree.popleft()
            
            # Put node back if it still has constraints to explore
            if node.tree:
                open_list.append(node)
            
            nodes_expanded += 1
            
            # Low-level expansion (Lines 9-13)
            if constraint.depth() < self.num_agents:
                agent_idx = constraint.depth()
                agent_id = node.order[agent_idx]
                current_vertex = node.config[agent_id]
                
                # Generate constraints for all neighbors + stay (Lines 11-13)
                neighbors = self.graph.neighbors(current_vertex) + [current_vertex]
                
                # Randomize order for diversity (mentioned in paper Section 3.3)
                np.random.shuffle(neighbors)
                
                for next_vertex in neighbors:
                    new_constraint = LaCAM.Constraint(
                        parent=constraint,
                        who=agent_id,
                        where=next_vertex
                    )
                    node.tree.append(new_constraint)
            
            # Generate new configuration using PIBT (Line 14)
            new_config = self.config_generator.generate(node.config, constraint)
            
            if new_config is None:
                continue  # Failed to generate valid configuration
            
            # Check if already explored (Line 16)
            if new_config in explored:
                # reinsert existing node to improve solution quality
                existing_node = explored[new_config]
                # if existing_node not in open_list:
                open_list.append(existing_node)
                continue
            
            # Create new high-level node (Line 17)
            new_order = self._get_order(new_config)
            new_node = LaCAM.HighLevelNode(new_config, new_order, node)
            
            # Add to collections (Line 18)
            open_list.append(new_node)
            explored[new_config] = new_node
            nodes_generated += 1
        
        print(f"LaCAM failed. Nodes expanded: {nodes_expanded}, generated: {nodes_generated}")
        return None
    
    def _get_initial_order(self) -> List[AgentID]:
        # get initial agent ordering, order by distance to goal - desc, agents with longer paths should have higher priority
        distances = [self.graph.get_distance(self.starts[i], self.goals[i]) 
                    for i in range(self.num_agents)]
        return list(np.argsort(distances)[::-1])
    
    def _get_order(self, config: Configuration) -> List[AgentID]:
        # priority rule from section 3.3
        not_at_goal = []
        at_goal = []
        
        for i in range(self.num_agents):
            if config[i] == self.goals[i]:
                at_goal.append(i)
            else:
                not_at_goal.append(i)
        
        return not_at_goal + at_goal
    
    def _get_depth(self, node: HighLevelNode) -> int:
        depth = 0
        current = node
        while current.parent is not None:
            depth += 1
            current = current.parent
        return depth
    
    def _backtrack(self, node: HighLevelNode) -> List[List[Vertex]]:
        configs = []
        current = node
        
        while current is not None:
            configs.append(current.config)
            current = current.parent
        
        configs.reverse()
        
        paths = [[] for _ in range(self.num_agents)]
        for config in configs:
            for agent_id in range(self.num_agents):
                paths[agent_id].append(config[agent_id])
        
        return paths


# PIBT-BASED CONFIGURATION GENERATOR FOR LaCAM

class PIBTConfigGenerator:
    
    def __init__(self, graph: Graph, starts: List[Vertex], goals: List[Vertex]):
        self.graph = graph
        self.num_agents = len(goals)
        
        self.goals = np.array(goals, dtype=np.int32)
        
        # priorty values (updated during generation)
        self.priorities = np.zeros(self.num_agents)
        
        self.current_locs = None  # (N, 2) array
        self.next_locs = None  # (N, 2) array
        self.constrained_agents = None
        
        # Occupancy maps
        self.occupied_current = np.full((graph.height, graph.width), -1, dtype=np.int32)
        self.occupied_next = np.full((graph.height, graph.width), -1, dtype=np.int32)
    
    def _update_occupancy_current(self):
        # Update the occupancy map for current locations
        self.occupied_current.fill(-1)
        for agent_id in range(self.num_agents):
            if self.current_locs[agent_id, 0] != -1:
                r, c = self.current_locs[agent_id]
                self.occupied_current[r, c] = agent_id
    
    def _update_occupancy_next(self):
        # update the occupancy map for next locations
        self.occupied_next.fill(-1)
        for agent_id in range(self.num_agents):
            if self.next_locs[agent_id, 0] != -1:
                r, c = self.next_locs[agent_id]
                self.occupied_next[r, c] = agent_id
    
    def generate(self, current_config: Configuration, 
                 constraint: LaCAM.Constraint) -> Optional[Configuration]:
        
        # Extract constraints
        constraints_dict = constraint.get_constraints()
        
        self.current_locs = np.array(current_config, dtype=np.int32)
        self.next_locs = np.full((self.num_agents, 2), -1, dtype=np.int32)
        self.constrained_agents = set(constraints_dict.keys())
        
        # Pre-assign constrained agents
        for agent_id, vertex in constraints_dict.items():
            self.next_locs[agent_id] = np.array(vertex, dtype=np.int32)
        
        # Update occupancy maps
        self._update_occupancy_current()
        self._update_occupancy_next()
        
        # Update priorities for PIBT
        for i in range(self.num_agents):
            if not np.array_equal(self.current_locs[i], self.goals[i]):
                self.priorities[i] = 100.0 + i * 0.001  # High priority
            else:
                self.priorities[i] = i * 0.001  # Low priority
        
        # Get agent order (sorted by priority, descending)
        agent_order = np.argsort(-self.priorities)
        
        # Plan for each unconstrained agent using PIBT
        for agent_id in agent_order:
            if agent_id in self.constrained_agents:
                continue  # Already assigned by constraint
            
            if self.next_locs[agent_id, 0] == -1:  # Not assigned
                success = self._pibt_recursive(agent_id, None)
                if not success:
                    return None  # Failed to generate valid configuration
        
        # Verify all constraints satisfied
        for agent_id, required_vertex in constraints_dict.items():
            if not np.array_equal(self.next_locs[agent_id], np.array(required_vertex)):
                return None  # Constraint violated
        
        # Verify no conflicts (all locations unique)
        next_locs_tuple = [tuple(loc) for loc in self.next_locs]
        if len(set(next_locs_tuple)) != len(next_locs_tuple):
            return None  # Vertex conflict
        
        return tuple(next_locs_tuple)
    
    def _pibt_recursive(self, agent_id: AgentID, blocked_by: Optional[AgentID]) -> bool:
        # if this agent is constrained, it's already assigned
        if agent_id in self.constrained_agents:
            return True
        
        current_pos = tuple(self.current_locs[agent_id])
        goal = tuple(self.goals[agent_id])
        
        # Get candidate nodes
        candidates = self.graph.neighbors(current_pos) + [current_pos]
        
        # OPTIMIZED: Sort by distance to goal using precomputed distances
        candidates.sort(key=lambda v: (
            self.graph.get_distance(v, goal),
            self.occupied_current[v] != -1  # Prefer unoccupied
        ))
        
        # Try each candidate
        for candidate in candidates:
            # OPTIMIZED: Check vertex conflict using occupancy map
            if self.occupied_next[candidate] != -1:
                continue
            
            # Check swap conflict
            if blocked_by is not None:
                blocked_pos = tuple(self.current_locs[blocked_by])
                if blocked_pos == candidate:
                    continue
            
            # Reserve this location
            self.next_locs[agent_id] = np.array(candidate, dtype=np.int32)
            self.occupied_next[candidate] = agent_id
            
            # Check if another agent occupies this location
            conflicting_agent = self.occupied_current[candidate]
            
            if (conflicting_agent != -1 and 
                conflicting_agent != agent_id and 
                self.next_locs[conflicting_agent, 0] == -1 and
                conflicting_agent not in self.constrained_agents):  # Don't move constrained agents!
                
                # Priority inheritance - rec call
                if not self._pibt_recursive(conflicting_agent, agent_id):
                    # Failed, unreserve and try next candidate
                    self.occupied_next[candidate] = -1
                    self.next_locs[agent_id] = np.array([-1, -1], dtype=np.int32)
                    continue
            
            # success!
            return True
        
        # no valid move - stay in place
        self.next_locs[agent_id] = self.current_locs[agent_id].copy()
        self.occupied_next[current_pos] = agent_id
        return False



# UTILITY FUNCTIONS

def create_grid_from_string(grid_str: str) -> np.ndarray:
    lines = [line.strip() for line in grid_str.strip().split('\n') if line.strip()]
    height = len(lines)
    width = len(lines[0])
    
    grid = np.zeros((height, width), dtype=int)
    for i, line in enumerate(lines):
        for j, char in enumerate(line):
            if char == '#' or char == '@':
                grid[i, j] = 1  # Obstacle
    
    return grid

def load_map_file(map_path: str) -> np.ndarray:
    with open(map_path, 'r') as f:
        lines = f.readlines()
    
    # Parse header
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
    
    height = header['height']
    width = header['width']
    
    # Parse map
    grid_map = np.zeros((height, width), dtype=int)
    for i in range(height):
        line = lines[map_start_idx + i].strip()
        for j, char in enumerate(line):
            if char in ['@', 'O', 'T', 'W']:  # Obstacles
                grid_map[i, j] = 1
            # '.' and 'G' are passable (0)
    
    return grid_map


def load_scenario_file(scen_path: str, num_agents: int = None) -> Tuple[str, List[Vertex], List[Vertex]]:
    with open(scen_path, 'r') as f:
        lines = f.readlines()
    
    starts = []
    goals = []
    map_name = None
    
    for line in lines[1:]:  # Skip version line
        if not line.strip():
            continue
        
        parts = line.strip().split()
        if len(parts) < 9:
            continue
        
        # Parse scenario line
        # bucket, map_file, width, height, start_col, start_row, goal_col, goal_row, optimal_length
        if map_name is None:
            map_name = parts[1]
        
        start_col = int(parts[4])
        start_row = int(parts[5])
        goal_col = int(parts[6])
        goal_row = int(parts[7])
        
        starts.append((start_row, start_col))
        goals.append((goal_row, goal_col))
        
        # Limit number of agents if specified
        if num_agents is not None and len(starts) >= num_agents:
            break
    
    return map_name, starts, goals

def visualize_solution(grid_map: np.ndarray, solution: List[List[Vertex]], 
                       starts: List[Vertex], goals: List[Vertex]):
    if solution is None:
        print("no solution to visualize")
        return
    
    num_agents = len(solution)
    timesteps = len(solution[0])
    
    print(f"\nSolution with {timesteps} timesteps:")
    print("=" * 50)
    
    # Show first, middle, and last few timesteps
    show_timesteps = []
    if timesteps <= 10:
        show_timesteps = list(range(timesteps))
    else:
        show_timesteps = [0, 1, 2] + [timesteps // 2] + [timesteps - 3, timesteps - 2, timesteps - 1]
    
    for t in show_timesteps:
        print(f"\nTimestep {t}:")
        grid = []
        for i in range(grid_map.shape[0]):
            row = []
            for j in range(grid_map.shape[1]):
                if grid_map[i, j] == 1:
                    row.append('#')
                else:
                    # Check if any agent is here
                    agent_here = None
                    for agent_id in range(num_agents):
                        if solution[agent_id][t] == (i, j):
                            agent_here = agent_id
                            break
                    
                    if agent_here is not None:
                        row.append(str(agent_here))
                    elif (i, j) in goals:
                        row.append('G')
                    elif (i, j) in starts:
                        row.append('S')
                    else:
                        row.append('.')
            grid.append(''.join(row))
        
        for row in grid:
            print(row)
        
        if t == timesteps // 2 and timesteps > 10:
            print("\n... (middle timesteps) ...\n")

def calculate_costs(solution: List[List[Vertex]], goals: List[Vertex]) -> Tuple[int, int]:
    if solution is None:
        return float('inf'), float('inf')
    
    num_agents = len(solution)
    timesteps = len(solution[0])
    
    # Sum of costs (each agent's path length until reaching goal)
    sum_of_costs = 0
    for agent_id in range(num_agents):
        cost = 0
        for t in range(timesteps):
            if solution[agent_id][t] != goals[agent_id]:
                cost += 1
        sum_of_costs += cost
    
    # makespan (max time any agent takes)
    makespan = timesteps - 1
    
    return sum_of_costs, makespan

def save_solution_to_file(solution: List[List[Vertex]], filepath: str, 
                         map_path: str = None, goals: List[Vertex] = None):
    if solution is None:
        print(f"No solution to save to {filepath}")
        return
    
    num_agents = len(solution)
    
    with open(filepath, 'w') as f:
        # Write map name if provided
        if map_path:
            f.write(f"Map_name: {map_path}\n")
        
        # Write number of agents
        f.write(f"Num_agents: {num_agents}\n")
        
        # Write goals if provided
        if goals:
            for agent_id, goal in enumerate(goals):
                f.write(f"Goal_for_agent {agent_id}: ({goal[0]},{goal[1]})\n")
        
        # Write paths
        for agent_id, path in enumerate(solution):
            # Format: Agent 0: (row,col)->(row,col)->
            path_str = '->'.join([f"({v[0]},{v[1]})" for v in path])
            path_str += '->'  # Add trailing arrow
            f.write(f"Agent {agent_id}: {path_str}\n")
    
    print(f"Solution saved to: {filepath}")

#TESTS
def animate_solution_simple(grid_map: np.ndarray, solution: List[List[Vertex]], 
                            starts: List[Vertex], goals: List[Vertex]):
    if solution is None:
        print("No solution to animate")
        return
    
    num_agents = len(solution)
    timesteps = len(solution[0])
    height, width = grid_map.shape
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Agent colors
    agent_colors = plt.cm.tab10(np.linspace(0, 1, num_agents))
    
    def update(frame):
        ax.clear()
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(-0.5, height - 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_xticks(range(width))
        ax.set_yticks(range(height))
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Timestep {frame}/{timesteps-1}", fontsize=16, fontweight='bold')
        
        # Draw obstacles
        for i in range(height):
            for j in range(width):
                if grid_map[i, j] == 1:
                    ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, 
                                              color='black', alpha=0.7))
        
        for agent_id, (row, col) in enumerate(goals):
            ax.plot(col, row, 'o', markersize=20, 
                   color=agent_colors[agent_id], alpha=0.3)
            ax.text(col, row, 'G', ha='center', va='center', 
                   fontsize=10, color='gray', fontweight='bold')
        
        for agent_id in range(num_agents):
            if frame > 0:
                path = solution[agent_id][:frame+1]
                rows = [p[0] for p in path]
                cols = [p[1] for p in path]
                ax.plot(cols, rows, '-', color=agent_colors[agent_id], 
                       alpha=0.5, linewidth=2)
        
        # Draw agents
        for agent_id in range(num_agents):
            row, col = solution[agent_id][frame]
            ax.plot(col, row, 'o', markersize=30, 
                   color=agent_colors[agent_id], 
                   markeredgecolor='black', markeredgewidth=2)
            ax.text(col, row, str(agent_id), ha='center', va='center', 
                   fontsize=14, color='white', fontweight='bold')
        
        return []
    
    anim = animation.FuncAnimation(fig, update, frames=timesteps, 
                                  interval=500, repeat=True, blit=False)
    
    plt.tight_layout()
    plt.show()
    plt.close()


def test_simple_example(visualize=False):
    print("TEST 1: Simple 10x10 grid with 4 agents")
    
    # Create grid
    grid_map = np.zeros((10, 10), dtype=int)
    grid_map[5, 3:7] = 1  # Horizontal wall
    
    graph = Graph(grid_map)
    
    # 4 agents need to swap corners
    starts = [(1, 1), (8, 8), (1, 8), (8, 1)]
    goals = [(8, 8), (1, 1), (8, 1), (1, 8)]
    
    print("\nGrid map (# = obstacle):")
    for i in range(grid_map.shape[0]):
        row = []
        for j in range(grid_map.shape[1]):
            if grid_map[i, j] == 1:
                row.append('#')
            elif (i, j) in starts:
                row.append('S')
            elif (i, j) in goals:
                row.append('G')
            else:
                row.append('.')
        print(''.join(row))
    
    print(f"\nStarts: {starts}")
    print(f"Goals:  {goals}")
    
    # Test PIBT
    print("\n" + "="*50)
    print("Running PIBT...")
    start_time = time.time()
    pibt = PIBT(graph, starts, goals)
    pibt_solution = pibt.solve(max_timesteps=100)
    pibt_time = time.time() - start_time
    
    if pibt_solution:
        soc, makespan = calculate_costs(pibt_solution, goals)
        print(f"✓ PIBT SUCCESS!")
        print(f"  Time: {pibt_time:.4f}s")
        print(f"  Timesteps: {len(pibt_solution[0])}")
        print(f"  Sum-of-costs: {soc}")
        print(f"  Makespan: {makespan}")
    else:
        print("✗ PIBT FAILED")
    
    # Test LaCAM
    print("\n" + "="*50)
    print("Running LaCAM...")
    start_time = time.time()
    lacam = LaCAM(graph, starts, goals)
    lacam_solution = lacam.solve(node_limit=10000, time_limit=10.0)
    lacam_time = time.time() - start_time
    
    if lacam_solution:
        soc, makespan = calculate_costs(lacam_solution, goals)
        print(f"✓ LaCAM SUCCESS!")
        print(f"  Time: {lacam_time:.4f}s")
        print(f"  Timesteps: {len(lacam_solution[0])}")
        print(f"  Sum-of-costs: {soc}")
        print(f"  Makespan: {makespan}")
        
        # Visualize if requested
        if visualize:
            print("\nOpening visualization window...")
            print("   (Close window to continue)")
            animate_solution_simple(grid_map, lacam_solution, starts, goals)
        
        return (grid_map, lacam_solution, starts, goals)
    else:
        print("✗ LaCAM FAILED")
        return None


def test_complex_example(visualize=False):
    print("\nTEST 2: Complex 15x15 maze with 6 agents")
    
    grid_str = """
    ...............
    .###.###.###...
    .#.#.#.#.#.#...
    .#.#.#.#.#.#...
    .#.....#...#...
    .#####.#####...
    .......#.......
    .###.#####.###.
    .#.#.......#.#.
    .#.#########.#.
    .#...........#.
    .#.#########.#.
    ...#.......#...
    .###.#####.###.
    ...............
    """
    
    grid_map = create_grid_from_string(grid_str)
    graph = Graph(grid_map)
    
    # 6 agents
    starts = [(0, 0), (0, 14), (14, 0), (14, 14), (7, 7), (10, 5)]
    goals = [(14, 14), (14, 0), (0, 14), (0, 0), (4, 11), (10, 9)]
    
    print(f"\nStarts: {starts}")
    print(f"Goals:  {goals}")
    
    # Test LaCAM only
    print("\nRunning LaCAM...")
    start_time = time.time()
    lacam = LaCAM(graph, starts, goals)
    lacam_solution = lacam.solve(node_limit=50000, time_limit=30.0)
    lacam_time = time.time() - start_time
    
    if lacam_solution:
        soc, makespan = calculate_costs(lacam_solution, goals)
        print(f"✓ LaCAM SUCCESS!")
        print(f"  Time: {lacam_time:.4f}s")
        print(f"  Timesteps: {len(lacam_solution[0])}")
        print(f"  Sum-of-costs: {soc}")
        print(f"  Makespan: {makespan}")
        
        # Visualize if requested
        if visualize:
            print("\nOpening visualization window...")
            print("   (Close window to continue)")
            animate_solution_simple(grid_map, lacam_solution, starts, goals)
        
        return (grid_map, lacam_solution, starts, goals)
    else:
        print("✗ LaCAM FAILED")
        return None


def test_dense_scenario(visualize=False):
    print("\nTEST 3: Dense scenario - 8 agents in 8x8 grid")
    
    grid_map = np.zeros((8, 8), dtype=int)
    graph = Graph(grid_map)
    
    # 8 agents around the perimeter
    starts = [(0, 0), (0, 7), (7, 0), (7, 7), (0, 3), (7, 3), (3, 0), (3, 7)]
    goals = [(7, 7), (7, 0), (0, 7), (0, 0), (7, 3), (0, 3), (3, 7), (3, 0)]
    
    print(f"\nStarts: {starts}")
    print(f"Goals:  {goals}")
    
    print("\nRunning LaCAM...")
    start_time = time.time()
    lacam = LaCAM(graph, starts, goals)
    lacam_solution = lacam.solve(node_limit=100000, time_limit=30.0)
    lacam_time = time.time() - start_time
    
    if lacam_solution:
        soc, makespan = calculate_costs(lacam_solution, goals)
        print(f"✓ LaCAM SUCCESS!")
        print(f"  Time: {lacam_time:.4f}s")
        print(f"  Timesteps: {len(lacam_solution[0])}")
        print(f"  Sum-of-costs: {soc}")
        print(f"  Makespan: {makespan}")
        
        # Visualize if requested
        if visualize:
            print("\nOpening visualization window...")
            print("   (Close window to continue)")
            animate_solution_simple(grid_map, lacam_solution, starts, goals)
        
        return (grid_map, lacam_solution, starts, goals)
    else:
        print("✗ LaCAM FAILED")
        return None
    

def test_benchmark_scenario(map_path: str, scen_path: str, num_agents: int = 10, 
                           visualize: bool = False, algorithm: str = 'lacam', save_solution: bool = True):

    os.makedirs('data/logs', exist_ok=True)
    os.makedirs('data/logs/tmpImages', exist_ok=True)

    print(f"\nTEST: Benchmark Scenario")
    print(f"Map: {map_path}")
    print(f"Scenario: {scen_path}")
    print(f"Agents: {num_agents}")
    
    # Load map
    grid_map = load_map_file(map_path)
    graph = Graph(grid_map)
    
    print(f"Grid size: {grid_map.shape[0]}x{grid_map.shape[1]}")
    
    # Load scenario
    map_name, starts, goals = load_scenario_file(scen_path, num_agents)
    
    print(f"Loaded {len(starts)} agents")
    print(f"First agent: start={starts[0]}, goal={goals[0]}")
    
    results = {}
    
    # Test PIBT
    if algorithm in ['pibt', 'both']:
        print("\n" + "="*50)
        print("Running PIBT...")
        start_time = time.time()
        pibt = PIBT(graph, starts, goals)
        pibt_solution = pibt.solve(max_timesteps=500)
        pibt_time = time.time() - start_time
        
        if pibt_solution:
            soc, makespan = calculate_costs(pibt_solution, goals)
            print(f"✓ PIBT SUCCESS!")
            print(f"  Time: {pibt_time:.4f}s")
            print(f"  Timesteps: {len(pibt_solution[0])}")
            print(f"  Sum-of-costs: {soc}")
            print(f"  Makespan: {makespan}")
            results['pibt'] = (pibt_solution, pibt_time, soc, makespan)

            if save_solution:
                save_solution_to_file(pibt_solution, 'data/logs/pibt_solution.txt',
                                    map_path=map_path, goals=goals)
        else:
            print("✗ PIBT FAILED")
            results['pibt'] = None
    
    # Test LaCAM
    if algorithm in ['lacam', 'both']:
        print("\n" + "="*50)
        print("Running LaCAM...")
        start_time = time.time()
        lacam = LaCAM(graph, starts, goals)
        lacam_solution = lacam.solve(node_limit=100000, time_limit=60.0)
        lacam_time = time.time() - start_time
        
        if lacam_solution:
            soc, makespan = calculate_costs(lacam_solution, goals)
            print(f"✓ LaCAM SUCCESS!")
            print(f"  Time: {lacam_time:.4f}s")
            print(f"  Timesteps: {len(lacam_solution[0])}")
            print(f"  Sum-of-costs: {soc}")
            print(f"  Makespan: {makespan}")
            results['lacam'] = (lacam_solution, lacam_time, soc, makespan)


            if save_solution:
                save_solution_to_file(lacam_solution, 'data/logs/lacam_solution.txt',
                                    map_path=map_path, goals=goals)
            
            # Visualize if requested
            if visualize and lacam_solution:
                print("\nOpening visualization window...")
                print("   (Close window to continue)")
                animate_solution_simple(grid_map, lacam_solution, starts, goals)
        else:
            print("✗ LaCAM FAILED")
            results['lacam'] = None
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='LaCAM with PIBT - MAPF (OPTIMIZED Version)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run custom tests
  python lacam_pibt_final.py --mode custom --viz --test 1
  
  # Run benchmark test
  python lacam_pibt_final.py --mode benchmark --map data/mapf-map/maze-32-32-4.map --scen data/mapf-scen-random/maze-32-32-4-random-1.scen --agents 10 --viz
  
  # Run benchmark with both algorithms
  python lacam_pibt_final.py --mode benchmark --map data/mapf-map/empty-32-32.map --scen data/mapf-scen-random/empty-32-32-random-1.scen --agents 20 --algo both
        """
    )
    
    parser.add_argument('--mode', 
                       type=str, 
                       default='custom',
                       choices=['custom', 'benchmark'],
                       help='Test mode: custom or benchmark')
    
    parser.add_argument('--viz', '--visualize', 
                       action='store_true',
                       help='Visualize solution')
    
    parser.add_argument('--test',
                       type=str,
                       default='all',
                       choices=['1', '2', '3', 'all'],
                       help='Which custom test to run (only for custom mode)')
    
    # Benchmark-specific arguments
    parser.add_argument('--map',
                       type=str,
                       help='Path to .map file (benchmark mode)')
    
    parser.add_argument('--scen',
                       type=str,
                       help='Path to .scen file (benchmark mode)')
    
    parser.add_argument('--agents',
                       type=int,
                       default=10,
                       help='Number of agents (benchmark mode)')
    
    parser.add_argument('--algo',
                       type=str,
                       default='lacam',
                       choices=['pibt', 'lacam', 'both'],
                       help='Algorithm to run (benchmark mode)')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("LaCAM with PIBT - FINAL OPTIMIZED VERSION")
    print("Optimizations: Lazy Backward BFS + NumPy Arrays + Occupancy Maps")
    print("="*70)
    
    if args.mode == 'custom':
        # Run custom tests
        viz_test1 = args.test in ['1', 'all']
        viz_test2 = args.test in ['2', 'all']
        viz_test3 = args.test in ['3', 'all']
        
        if args.viz:
            viz_test1 = viz_test2 = viz_test3 = True
        
        if args.test in ['1', 'all']:
            test_simple_example(visualize=viz_test1)
        if args.test in ['2', 'all']:
            test_complex_example(visualize=viz_test2)
        if args.test in ['3', 'all']:
            test_dense_scenario(visualize=viz_test3)
    
    elif args.mode == 'benchmark':
        # Run benchmark test
        if not args.map or not args.scen:
            print("ERROR: --map and --scen are required for benchmark mode")
            sys.exit(1)
        
        test_benchmark_scenario(
            map_path=args.map,
            scen_path=args.scen,
            num_agents=args.agents,
            visualize=args.viz,
            algorithm=args.algo
        )
    
    print("\n" + "="*70)
    print("All tests complete!")
    print("="*70)
