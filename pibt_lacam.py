import numpy as np
from collections import deque
from typing import List, Tuple, Set, Dict, Optional
import time
import sys
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors

Vertex = Tuple[int, int]  #(row, col)
Configuration = Tuple[Vertex, ...]  #tuple of all agent locations
AgentID = int
Path = List[Vertex]

#GRAPH CLASS

class Graph:    
    def __init__(self, grid_map: np.ndarray):
        self.grid_map = grid_map
        self.height, self.width = grid_map.shape
        
        # precompute distance tables for efficiency
        self.distance_cache = {}
    
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
        key = (start, goal)
        if key in self.distance_cache:
            return self.distance_cache[key]
        
        if start == goal:
            return 0
        
        # bfs
        queue = deque([(start, 0)])
        visited = {start}
        
        while queue:
            current, dist = queue.popleft()
            
            for neighbor in self.neighbors(current):
                if neighbor == goal:
                    self.distance_cache[key] = dist + 1
                    return dist + 1
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        
        # no path exists
        return float('inf')
    
    def precompute_distances(self, vertices: List[Vertex]):
        
        #pairwise distances for given vertices
        for i, v1 in enumerate(vertices):
            for v2 in vertices[i:]:
                self.get_distance(v1, v2)
                self.get_distance(v2, v1)


# PIBT IMPLEMENTATION

class PIBT:
    def __init__(self, graph: Graph, starts: List[Vertex], goals: List[Vertex]):
        self.graph = graph
        self.starts = starts
        self.goals = goals
        self.num_agents = len(starts)
        
        # tie breaker
        self.priorities = np.array([i * 0.001 for i in range(self.num_agents)])
        
        self.current_locs = list(starts)
        
        self.next_locs = [None] * self.num_agents
    
    
    def solve(self, max_timesteps: int = 1000) -> Optional[List[List[Vertex]]]:

        solution = [[loc] for loc in self.starts]
        
        for timestep in range(max_timesteps):
            # check if all agents reached goals
            if all(self.current_locs[i] == self.goals[i] for i in range(self.num_agents)):
                return solution
            
            if not self.plan_one_timestep():
                return None  # failed
            
            # move agents
            self.current_locs = self.next_locs[:]
            for i in range(self.num_agents):
                solution[i].append(self.current_locs[i])
        
        return None 
    
    def plan_one_timestep(self) -> bool:
        
        # update priorities (line3)
        for i in range(self.num_agents):
            if self.current_locs[i] != self.goals[i]:
                self.priorities[i] += 1
            else:
                self.priorities[i] = i * 0.001  # reset with tie-breaker
        
        # sort agents by priority (line4)
        agent_order = np.argsort(-self.priorities)
        
        # initialize next loc
        self.next_locs = [None] * self.num_agents
        
        # plan for each agent (lines 5-7)
        for agent_id in agent_order:
            if self.next_locs[agent_id] is None:
                success = self._pibt_recursive(agent_id, None)
                if not success:
                    return False 
        
        return True
    
    def _pibt_recursive(self, agent_id: AgentID, blocked_by: Optional[AgentID]) -> bool:
        
        current_pos = self.current_locs[agent_id]
        goal = self.goals[agent_id]
        
        # get candidate nodes (line9)
        candidates = self.graph.neighbors(current_pos) + [current_pos]
        
        # sort by distance to goal (line10)
        # prefr unoccupied vertices to avoid unnecessary PI

        candidates.sort(key=lambda v: (
            self.graph.get_distance(v, goal),
            any(self.current_locs[k] == v for k in range(self.num_agents))
        ))
        
        #try each candidate (lines 11-19)
        for candidate in candidates:
            # check vertex conflict (line12)
            if any(self.next_locs[k] == candidate for k in range(self.num_agents)):
                continue
            
            # check swap conflict (line13)
            if blocked_by is not None and self.current_locs[blocked_by] == candidate:
                continue
            
            # Reserve this location (Line 14)
            self.next_locs[agent_id] = candidate
            
            # Check if another agent occupies this location (Line 15)
            conflicting_agent = None
            for k in range(self.num_agents):
                if (k != agent_id and 
                    self.current_locs[k] == candidate and 
                    self.next_locs[k] is None):
                    conflicting_agent = k
                    break
            
            if conflicting_agent is not None:
                # Priority inheritance (Line 16)
                if not self._pibt_recursive(conflicting_agent, agent_id):
                    continue  # Failed, try next candidate
            
            # Success! (Line 18)
            return True
        
        # No valid move found (Lines 20-21)
        self.next_locs[agent_id] = current_pos
        return False


# LaCAM IMPLEMENTATION


class LaCAM:
    """
    Lazy Constraints Addition Search for MAPF
    Based on Algorithm 1 from the LaCAM paper
    
    KEY: Uses full PIBT (with priority inheritance and backtracking) 
         for configuration generation
    """
    
    class Constraint:
        """Low-level constraint node"""
        def __init__(self, parent: Optional['LaCAM.Constraint'], 
                     who: Optional[AgentID], where: Optional[Vertex]):
            self.parent = parent
            self.who = who  # Which agent
            self.where = where  # Must go to which vertex
        
        def depth(self) -> int:
            """Get depth in constraint tree"""
            if self.parent is None:
                return 0
            return 1 + self.parent.depth()
        
        def get_constraints(self) -> Dict[AgentID, Vertex]:
            """Extract all constraints from root to this node"""
            constraints = {}
            current = self
            while current is not None and current.who is not None:
                constraints[current.who] = current.where
                current = current.parent
            return constraints
    
    class HighLevelNode:
        """High-level search node"""
        def __init__(self, config: Configuration, order: List[AgentID], 
                     parent: Optional['LaCAM.HighLevelNode']):
            self.config = config
            self.order = order  # Agent ordering for constraint generation
            self.parent = parent
            self.tree = deque()  # Queue of constraints (BFS)
            
            # Initialize with root constraint
            self.tree.append(LaCAM.Constraint(parent=None, who=None, where=None))
    
    def __init__(self, graph: Graph, starts: List[Vertex], goals: List[Vertex]):
        self.graph = graph
        self.starts = tuple(starts)
        self.goals = tuple(goals)
        self.num_agents = len(starts)
        
        # PIBT-based configuration generator
        self.config_generator = PIBTConfigGenerator(graph, starts, goals)
    
    def solve(self, node_limit: int = 100000, time_limit: float = 30.0) -> Optional[List[List[Vertex]]]:
        """
        Solve MAPF using LaCAM
        
        Args:
            node_limit: Maximum number of high-level nodes to generate
            time_limit: Maximum time in seconds
        
        Returns:
            List of paths for each agent, or None if failed
        """
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
                return self._backtrack(node)
            
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
                # Optional: Reinsert existing node to improve solution quality
                existing_node = explored[new_config]
                if existing_node not in open_list:
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
        self.goals = goals
        self.num_agents = len(goals)
        
        # Priority values (updated during generation)
        self.priorities = np.zeros(self.num_agents)
        
        # Working memory for configuration generation
        self.current_locs = None
        self.next_locs = None
        self.constrained_agents = None
    
    def generate(self, current_config: Configuration, 
                 constraint: LaCAM.Constraint) -> Optional[Configuration]:
        
        # Extract constraints
        constraints_dict = constraint.get_constraints()
        
        # Initialize state
        self.current_locs = list(current_config)
        self.next_locs = [None] * self.num_agents
        self.constrained_agents = set(constraints_dict.keys())
        
        # Pre-assign constrained agents
        for agent_id, vertex in constraints_dict.items():
            self.next_locs[agent_id] = vertex
        
        # Update priorities for PIBT
        # Agents not at goal should have higher priority
        for i in range(self.num_agents):
            if self.current_locs[i] != self.goals[i]:
                self.priorities[i] = 100.0 + i * 0.001  # High priority
            else:
                self.priorities[i] = i * 0.001  # Low priority
        
        # Get agent order (sorted by priority, descending)
        agent_order = np.argsort(-self.priorities)
        
        # Plan for each unconstrained agent using PIBT
        for agent_id in agent_order:
            if agent_id in self.constrained_agents:
                continue  # Already assigned by constraint
            
            if self.next_locs[agent_id] is None:
                # THIS IS THE KEY: Call full PIBT recursive procedure
                success = self._pibt_recursive(agent_id, None)
                if not success:
                    return None  # Failed to generate valid configuration
        
        # Verify all constraints satisfied
        for agent_id, required_vertex in constraints_dict.items():
            if self.next_locs[agent_id] != required_vertex:
                return None  # Constraint violated
        
        # Verify no conflicts
        if len(set(self.next_locs)) != len(self.next_locs):
            return None  # Vertex conflict
        
        return tuple(self.next_locs)
    
    def _pibt_recursive(self, agent_id: AgentID, blocked_by: Optional[AgentID]) -> bool:
        # if this agent is constrained, it's already assigned
        if agent_id in self.constrained_agents:
            return True
        
        current_pos = self.current_locs[agent_id]
        goal = self.goals[agent_id]
        
        # Get candidate nodes (line9)
        candidates = self.graph.neighbors(current_pos) + [current_pos]
        
        # Sort by distance to goal (line10)
        # Tie-break: prefer unoccupied to avoid unnecessary priority inheritance
        candidates.sort(key=lambda v: (
            self.graph.get_distance(v, goal),
            any(self.current_locs[k] == v for k in range(self.num_agents))
        ))
        
        # Try each candidate (lines 11-19)
        for candidate in candidates:
            # Check vertex conflict (line12)
            if any(self.next_locs[k] == candidate for k in range(self.num_agents)):
                continue
            
            # Check swap conflict (line13)
            if blocked_by is not None and self.current_locs[blocked_by] == candidate:
                continue
            
            # Reserve this location (line14)
            self.next_locs[agent_id] = candidate
            
            # Check if another agent occupies this location (line15)
            conflicting_agent = None
            for k in range(self.num_agents):
                if (k != agent_id and 
                    self.current_locs[k] == candidate and 
                    self.next_locs[k] is None and
                    k not in self.constrained_agents):  # Don't try to move constrained agents!
                    conflicting_agent = k
                    break
            
            if conflicting_agent is not None:
                # Priority inheritance - RECURSIVE CALL (line16)
                if not self._pibt_recursive(conflicting_agent, agent_id):
                    continue  #try next candidate
            
            # success! (line18)
            return True
        
        # no valid (lines 20-21)
        self.next_locs[agent_id] = current_pos
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
    
    # Precompute distances
    all_vertices = starts + goals
    graph.precompute_distances(all_vertices)
    
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
    print("Running PIBT...")
    pibt = PIBT(graph, starts, goals)
    pibt_solution = pibt.solve(max_timesteps=100)
    
    if pibt_solution:
        soc, makespan = calculate_costs(pibt_solution, goals)
        print(f"✓ PIBT SUCCESS!")
        print(f"  Timesteps: {len(pibt_solution[0])}")
        print(f"  Sum-of-costs: {soc}")
        print(f"  Makespan: {makespan}")
    else:
        print("✗ PIBT FAILED")
    
    # Test LaCAM
    print("Running LaCAM...")
    lacam = LaCAM(graph, starts, goals)
    lacam_solution = lacam.solve(node_limit=10000, time_limit=10.0)
    
    if lacam_solution:
        soc, makespan = calculate_costs(lacam_solution, goals)
        print(f"LaCAM SUCCESS!")
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
    print("TEST 2: Complex 15x15 maze with 6 agents")
    
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
    
    # Test LaCAM only (PIBT might struggle with this)
    print("Running LaCAM...")
    lacam = LaCAM(graph, starts, goals)
    lacam_solution = lacam.solve(node_limit=50000, time_limit=30.0)
    
    if lacam_solution:
        soc, makespan = calculate_costs(lacam_solution, goals)
        print(f" LaCAM SUCCESS!")
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
    print("TEST 3: Dense scenario - 8 agents in 8x8 grid")
    
    grid_map = np.zeros((8, 8), dtype=int)
    graph = Graph(grid_map)
    
    # 8 agents around the perimeter
    starts = [(0, 0), (0, 7), (7, 0), (7, 7), (0, 3), (7, 3), (3, 0), (3, 7)]
    goals = [(7, 7), (7, 0), (0, 7), (0, 0), (7, 3), (0, 3), (3, 7), (3, 0)]
    
    print(f"\nStarts: {starts}")
    print(f"Goals:  {goals}")
    
    print("Running LaCAM...")
    lacam = LaCAM(graph, starts, goals)
    lacam_solution = lacam.solve(node_limit=100000, time_limit=30.0)
    
    if lacam_solution:
        soc, makespan = calculate_costs(lacam_solution, goals)
        print(f"LaCAM SUCCESS!")
        print(f"Timesteps: {len(lacam_solution[0])}")
        print(f"Sum-of-costs: {soc}")
        print(f"Makespan: {makespan}")
        
        # Visualize if requested
        if visualize:
            print("\nOpening visualization window...")
            print(" (Close window to continue)")
            animate_solution_simple(grid_map, lacam_solution, starts, goals)
        
        return (grid_map, lacam_solution, starts, goals)
    else:
        print("LaCAM FAILED")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='LaCAM with PIBT - MAPF lesgo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests (no visualization)
  python lacam_pibt.py
  
  # Run and visualize test 1
  python lacam_pibt.py --viz 1
  
  # Run and visualize test 2
  python lacam_pibt.py --viz 2
  
  # Run and visualize test 3
  python lacam_pibt.py --viz 3
  
  # Run and visualize all tests
  python lacam_pibt.py --viz all
        """
    )
    
    parser.add_argument('--viz', '--visualize', 
                       type=str, 
                       default=None,
                       choices=['1', '2', '3', 'all'],
                       help='Visualize test: 1, 2, 3, or all')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("LaCAM with PIBT lesgo")
    print("="*70)
    
    viz_test1 = args.viz in ['1', 'all']
    viz_test2 = args.viz in ['2', 'all']
    viz_test3 = args.viz in ['3', 'all']
    
    test_simple_example(visualize=viz_test1)
    test_complex_example(visualize=viz_test2)
    test_dense_scenario(visualize=viz_test3)
    
    print("All tests complete!")
