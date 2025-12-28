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
import matplotlib.patches as patches
import matplotlib.cm as cm


Vertex = Tuple[int, int]
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
            if (0 <= nr < self.height and 0 <= nc < self.width and self.grid_map[nr, nc] == 0):
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
        # Backward BFS from goal to ALL reachable vertices
        distances = {goal: 0}
        queue = deque([(goal, 0)])
        
        while queue:
            current, dist = queue.popleft()
            
            for neighbor in self.neighbors(current):
                if neighbor not in distances:
                    distances[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))
        
        # Cache ALL distances for this goal
        self.goal_distances[goal] = distances


# PIBT CLASS
class PIBT:
    def __init__(self, graph: Graph, starts: List[Vertex], goals: List[Vertex]):
        self.graph = graph
        self.num_agents = len(starts)
        
        self.starts = np.array(starts, dtype=np.int32)
        self.goals = np.array(goals, dtype=np.int32)
                
        self.priorities = np.array([i * 0.001 for i in range(self.num_agents)])
        
        self.current_locs = self.starts.copy()
        self.next_locs = np.full((self.num_agents, 2), -1, dtype=np.int32)  # -1 = unassigned
        
        # Occupancy maps for fast conflict checking
        self.occupied_current = np.full((graph.height, graph.width), -1, dtype=np.int32)
        self.occupied_next = np.full((graph.height, graph.width), -1, dtype=np.int32)
    
    def _update_occupancy_current(self):
        self.occupied_current.fill(-1)
        for agent_id in range(self.num_agents):
            r, c = self.current_locs[agent_id]
            self.occupied_current[r, c] = agent_id
    
    def _update_occupancy_next(self):
        self.occupied_next.fill(-1)
        for agent_id in range(self.num_agents):
            if self.next_locs[agent_id, 0] != -1:  # if assigned
                r, c = self.next_locs[agent_id]
                self.occupied_next[r, c] = agent_id
    
    def solve(self, max_timesteps: int = 1000) -> Optional[List[List[Vertex]]]:
        solution = [[tuple(loc)] for loc in self.starts]
        
        for timestep in range(max_timesteps):
            if np.all(np.all(self.current_locs == self.goals, axis=1)):
                return solution
            
            if not self.plan_one_timestep():
                return None  # failed
            
            # apply moves to agents
            self.current_locs = self.next_locs.copy()
            for i in range(self.num_agents):
                solution[i].append(tuple(self.current_locs[i]))
        
        return None 
    
    def plan_one_timestep(self) -> bool:
        for i in range(self.num_agents):
            if not np.array_equal(self.current_locs[i], self.goals[i]):
                self.priorities[i] += 1
            else:
                self.priorities[i] = i * 0.001
        
        agent_order = np.argsort(-self.priorities)
        
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
        
        candidates = self.graph.neighbors(current_pos) + [current_pos]
        
        # Sort by distance to goal using precomputed distances, check occupancy using occupancy map
        candidates.sort(key=lambda v: (
            self.graph.get_distance(v, goal),
            self.occupied_current[v] != -1  # True if occupied (moves occupied to end)
        ))
        
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
            
            return True
        
        # No valid move found - stay in place
        self.next_locs[agent_id] = self.current_locs[agent_id].copy()
        self.occupied_next[current_pos] = agent_id
        return False
    

    def detect_groups(self) -> Dict[int, List[int]]:
        from collections import defaultdict
        
        # Build conflict graph
        conflict_graph = defaultdict(set)
        
        for i in range(self.num_agents):
            # Check if agent i wanted to move but couldn't
            desired_next = self._get_desired_position(i)
            actual_next = tuple(self.next_locs[i])
            
            if desired_next != actual_next:
                # Agent i is stuck - check what blocked it
                dr, dc = desired_next
                
                # Check if blocked by vertex conflict
                blocker = self.occupied_next[dr, dc]
                if blocker != -1 and blocker != i:
                    conflict_graph[i].add(blocker)
                    conflict_graph[blocker].add(i)
                
                # Check if blocked by another agent at current position
                at_desired = self.occupied_current[dr, dc]
                if at_desired != -1 and at_desired != i:
                    conflict_graph[i].add(at_desired)
                    conflict_graph[at_desired].add(i)
        
        # Find connected components (groups) using DFS
        visited = set()
        groups = {}
        group_id = 0
        
        for agent in range(self.num_agents):
            if agent not in visited:
                # New group - do DFS
                group = []
                stack = [agent]
                
                while stack:
                    curr = stack.pop()
                    if curr in visited:
                        continue
                    
                    visited.add(curr)
                    group.append(curr)
                    
                    # Add neighbors in conflict graph
                    for neighbor in conflict_graph[curr]:
                        if neighbor not in visited:
                            stack.append(neighbor)
                
                groups[group_id] = group
                group_id += 1
        
        return groups

    def _get_desired_position(self, agent_id: int) -> Vertex:
        # Get the position agent wanted to move to (before conflicts).
        # This is the position with minimum distance to goal.
        current_pos = tuple(self.current_locs[agent_id])
        goal = tuple(self.goals[agent_id])
        
        candidates = self.graph.neighbors(current_pos) + [current_pos]
        
        # Choose candidate with minimum distance to goal
        best = min(candidates, 
                key=lambda v: self.graph.get_distance(v, goal))
        
        return best

    def visualize_with_groups(self, title="PIBT Agent Groups"):
        # Visualize current configuration with groups colored.
        # Each group gets a unique color.
        
        # Detect groups
        groups = self.detect_groups()
        
        # Generate colors
        num_groups = len(groups)
        if num_groups == 0:
            return
        
        colors_map = cm.tab10(np.linspace(0, 1, max(10, num_groups)))
        
        # Map agents to colors
        agent_colors = {}
        for group_id, agents in groups.items():
            for agent in agents:
                agent_colors[agent] = colors_map[group_id % len(colors_map)]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Draw grid
        for r in range(self.graph.height):
            for c in range(self.graph.width):
                if self.graph.grid_map[r, c] == 1:
                    # Obstacle
                    rect = patches.Rectangle((c, r), 1, 1,
                                            facecolor='black', edgecolor='gray')
                else:
                    # Empty cell
                    rect = patches.Rectangle((c, r), 1, 1,
                                            facecolor='white', edgecolor='gray',
                                            linewidth=0.5)
                ax.add_patch(rect)
        
        # Draw goals
        for agent_id in range(self.num_agents):
            gr, gc = self.goals[agent_id]
            color = agent_colors.get(agent_id, [0.5, 0.5, 0.5, 0.3])
            ax.add_patch(patches.Circle((gc + 0.5, gr + 0.5), 0.3,
                                    facecolor=color, alpha=0.3,
                                    edgecolor='none'))
            ax.text(gc + 0.5, gr + 0.5, 'G',
                ha='center', va='center',
                fontsize=8, color='gray', fontweight='bold')
        
        # Draw agents with group colors
        for agent_id in range(self.num_agents):
            r, c = self.current_locs[agent_id]
            color = agent_colors[agent_id]
            
            # Draw circle for agent
            circle = patches.Circle((c + 0.5, r + 0.5), 0.4,
                                facecolor=color, edgecolor='black',
                                linewidth=2)
            ax.add_patch(circle)
            
            # Draw agent ID
            ax.text(c + 0.5, r + 0.5, str(agent_id),
                ha='center', va='center',
                fontsize=12, color='white', fontweight='bold')
        
        # Add legend showing groups
        legend_text = []
        for group_id, agents in groups.items():
            if len(agents) > 1:  # Only show groups with multiple agents
                legend_text.append(f"Group {group_id}: {agents}")
        
        if legend_text:
            ax.text(0.02, 0.98, '\n'.join(legend_text),
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=10, family='monospace')
        
        ax.set_xlim(0, self.graph.width)
        ax.set_ylim(0, self.graph.height)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

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
    
    for line in lines[1:]: 
        if not line.strip():
            continue
        
        parts = line.strip().split()
        if len(parts) < 9:
            continue
        
        if map_name is None:
            map_name = parts[1]
        
        start_col = int(parts[4])
        start_row = int(parts[5])
        goal_col = int(parts[6])
        goal_row = int(parts[7])
        
        starts.append((start_row, start_col))
        goals.append((goal_row, goal_col))
        
        if num_agents is not None and len(starts) >= num_agents:
            break
    
    return map_name, starts, goals


def animate_solution_with_groups(grid_map: np.ndarray, pibt: PIBT, 
                                 solution: List[List[Vertex]], 
                                 starts: List[Vertex], goals: List[Vertex],
                                 save_path: str = None):
    
    if solution is None:
        print("No solution to animate")
        return
    
    num_agents = len(solution)
    timesteps = len(solution[0])
    height, width = grid_map.shape
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    group_colors = plt.cm.Set3(np.linspace(0, 1, 12))
    
    def update(frame):
        ax.clear()
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(-0.5, height - 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_xticks(range(width))
        ax.set_yticks(range(height))
        ax.grid(True, alpha=0.3)
        
        pibt.current_locs = np.array([solution[i][frame] for i in range(num_agents)], dtype=np.int32)
        if frame < timesteps - 1:
            pibt.next_locs = np.array([solution[i][frame+1] for i in range(num_agents)], dtype=np.int32)
            pibt._update_occupancy_current()
            pibt._update_occupancy_next()
            
            # Detect groups at this timestep
            groups = pibt.detect_groups()
        else:
            groups = {i: [i] for i in range(num_agents)}  # All separate at end
        
        # Map agents to colors based on groups
        agent_colors = {}
        for group_id, agents in groups.items():
            for agent in agents:
                agent_colors[agent] = group_colors[group_id % len(group_colors)]
        
        num_groups = len([g for g in groups.values() if len(g) > 1])
        ax.set_title(f"Timestep {frame}/{timesteps-1} - {num_groups} Coupled Groups", 
                    fontsize=16, fontweight='bold')
        
        for i in range(height):
            for j in range(width):
                if grid_map[i, j] == 1:
                    ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, 
                                              color='black', alpha=0.7))
        
        for agent_id, (row, col) in enumerate(goals):
            color = agent_colors.get(agent_id, [0.5, 0.5, 0.5, 0.3])
            ax.plot(col, row, 'o', markersize=20, color=color, alpha=0.3)
            ax.text(col, row, 'G', ha='center', va='center', 
                   fontsize=10, color='gray', fontweight='bold')
        
        for agent_id in range(num_agents):
            if frame > 0:
                path = solution[agent_id][:frame+1]
                rows = [p[0] for p in path]
                cols = [p[1] for p in path]
                color = agent_colors[agent_id]
                ax.plot(cols, rows, '-', color=color, alpha=0.5, linewidth=2)
        
        for agent_id in range(num_agents):
            row, col = solution[agent_id][frame]
            color = agent_colors[agent_id]
            ax.plot(col, row, 'o', markersize=30, color=color, 
                   markeredgecolor='black', markeredgewidth=2)
            ax.text(col, row, str(agent_id), ha='center', va='center', 
                   fontsize=14, color='white', fontweight='bold')
        
        legend_text = []
        for group_id, agents in groups.items():
            if len(agents) > 1:
                legend_text.append(f"Group {group_id}: {agents}")
        
        if legend_text:
            ax.text(0.02, 0.98, '\n'.join(legend_text),
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                   fontsize=10, family='monospace')
        
        return []
    
    anim = animation.FuncAnimation(fig, update, frames=timesteps, 
                                  interval=500, repeat=True, blit=False)
    
    if save_path:
        print(f"Saving animation to {save_path}...")
        anim.save(save_path, writer='pillow', fps=2)
        print(f"Animation saved!")
    
    plt.tight_layout()
    plt.show()
    plt.close()

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
    
    makespan = timesteps - 1
    
    return sum_of_costs, makespan

def save_solution_to_file(solution: List[List[Vertex]], filepath: str, 
                         map_path: str = None, goals: List[Vertex] = None):
    if solution is None:
        print(f"No solution to save to {filepath}")
        return
    
    num_agents = len(solution)
    
    with open(filepath, 'w') as f:
        if map_path:
            f.write(f"Map_name: {map_path}\n")
        
        f.write(f"Num_agents: {num_agents}\n")
        
        if goals:
            for agent_id, goal in enumerate(goals):
                f.write(f"Goal_for_agent {agent_id}: ({goal[0]},{goal[1]})\n")
        
        for agent_id, path in enumerate(solution):
            # Format: Agent 0: (row,col)->(row,col)->
            path_str = '->'.join([f"({v[0]},{v[1]})" for v in path])
            path_str += '->'  
            f.write(f"Agent {agent_id}: {path_str}\n")
    
    print(f"Solution saved to: {filepath}")

def test_pibt_groups(num_agents=4, visualize=True):
    print("\nTEST: PIBT Group Detection")
    print("="*70)
    
    grid_map = np.zeros((5, 5), dtype=int)
    graph = Graph(grid_map)
    
    # Two pairs of agents that will swap (should form 2 groups)
    starts = [
        (0, 0), (0, 4),  # Pair 1: swap horizontal
        (4, 0), (4, 4),  # Pair 2: swap horizontal
    ]
    goals = [
        (0, 4), (0, 0),  # Pair 1 goals
        (4, 4), (4, 0),  # Pair 2 goals
    ]
    
    print(f"\nAgents: {num_agents}")
    print(f"Starts: {starts}")
    print(f"Goals:  {goals}")
    
    pibt = PIBT(graph, starts, goals)
    
    print("\nPlanning first timestep...")
    pibt.plan_one_timestep()
    
    # Detect groups
    groups = pibt.detect_groups()
    
    print(f"\nDetected {len(groups)} groups:")
    for group_id, agents in groups.items():
        if len(agents) > 1:
            print(f"  Group {group_id}: Agents {agents} (coupled)")
        else:
            print(f"  Group {group_id}: Agent {agents} (independent)")
    
    # Visualize if requested
    if visualize:
        print("\nOpening visualization...")
        pibt.visualize_with_groups(title="PIBT Groups After 1 Timestep")
    
    return groups

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
            print(f"PIBT SUCCESS!")
            print(f"  Time: {pibt_time:.4f}s")
            print(f"  Timesteps: {len(pibt_solution[0])}")
            print(f"  Sum-of-costs: {soc}")
            print(f"  Makespan: {makespan}")
            results['pibt'] = (pibt_solution, pibt_time, soc, makespan)

            if save_solution:
                save_solution_to_file(pibt_solution, 'data/logs/pibt_solution.txt',
                                    map_path=map_path, goals=goals)
        else:
            print("PIBT FAILED")
            results['pibt'] = None
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='PIBT - Priority Inheritance with Backtracking for Multi-Agent Pathfinding',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run default test with static group viz
  python pibt_solver.py --groups
  
  # Run benchmark with group-colored GIF
  python pibt_solver.py --map data/mapf-map/maze-32-32-4.map \
      --scen data/mapf-scen-random/maze-32-32-4-random-1.scen \
      --agents 5 --gif
        """
    )
    parser.add_argument('--map', help='Path to map file')
    parser.add_argument('--scen', help='Path to scenario file')
    parser.add_argument('--agents', type=int, default=10, help='Number of agents')
    parser.add_argument('--gif', action='store_true', help='Save GIF animation with groups')
    parser.add_argument('--groups', action='store_true', help='Show static group visualization')
    
    args = parser.parse_args()
    
    os.makedirs('data/logs', exist_ok=True)
    
    print("\n" + "="*70)
    print("PIBT - Multi-Agent Pathfinding")
    print("="*70)
    
    if args.map and args.scen:
        # Benchmark mode
        grid_map = load_map_file(args.map)
        graph = Graph(grid_map)
        map_name, starts, goals = load_scenario_file(args.scen, args.agents)
        
        print(f"\nRunning PIBT on {args.map}")
        print(f"Agents: {len(starts)}")
        
        pibt = PIBT(graph, starts, goals)
        solution = pibt.solve(max_timesteps=1000)
        
        if solution:
            soc, makespan = calculate_costs(solution, goals)
            print(f"\nSUCCESS!")
            print(f"  Timesteps: {len(solution[0])}")
            print(f"  Sum-of-costs: {soc}")
            print(f"  Makespan: {makespan}")
            
            if args.gif:
                print("\nCreating grouped animation...")
                animate_solution_with_groups(
                    grid_map, pibt, solution, starts, goals,
                    save_path='data/logs/pibt_groups.gif'
                )
        else:
            print("\nFAILED")
    else:
        # Default test
        test_pibt_groups(num_agents=4, visualize=args.groups)
    
    print("\n" + "="*70)
    print("Complete!")
    print("="*70)
