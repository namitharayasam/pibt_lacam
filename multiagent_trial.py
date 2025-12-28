import numpy as np
from collections import deque, defaultdict
from typing import List, Tuple, Set, Dict, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import argparse
import os

Vertex = Tuple[int, int]

class Graph:    
    def __init__(self, grid_map: np.ndarray):
        self.grid_map = grid_map
        self.height, self.width = grid_map.shape
        self.goal_distances = {}
    
    def neighbors(self, v: Vertex) -> List[Vertex]:
        row, col = v
        neighbors = []
        
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if (0 <= nr < self.height and 0 <= nc < self.width and 
                self.grid_map[nr, nc] == 0):
                neighbors.append((nr, nc))
        
        return neighbors
    
    def get_distance(self, start: Vertex, goal: Vertex) -> int:
        if start == goal:
            return 0
        
        if goal not in self.goal_distances:
            self._compute_all_distances_from_goal(goal)
        
        return self.goal_distances[goal].get(start, float('inf'))
    
    def _compute_all_distances_from_goal(self, goal: Vertex):
        # Backward BFS from goal to all reachable vertices (not manhattam)
        distances = {goal: 0}
        queue = deque([(goal, 0)])
        
        while queue:
            current, dist = queue.popleft()
            
            for neighbor in self.neighbors(current):
                if neighbor not in distances:
                    distances[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))
        
        self.goal_distances[goal] = distances


class MultiAgentRealtimeSearch:
    def __init__(self, graph: Graph, starts: List[Vertex], goals: List[Vertex], 
                 verbose: bool = True):
        self.graph = graph
        self.num_agents = len(starts)
        self.starts = np.array(starts, dtype=np.int32)
        self.goals = np.array(goals, dtype=np.int32)
        self.verbose = verbose
        
        # Group penalties: (agent_tuple, config_tuple) -> penalty_value
        # Ex: ((0, 1), ((1,2), (3,4))) -> 5.0, means agents 0,1 at positions (1,2),(3,4) has penalty 5.0
        self.group_penalties = {}
        
        self.current_locs = self.starts.copy()
        self.next_locs = np.full((self.num_agents, 2), -1, dtype=np.int32)
        
        self.priorities = np.array([i * 0.001 for i in range(self.num_agents)])
        
        self.occupied_current = np.full((graph.height, graph.width), -1, dtype=np.int32)
        self.occupied_next = np.full((graph.height, graph.width), -1, dtype=np.int32)
        
        # For tracking conflicts/coupling
        self.desired_locations = {}  # agent_id -> desired_location
        self.actual_locations = {}   # agent_id -> actual_location
        
        self.timestep = 0
    
    def solve(self, max_timesteps: int = 1000) -> Optional[List[List[Vertex]]]:
        if self.verbose:
            print(f"Multi-Agent Real-Time Search with WinC-MAPF Group Penalties (PIBT)")
            print(f"Agents: {self.num_agents}")
            print(f"Starts: {[tuple(s) for s in self.starts]}")
            print(f"Goals:  {[tuple(g) for g in self.goals]}\n")
        
        solution = [[tuple(loc)] for loc in self.starts]
        
        for t in range(max_timesteps):
            self.timestep = t
            
            if self.verbose:
                locs = [(int(loc[0]), int(loc[1])) for loc in self.current_locs]
                print(f"\n{'='*70}")
                print(f"t={t}: Locations: {locs}")
                print(f"       Total group penalties stored: {len(self.group_penalties)}")
            
            all_at_goal = True
            for i in range(self.num_agents):
                if not np.array_equal(self.current_locs[i], self.goals[i]):
                    all_at_goal = False
                    break
            
            if all_at_goal:
                if self.verbose:
                    print(f"\n{'='*70}")
                    print(f"All agents reached goals at timestep {t}")
                return solution
            
            if not self.plan_one_timestep():
                if self.verbose:
                    print(f"\nFailed to plan at timestep {t}")
                return None
            
            # Detect coupled agent groups and update penalties
            self._update_group_penalties()
            
            # Execute moves
            self.current_locs = self.next_locs.copy()
            for i in range(self.num_agents):
                solution[i].append(tuple(self.current_locs[i]))
        
        if self.verbose:
            print(f"\nTimeout after {max_timesteps} timesteps")
        return None
    
    def plan_one_timestep(self) -> bool:
        for i in range(self.num_agents):
            if not np.array_equal(self.current_locs[i], self.goals[i]):
                self.priorities[i] += 1
            else:
                self.priorities[i] = i * 0.001
        
        agent_order = np.argsort(-self.priorities)
        
        # Reset next locations and tracking
        self.next_locs.fill(-1)
        self._update_occupancy_current()
        self.occupied_next.fill(-1)
        
        self.desired_locations.clear()
        self.actual_locations.clear()
        
        # Plan for each agent in priority order
        for agent_id in agent_order:
            if self.next_locs[agent_id, 0] == -1:
                success = self._plan_agent_with_pibt(agent_id, None)
                if not success:
                    if self.verbose:
                        print(f"Agent {agent_id} failed to plan")
                    return False
        
        return True
    
    def _update_occupancy_current(self):
        self.occupied_current.fill(-1)
        for agent_id in range(self.num_agents):
            r, c = self.current_locs[agent_id]
            self.occupied_current[r, c] = agent_id
    
    def _get_group_config_key(self, agent_ids: List[int], 
                              config_locs: Optional[np.ndarray] = None) -> tuple:
        if config_locs is None:
            config_locs = self.current_locs
        
        agent_tuple = tuple(sorted(agent_ids))
        config_tuple = tuple(tuple(config_locs[aid]) for aid in agent_tuple)
        return (agent_tuple, config_tuple)
    
    def _get_group_penalty(self, agent_ids: List[int], 
                          config_locs: Optional[np.ndarray] = None) -> float:
        # Get penalty for a specific group configuration
        key = self._get_group_config_key(agent_ids, config_locs)
        return self.group_penalties.get(key, 0.0)
    
    def _set_group_penalty(self, agent_ids: List[int], 
                          config_locs: np.ndarray, penalty: float):
        # Set penalty for a specific group configuration
        key = self._get_group_config_key(agent_ids, config_locs)
        self.group_penalties[key] = penalty
    
    def _compute_configuration_heuristic(self, config_locs: np.ndarray) -> float:
        
        # Compute h(C) = h_BD(C) + Σ h_p(C_Gr_i) for disjoint groups.
    
        # Base heuristic: sum of individual BFS distances
        h_base = sum([
            self.graph.get_distance(tuple(config_locs[i]), tuple(self.goals[i]))
            for i in range(self.num_agents)
        ])
        
        # Find applicable group penalties
        # We need disjoint groups whose configurations match current config
        applicable_penalties = 0.0
        used_agents = set()
        
        # Sort penalties by group size (larger groups first) for greedy matching
        sorted_penalties = sorted(
            self.group_penalties.items(),
            key=lambda x: len(x[0][0]),  # agent_tuple
            reverse=True
        )
        
        for (agent_tuple, config_tuple), penalty in sorted_penalties:
            # Check if this group is disjoint from already used agents
            if any(agent_id in used_agents for agent_id in agent_tuple):
                continue
            
            # Check if configuration matches
            config_matches = True
            for i, agent_id in enumerate(agent_tuple):
                agent_loc = tuple(config_locs[agent_id])
                expected_loc = config_tuple[i]
                if agent_loc != expected_loc:
                    config_matches = False
                    break
            
            if config_matches:
                applicable_penalties += penalty
                used_agents.update(agent_tuple)
        
        return h_base + applicable_penalties
    
    def _plan_agent_with_pibt(self, agent_id: int, blocked_by: Optional[int]) -> bool:
        current_pos = tuple(self.current_locs[agent_id])
        goal = tuple(self.goals[agent_id])
        
        if current_pos == goal:
            self.next_locs[agent_id] = self.current_locs[agent_id].copy()
            self.occupied_next[current_pos] = agent_id
            self.desired_locations[agent_id] = current_pos
            self.actual_locations[agent_id] = current_pos
            return True
        
        candidates = self.graph.neighbors(current_pos) + [current_pos]
        
        # Sort candidates by configuration heuristic (includes group penalties)
        def evaluate_candidate(candidate):
            # Create hypothetical configuration
            hypothetical_config = self.current_locs.copy()
            hypothetical_config[agent_id] = np.array(candidate, dtype=np.int32)
            
            # Compute heuristic with group penalties
            return self._compute_configuration_heuristic(hypothetical_config)
        
        candidates.sort(key=evaluate_candidate)
        
        # Store desired location (best option)
        self.desired_locations[agent_id] = candidates[0]
        
        # Try candidates in order
        for candidate in candidates:
            # Check vertex conflict
            if self.occupied_next[candidate] != -1:
                continue
            
            # Check swap conflict with blocking agent
            if blocked_by is not None:
                blocked_pos = tuple(self.current_locs[blocked_by])
                if blocked_pos == candidate:
                    continue
            
            # Reserve location
            self.next_locs[agent_id] = np.array(candidate, dtype=np.int32)
            self.occupied_next[candidate] = agent_id
            
            # Handle conflicting agent at target
            conflicting_agent = self.occupied_current[candidate]
            if (conflicting_agent != -1 and 
                conflicting_agent != agent_id and 
                self.next_locs[conflicting_agent, 0] == -1):
                
                # Priority inheritance - try to move conflicting agent
                if not self._plan_agent_with_pibt(conflicting_agent, agent_id):
                    # Failed, unreserve and try next candidate
                    self.occupied_next[candidate] = -1
                    self.next_locs[agent_id] = np.array([-1, -1], dtype=np.int32)
                    continue
            
            # Success, record actual location
            self.actual_locations[agent_id] = candidate
            return True
        
        # No valid move - stay in place
        self.next_locs[agent_id] = self.current_locs[agent_id].copy()
        self.occupied_next[current_pos] = agent_id
        self.actual_locations[agent_id] = current_pos
        return False
    
    def _detect_coupled_agent_groups(self) -> List[List[int]]:
        # Build undirected dependency graph
        conflict_graph = defaultdict(set)
        
        for agent_id in range(self.num_agents):
            # Skip if agent data not available
            if agent_id not in self.desired_locations or agent_id not in self.actual_locations:
                continue
            
            desired = self.desired_locations[agent_id]
            actual = self.actual_locations[agent_id]
            
            # Agent is NOT on its optimal path - someone blocked it
            if desired != actual:
                current_loc = tuple(self.current_locs[agent_id])
                
                # WHO blocked this agent? Check several cases:
                
                # Case 1: Vertex conflict - someone occupies desired location
                if self.occupied_next[desired] != -1:
                    blocker = self.occupied_next[desired]
                    if blocker != agent_id:
                        conflict_graph[agent_id].add(blocker)
                        conflict_graph[blocker].add(agent_id) 
                
                # Case 2: Someone at desired location in current timestep
                if self.occupied_current[desired] != -1:
                    blocker = self.occupied_current[desired]
                    if blocker != agent_id:
                        conflict_graph[agent_id].add(blocker)
                        conflict_graph[blocker].add(agent_id)
                
                # Case 3: swap conflict
                for other_agent in range(self.num_agents):
                    if other_agent == agent_id:
                        continue
                    
                    other_current = tuple(self.current_locs[other_agent])
                    other_next = tuple(self.next_locs[other_agent])
                    
                    # Swap: this agent wants current->desired
                    # other agent did desired->current
                    if other_current == desired and other_next == current_loc:
                        conflict_graph[agent_id].add(other_agent)
                        conflict_graph[other_agent].add(agent_id)
        
        # Find connected components via DFS
        visited = set()
        groups = []
        
        for agent in range(self.num_agents):
            if agent in visited:
                continue
            
            # DFS to find connected component
            group = []
            stack = [agent]
            
            while stack:
                curr = stack.pop()
                if curr in visited:
                    continue
                
                visited.add(curr)
                group.append(curr)
                
                # Add all neighbors in conflict graph
                for neighbor in conflict_graph[curr]:
                    if neighbor not in visited:
                        stack.append(neighbor)
            
            groups.append(sorted(group))  # Sort for consistency
        
        return groups
    
    def _update_group_penalties(self):
        groups = self._detect_coupled_agent_groups()
        
        if self.verbose:
            coupled_groups = [g for g in groups if len(g) > 1]
            if coupled_groups:
                print(f"Detected {len(coupled_groups)} coupled group(s):")
                for i, group in enumerate(coupled_groups):
                    locs = [tuple(self.current_locs[aid]) for aid in group]
                    print(f"     Group {i}: Agents {group} at {locs}")
        
        for group in groups:
            if len(group) <= 1:
                continue 
            
            # Current group configuration C_Gr
            current_config = self.current_locs.copy()
            
            # Next group configuration C^W_Gr (where they moved to)
            next_config = self.next_locs.copy()
            
            # h_BD(C^W_Gr) = sum of BFS distances for next config
            h_next = sum([
                self.graph.get_distance(
                    tuple(next_config[agent_id]),
                    tuple(self.goals[agent_id])
                )
                for agent_id in group
            ])
            
            cost = 1
            
            # Get old penalty for current configuration
            old_penalty = self._get_group_penalty(group, current_config)
            
            # Base heuristic for current config
            h_base_current = sum([
                self.graph.get_distance(
                    tuple(current_config[agent_id]),
                    tuple(self.goals[agent_id])
                )
                for agent_id in group
            ])
            
            # Total old heuristic
            h_old = h_base_current + old_penalty
            
            # Equation 1: h(C_Gr) ← max(h(C_Gr), c(C_Gr, C^W_Gr) + h(C^W_Gr))
            h_new = max(h_old, cost + h_next)
            
            # New penalty is the increase over base heuristic
            new_penalty = h_new - h_base_current
            
            penalty_increase = new_penalty - old_penalty
            
            if penalty_increase > 0:
                # Store the new penalty for this group configuration
                self._set_group_penalty(group, current_config, new_penalty)
                
                if self.verbose:
                    group_locs = [tuple(current_config[aid]) for aid in group]
                    print(f"Penalty: {old_penalty:.1f} → {new_penalty:.1f} (+{penalty_increase:.1f})")
                    print(f"Config: {group_locs}")
    
    def extract_clean_solutions(self, solution: List[List[Vertex]]) -> List[List[Vertex]]:
        clean_solutions = []
        
        for agent_id in range(len(solution)):
            messy_path = solution[agent_id]
            clean = []
            visited_positions = {}
            
            for loc in messy_path:
                if loc in visited_positions:
                    backtrack_idx = visited_positions[loc]
                    clean = clean[:backtrack_idx + 1]
                else:
                    clean.append(loc)
                    visited_positions[loc] = len(clean) - 1
            
            clean_solutions.append(clean)
        
        return clean_solutions


def animate_multiagent_search(grid_map: np.ndarray, solution: List[List[Vertex]],
                              starts: List[Vertex], goals: List[Vertex],
                              solver,
                              save_path: str = 'multiagent_realtime.gif'):
    
    num_agents = len(solution)
    timesteps = len(solution[0])
    height, width = grid_map.shape
    
    fig, ax = plt.subplots(figsize=(16, 12))
    agent_colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
    group_highlight_colors = ['yellow', 'pink', 'lightblue', 'lightgreen', 'lightsalmon', 'lavender']
    
    def update(frame):
        ax.clear()
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(-0.5, height - 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_xticks(np.arange(width))
        ax.set_yticks(np.arange(height))
        ax.set_xticklabels(range(width), fontsize=14)
        ax.set_yticklabels(range(height), fontsize=14)
        ax.grid(True, linewidth=2, color='gray', alpha=0.5)
        ax.set_title(f"Multi-Agent Real-Time Search (WinC-MAPF w/ PIBT)\nTimestep {frame}/{timesteps-1}",
                    fontsize=18, fontweight='bold', pad=20)
        
        # Get current configuration
        current_config = np.array([solution[i][frame] for i in range(num_agents)])
        
        # We need to find which group penalties apply to current configuration
        current_groups = []
        used_agents = set()
        
        for (agent_tuple, config_tuple), penalty in sorted(
            solver.group_penalties.items(),
            key=lambda x: len(x[0][0]),
            reverse=True
        ):
            # Check if this group config matches current config
            if any(aid in used_agents for aid in agent_tuple):
                continue
            
            config_matches = all(
                tuple(current_config[aid]) == config_tuple[i]
                for i, aid in enumerate(agent_tuple)
            )
            
            if config_matches and len(agent_tuple) > 1:
                current_groups.append((list(agent_tuple), penalty))
                used_agents.update(agent_tuple)
        
        for i in range(height):
            for j in range(width):
                if grid_map[i, j] == 1:
                    rect = patches.Rectangle((j-0.5, i-0.5), 1, 1,
                                            facecolor='black', 
                                            edgecolor='white',
                                            linewidth=2)
                    ax.add_patch(rect)
                else:
                    max_penalty = 0
                    for (agent_tuple, config_tuple), penalty in solver.group_penalties.items():
                        for agent_idx, loc in zip(agent_tuple, config_tuple):
                            if loc == (i, j):
                                max_penalty = max(max_penalty, penalty / len(agent_tuple))
                    
                    if max_penalty > 0:
                        alpha = min(0.2 + max_penalty * 0.05, 0.7)
                        rect = patches.Rectangle((j-0.5, i-0.5), 1, 1,
                                                facecolor='red',
                                                alpha=alpha,
                                                edgecolor='darkred',
                                                linewidth=1)
                        ax.add_patch(rect)
                        ax.text(j, i-0.35, f'P:{max_penalty:.0f}', 
                               ha='center', va='center',
                               fontsize=8, color='darkred', fontweight='bold')
                    else:
                        rect = patches.Rectangle((j-0.5, i-0.5), 1, 1,
                                                facecolor='white',
                                                edgecolor='lightgray',
                                                linewidth=1)
                        ax.add_patch(rect)
        
        for group_idx, (group_agents, penalty) in enumerate(current_groups):
            group_color = group_highlight_colors[group_idx % len(group_highlight_colors)]
            
            group_locs = [tuple(current_config[aid]) for aid in group_agents]
            
            if len(group_locs) >= 2:
                for i in range(len(group_locs)):
                    for j in range(i+1, len(group_locs)):
                        loc1, loc2 = group_locs[i], group_locs[j]
                        ax.plot([loc1[1], loc2[1]], [loc1[0], loc2[0]], 
                               linestyle='--', linewidth=3, 
                               color=group_color, alpha=0.6, zorder=5)
            
            for aid in group_agents:
                r, c = tuple(current_config[aid])
                highlight = patches.Rectangle((c-0.45, r-0.45), 0.9, 0.9,
                                             facecolor=group_color,
                                             alpha=0.3,
                                             edgecolor=group_color,
                                             linewidth=4,
                                             linestyle='-',
                                             zorder=3)
                ax.add_patch(highlight)
        
        for agent_id, (gr, gc) in enumerate(goals):
            color = agent_colors[agent_id % len(agent_colors)]
            goal_rect = patches.Rectangle((gc-0.4, gr-0.4), 0.8, 0.8,
                                         facecolor=color, alpha=0.2,
                                         edgecolor=color, linewidth=3, linestyle='--')
            ax.add_patch(goal_rect)
            ax.text(gc, gr, f'G{agent_id}', ha='center', va='center',
                   fontsize=16, color=color, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor=color, linewidth=2))
        
        if frame > 0:
            for agent_id in range(num_agents):
                color = agent_colors[agent_id % len(agent_colors)]
                path = solution[agent_id][:frame+1]
                rows, cols = [p[0] for p in path], [p[1] for p in path]
                ax.plot(cols, rows, '-', color=color, alpha=0.6, linewidth=4, zorder=1)
                ax.plot(cols, rows, 'o', color=color, markersize=8, alpha=0.4, zorder=2)
        
        for agent_id in range(num_agents):
            color = agent_colors[agent_id % len(agent_colors)]
            r, c = solution[agent_id][frame]
            circle = patches.Circle((c, r), 0.35, facecolor=color,
                                   edgecolor='black', linewidth=3, zorder=10)
            ax.add_patch(circle)
            ax.text(c, r, str(agent_id), ha='center', va='center',
                   fontsize=18, color='white', fontweight='bold', zorder=11)
        
        legend_text = f"Frame {frame}/{timesteps-1}\n"
        legend_text += f"Total Group Penalties: {len(solver.group_penalties)}\n"
        legend_text += f"Active Groups: {len(current_groups)}\n\n"
        
        config_h = solver._compute_configuration_heuristic(current_config)
        base_h = sum([solver.graph.get_distance(tuple(current_config[i]), tuple(solver.goals[i]))
                     for i in range(num_agents)])
        penalty = config_h - base_h
        legend_text += f"Config Penalty: {penalty:.1f}\n\n"
        
        if current_groups:
            legend_text += "Active Coupled Groups:\n"
            for group_idx, (group_agents, group_penalty) in enumerate(current_groups):
                group_color = group_highlight_colors[group_idx % len(group_highlight_colors)]
                legend_text += f"  Group {group_idx}: {group_agents} [P:{group_penalty:.1f}]\n"
            legend_text += "\n"
        
        for agent_id in range(num_agents):
            start = starts[agent_id]
            goal = goals[agent_id]
            
            in_group = ""
            for group_idx, (group_agents, _) in enumerate(current_groups):
                if agent_id in group_agents:
                    in_group = f" [Group {group_idx}]"
                    break
            
            legend_text += f"A{agent_id}: {start}→{goal}{in_group}\n"
        
        ax.text(0.02, 0.98, legend_text.strip(),
               transform=ax.transAxes, verticalalignment='top',
               fontsize=11, family='monospace',
               bbox=dict(boxstyle='round,pad=0.8', facecolor='wheat', 
                        alpha=0.9, edgecolor='black', linewidth=2))
        
        return []
    
    anim = animation.FuncAnimation(fig, update, frames=timesteps,
                                  interval=800, repeat=True, blit=False)
    
    print(f"Saving animation to {save_path}...")
    anim.save(save_path, writer='pillow', fps=1.25)
    print(f"Animation saved!")
    plt.close()


def test_multiagent_realtime():
    """Test with 3 agents on 4x4 grid with obstacles."""
    
    print("\n" + "="*70)
    print("TEST: Multi-Agent Real-Time Search with WinC-MAPF (PIBT)")
    print("="*70)
    
    # 4x4 grid with obstacles
    grid_map = np.zeros((4, 4), dtype=int)
    grid_map[1, 1] = 1
    grid_map[2, 1] = 1
    grid_map[1, 2] = 1
    
    graph = Graph(grid_map)
    
    # 3 agents
    starts = [
        (0, 0),
        (3, 0),
        (0, 3),
    ]
    
    goals = [
        (3, 3),
        (0, 3),
        (3, 0),
    ]
    
    print("\nGrid (# = obstacle, 0/1/2 = agents):")
    print("0 . . 2")
    print(". # # .")
    print(". # . .")
    print("1 . . .")
    print()
    
    print(f"Starts: {starts}")
    print(f"Goals:  {goals}")
    print()
    
    solver = MultiAgentRealtimeSearch(graph, starts, goals, verbose=True)
    solution = solver.solve(max_timesteps=100)
    
    if solution:
        print(f"\n{'='*70}")
        print(f"SUCCESS!")
        print(f"{'='*70}")
        
        print("\nMESSY SOLUTIONS (with exploration):")
        for agent_id in range(len(solution)):
            clean_path = [(int(p[0]), int(p[1])) for p in solution[agent_id]]
            print(f"  Agent {agent_id}: {len(clean_path)} steps")
            print(f"    Path: {' -> '.join(str(p) for p in clean_path[:15])}" + 
              ("..." if len(clean_path) > 15 else ""))
        
        print("\nCLEAN SOLUTIONS (loops removed):")
        clean_solutions = solver.extract_clean_solutions(solution)
        for agent_id in range(len(clean_solutions)):
            clean_path = [(int(p[0]), int(p[1])) for p in clean_solutions[agent_id]]
            print(f"  Agent {agent_id}: {len(clean_path)} steps")
            print(f"    Path: {' -> '.join(str(p) for p in clean_path)}")
        
        print(f"\nStatistics:")
        print(f"  Total group penalties created: {len(solver.group_penalties)}")
        print(f"  Total timesteps: {len(solution[0])}")
        
        # Save GIF
        print("\n" + "="*70)
        print("Creating animation...")
        print("="*70)
        animate_multiagent_search(
            grid_map,
            solution,
            starts,
            goals,
            solver,
            save_path='data/logs/multiagent_winc_pibt.gif'
        )
        
        return True
    else:
        print("\nFAILED")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Multi-Agent Real-Time Search'
    )
    parser.add_argument('--test', action='store_true', help='Run test')
    
    args = parser.parse_args()
    
    os.makedirs('data/logs', exist_ok=True)
    
    test_multiagent_realtime()
    
    print("\n" + "="*70)
    print("Complete!")
    print("="*70)
