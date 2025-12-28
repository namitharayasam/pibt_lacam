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


Vertex = Tuple[int, int]  # (row, col)

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

class RealtimeHeuristicSearch:
    def __init__(self, graph: Graph, start: Vertex, goal: Vertex, verbose: bool = True):
        self.graph = graph
        self.start = start
        self.goal = goal
        
        self.verbose = verbose
        
        # location -> {'actions_tried': set(), 'edges': list()}
        self.database: Dict[Vertex, Dict] = {}
        self.discovery_order = {}
        
        self.current_loc = start    
        self.messy_solution = [start]
        self.timestep = 0
        
        self.viz_history = []
        self.timestep_snapshots = []
    
    def solve(self, max_timesteps: int = 1000) -> Optional[List[Vertex]]:
        if self.verbose:
            print(f"Real-time Heuristic Search")
            print(f"Start: {self.start}, Goal: {self.goal}\n")
        
        for t in range(max_timesteps):
            self.timestep = t

            self.timestep_snapshots.append({
            'timestep': t,
            'location': self.current_loc,
            'database_snapshot': {
                loc: {
                    'actions_tried': info['actions_tried'].copy(),
                    'edges': info['edges'].copy()
                }
                for loc, info in self.database.items()
            }
        })
            
            if self.current_loc == self.goal:
                if self.verbose:
                    print(f"\nReached goal at timestep {t}")
                return self.messy_solution
            
            action = self._get_next_action()
            
            # Handle backtracking
            if action is None: # only way to get none is when ALL actins from current location have been tried
                if self.verbose:
                    print(f"t={t}: Stuck at {self.current_loc}, backtracking...")
                
                previous_loc = self._backtrack()
                if previous_loc is None:
                    if self.verbose:
                        print(f"Cannot backtrack - no solution")
                    return None
                
                self.current_loc = previous_loc
                self.messy_solution.append(self.current_loc)
                continue
            
            next_loc = self._execute_action(action)
            
            if self.verbose:
                available = self._get_available_actions()
                tried = self.database[self.current_loc]['actions_tried']
                print(f"t={t}: At {self.current_loc}, available={available}, tried={tried}, chose {action}")


            self._record_action(self.current_loc, action)
            self._record_edge(self.current_loc, next_loc)
            
            self.current_loc = next_loc
            self.messy_solution.append(self.current_loc)
        
        if self.verbose:
            print(f"\nTimeout after {max_timesteps} timesteps")
        return None
    
    def _get_next_action(self) -> Optional[Tuple[str, Vertex]]:
        if self.current_loc not in self.database:
            self.database[self.current_loc] = {
                'actions_tried': set(),
                'edges': []
            }

            self.discovery_order[self.current_loc] = self.timestep
        
        available_actions = self._get_available_actions()
        tried_actions = self.database[self.current_loc]['actions_tried']
        
        untried = [a for a in available_actions if a not in tried_actions]
        
        if not untried:
            return None  # backtrack
        
        untried.sort(key=lambda a: self._heuristic_after_action(a))
        
        return untried[0]
    
    def _get_available_actions(self) -> List[Tuple[str, Vertex]]:
        actions = []
        
        for neighbor in self.graph.neighbors(self.current_loc):
            actions.append(('MOVE', neighbor))
        
        actions.append(('STAY', self.current_loc))
        
        return actions
    
    def _heuristic_after_action(self, action: Tuple[str, Vertex]) -> float:
        _, location = action
        return abs(location[0] - self.goal[0]) + abs(location[1] - self.goal[1])
    
    def _execute_action(self, action: Tuple[str, Vertex]) -> Vertex:
        _, location = action
        return location
    
    def _record_action(self, loc: Vertex, action: Tuple[str, Vertex]):
        self.database[loc]['actions_tried'].add(action)
    
    def _record_edge(self, from_loc: Vertex, to_loc: Vertex):
        if to_loc not in self.database:
            self.database[to_loc] = {
                'actions_tried': set(),
                'edges': []
            }
            self.discovery_order[to_loc] = self.timestep
        
        if from_loc != to_loc:
            self.database[to_loc]['edges'].append((from_loc, to_loc))
    
    def _backtrack(self) -> Optional[Vertex]:
        if not self.database[self.current_loc]['edges']:
            return None
        
        from_loc, to_loc = self.database[self.current_loc]['edges'].pop()
        return from_loc
    
    def extract_clean_solution(self) -> Optional[List[Vertex]]:
        if not self.messy_solution:
            return None
        
        clean = []
        visited_positions = {}  # position -> index in clean path
        
        for loc in self.messy_solution:
            if loc in visited_positions:
                # we've been here before - backtracked, so remove loop
                backtrack_idx = visited_positions[loc]
                clean = clean[:backtrack_idx + 1] 
            else:
                clean.append(loc)
                visited_positions[loc] = len(clean) - 1
        
        return clean


def animate_realtime_search(grid_map: np.ndarray, messy_solution: List[Vertex], 
                           start: Vertex, goal: Vertex, 
                           database: Dict, save_path: str = 'realtime_search.gif'):    
    timesteps = len(messy_solution)
    height, width = grid_map.shape
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    def update(frame):
        ax.clear()
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(-0.5, height - 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_xticks(range(width))
        ax.set_yticks(range(height))
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Real-Time Heuristic Search - Timestep {frame}/{timesteps-1}", 
                    fontsize=16, fontweight='bold')
        
        for i in range(height):
            for j in range(width):
                if grid_map[i, j] == 1:
                    ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, 
                                              color='black', alpha=0.7))
        
        gr, gc = goal
        ax.add_patch(patches.Circle((gc, gr), 0.35, 
                                   facecolor='gold', alpha=0.4, edgecolor='orange', linewidth=2))
        ax.text(gc, gr, 'G', ha='center', va='center', 
               fontsize=12, color='orange', fontweight='bold')
        
        for loc in database.keys():
            r, c = loc
            if loc != start and loc != goal:
                ax.add_patch(patches.Rectangle((c-0.4, r-0.4), 0.8, 0.8,
                                              facecolor='lightblue', alpha=0.3,
                                              edgecolor='blue', linewidth=1))
        
        if frame > 0:
            path = messy_solution[:frame+1]
            rows = [p[0] for p in path]
            cols = [p[1] for p in path]
            ax.plot(cols, rows, '-', color='red', alpha=0.6, linewidth=3)
        
        r, c = messy_solution[frame]
        ax.add_patch(patches.Circle((c, r), 0.4,
                                   facecolor='red', edgecolor='darkred', linewidth=2))
        ax.text(c, r, 'A', ha='center', va='center', 
               fontsize=14, color='white', fontweight='bold')
        
        status = f"Location: {messy_solution[frame]}"
        if frame > 0:
            prev = messy_solution[frame-1]
            if prev == messy_solution[frame]:
                status += " (STAY)"
        
        ax.text(0.02, 0.98, status,
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               fontsize=10, family='monospace')
        
        return []
    
    anim = animation.FuncAnimation(fig, update, frames=timesteps, 
                                  interval=500, repeat=True, blit=False)
    
    anim.save(save_path, writer='pillow', fps=2)
    
    plt.close()

def test_realtime_backtracking():
    grid_map = np.zeros((4, 3), dtype=int)
    grid_map[1, 1] = 1 
    grid_map[2, 1] = 1  
    grid_map[3, 1] = 1 
    
    graph = Graph(grid_map)
    
    start = (1, 0) 
    goal = (3, 2)
    
    print("\nGrid (# = obstacle, S = start, G = goal):")
    print(". . .")
    print("S # .")
    print(". # .")
    print(". # G")
    print()
    
    print(f"Start: {start}")
    print(f"Goal:  {goal}")
    print()
    
    solver = RealtimeHeuristicSearch(graph, start, goal, verbose=True)
    solution = solver.solve(max_timesteps=50)
    
    if solution:
        print(f"\nSUCCESS!")
        print(f"  Messy solution length: {len(solution)}")
        print(f"  Path: {' -> '.join(str(p) for p in solution)}")
        
        clean = solver.extract_clean_solution()
        if clean:
            print(f"  Clean solution length: {len(clean)}")
            print(f"  Clean path: {' -> '.join(str(p) for p in clean)}")
        
        print("\n" + "="*70)
        print("DATABASE TABLE:")
        print("="*70)
        print(f"{'Location':<12} {'Actions Tried':<30} {'Parent':<12} {'All Edges':<25}")
        print("-" * 79)

        for loc in sorted(solver.database.keys(), key=lambda x: solver.discovery_order[x]):
            info = solver.database[loc]
            
            if info['actions_tried']:
                actions_str = ', '.join([f"{a[0][0]}{a[1]}" for a in info['actions_tried']])
            else:
                actions_str = "none"
            
            # Get parent
            if info['edges']:
                parent = str(info['edges'][-1][0])  # Last edge's from_loc
            else:
                parent = "⊥"
            
            # All edges
            if info['edges']:
                edges_str = ', '.join([f"{e[0]}→{e[1]}" for e in info['edges']])
            else:
                edges_str = "none"
            
            print(f"{str(loc):<12} {actions_str:<30} {parent:<12} {edges_str:<25}")

        print("\n" + "="*70)
        print("TIMESTEP-BY-TIMESTEP TRACE:")
        print("="*70)
        print(f"{'Timestep':<10} {'Location':<12} {'Action':<20} {'Parent':<12} {'Edges':<25}")
        print("-" * 79)

        for snapshot in solver.timestep_snapshots:
            t = snapshot['timestep']
            loc = snapshot['location']
            db = snapshot['database_snapshot']
            
            if t == 0:
                action = "START"
            else:
                prev_loc = solver.messy_solution[t-1]
                action = "STAY" if prev_loc == loc else f"MOVE to {loc}"
            
            if loc in db:
                edges = db[loc]['edges']
                parent = str(edges[-1][0]) if edges else "⊥"
                edges_str = ', '.join([f"{e[0]}→{e[1]}" for e in edges]) if edges else "none"
            else:
                parent = "⊥"
                edges_str = "none"
            
            print(f"t={t:<8} {str(loc):<12} {action:<20} {parent:<12} {edges_str:<25}")

        
        animate_realtime_search(
                grid_map, 
                solver.messy_solution, 
                start, 
                goal, 
                solver.database,
                save_path='data/logs/realtime_search.gif'
            )
        
        return True
                    
    else:
        print("\nFAILED")
        return False

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Real-Time Heuristic Search for Single-Agent Pathfinding',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--map', help='Path to map file')
    parser.add_argument('--start', help='Start position (row,col)')
    parser.add_argument('--goal', help='Goal position (row,col)')
    parser.add_argument('--gif', action='store_true', help='Save GIF')
    
    args = parser.parse_args()
    
    os.makedirs('data/logs', exist_ok=True)
    
    print("\n" + "="*70)
    print("REAL-TIME HEURISTIC SEARCH - Single Agent Pathfinding")
    print("="*70)
    
    if args.map and args.start and args.goal:
        grid_map = load_map_file(args.map)
        start = tuple(map(int, args.start.split(',')))
        goal = tuple(map(int, args.goal.split(',')))
        
        graph = Graph(grid_map)
        solver = RealtimeHeuristicSearch(graph, start, goal, verbose=True)
        solution = solver.solve(max_timesteps=1000)
        
        if solution:
            print(f"\nSUCCESS!")
            print(f"  Messy: {len(solution)} steps")
            clean = solver.extract_clean_solution()
            if clean:
                print(f"  Clean: {len(clean)} steps")
            
            if args.gif:
                print("\nSaving GIF...")
                animate_realtime_search(grid_map, solver.messy_solution, 
                                       start, goal, solver.database,
                                       save_path='data/logs/realtime_search.gif')
                print("Saved to data/logs/realtime_search.gif")
        else:
            print("\nFAILED")
    else:
        # default
        test_realtime_backtracking()
    
    print("\n" + "="*70)
    print("Complete!")
    print("="*70)
