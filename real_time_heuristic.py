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
import networkx as nx
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from networkx.drawing.nx_agraph import graphviz_layout
import random 

Vertex = Tuple[int, int]  # (row, col)

class Graph:    
    def __init__(self, grid_map: np.ndarray):
        self.grid_map = grid_map
        self.height, self.width = grid_map.shape
        
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
        
        # location -> {'actions_tried': set(), 'incoming_edges': list(), 'incoming_edges_left': list()}
        self.database: Dict[Vertex, Dict] = {}
        self.discovery_order = {}
        
        self.current_loc = start    
        self.messy_solution = [start]
        self.move_types = [] # explore or backtrack
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
                    'incoming_edges': info['incoming_edges'].copy(),
                    'incoming_edges_left': info['incoming_edges_left'].copy()
                }
                for loc, info in self.database.items()
            }
        })
            
            if self.current_loc == self.goal:
                if self.verbose:
                    print(f"\nReached goal at timestep {t}")
                return self.messy_solution
            
            action = self._get_next_action()
            
            if action is None:
                if self.verbose:
                    print(f"t={t}: Stuck at {self.current_loc}, backtracking...")
                
                previous_loc = self._backtrack()
                if previous_loc is None:
                    if self.verbose:
                        print(f"Cannot backtrack - no solution")
                    return None
                
                self.current_loc = previous_loc
                self.messy_solution.append(self.current_loc)
                self.move_types.append('backtrack') 
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
            self.move_types.append('explore') 
        
        if self.verbose:
            print(f"\nTimeout after {max_timesteps} timesteps")
        return None
    
    def _get_next_action(self) -> Optional[Tuple[str, Vertex]]:
        if self.current_loc not in self.database:
            self.database[self.current_loc] = {
                'actions_tried': set(),
                'incoming_edges': [],
                'incoming_edges_left': []
            }
            self.discovery_order[self.current_loc] = self.timestep
        
        available_actions = self._get_available_actions()
        tried_actions = self.database[self.current_loc]['actions_tried']
        
        untried = [a for a in available_actions if a not in tried_actions]
        
        if not untried:
            return None 
        
        untried.sort(key=lambda a: self._heuristic_after_action(a))
        best_h = self._heuristic_after_action(untried[0])
        best_actions = [a for a in untried if self._heuristic_after_action(a) == best_h]
        
        return random.choice(best_actions)
    
    def _get_available_actions(self) -> List[Tuple[str, Vertex]]:
        actions = []
        
        for neighbor in self.graph.neighbors(self.current_loc):
            actions.append(('MOVE', neighbor))
        
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
                'incoming_edges': [],
                'incoming_edges_left': []
            }
            self.discovery_order[to_loc] = self.timestep
        
        if from_loc != to_loc:
            edge = (from_loc, to_loc)
            self.database[to_loc]['incoming_edges'].append(edge)
            self.database[to_loc]['incoming_edges_left'].append(edge)
    
    def _backtrack(self) -> Optional[Vertex]:
        if not self.database[self.current_loc]['incoming_edges_left']:
            return None
        
        from_loc, to_loc = self.database[self.current_loc]['incoming_edges_left'].pop()
        return from_loc
    
    def extract_clean_solution(self) -> Optional[List[Vertex]]:
        if not self.messy_solution:
            return None
        
        clean = [self.messy_solution[0]]
        
        for i in range(1, len(self.messy_solution)):
            current = self.messy_solution[i]
            
            if current in clean:
                idx = len(clean) - 1
                while idx >= 0 and clean[idx] != current:
                    idx -= 1
                clean = clean[:idx + 1]
            else:
                clean.append(current)
        
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
        
        ax.text(0.02, 0.98, status,
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               fontsize=10, family='monospace')
        
        return []
    
    anim = animation.FuncAnimation(fig, update, frames=timesteps, 
                                  interval=500, repeat=True, blit=False)
    
    anim.save(save_path, writer='pillow', fps=2)
    
    plt.close()


def visualize_search_graph(database: Dict, start: Vertex, goal: Vertex, 
                          discovery_order: Dict, messy_solution: List[Vertex],
                          move_types: List[str],
                          save_path: str = 'search_graph.png'):

    G = nx.DiGraph()
    
    for loc in database.keys():
        G.add_node(loc)
    
    for loc, info in database.items():
        for edge in info['incoming_edges']:
            from_loc, to_loc = edge
            G.add_edge(from_loc, to_loc)
    
    exploration_edges = set()
    backtrack_edges = {}
    
    for i in range(len(move_types)):
        from_loc = messy_solution[i]
        to_loc = messy_solution[i + 1]
        
        if from_loc == to_loc:
            continue
        
        edge = (from_loc, to_loc)
        timestep = i + 1  
        
        if move_types[i] == 'backtrack':
            if edge not in backtrack_edges:
                backtrack_edges[edge] = []
            backtrack_edges[edge].append(timestep)
        else:
            exploration_edges.add(edge)
    
    fig, ax = plt.subplots(figsize=(34, 28))
    
    try:
        # pos = nx.planar_layout(G)
        pos = graphviz_layout(G, prog='dot')
    except:
        try:
            # pos = nx.kamada_kawai_layout(G)
             pos = nx.kamada_kawai_layout(G)
        except:
            pos = nx.spring_layout(G, k=3, iterations=100, seed=42)
    
    node_colors = []
    for node in G.nodes():
        if node == start:
            node_colors.append('#90EE90')  
        elif node == goal:
            node_colors.append('#FFB6C6') 
        else:
            node_colors.append('#ADD8E6')  
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=1500, alpha=0.95, ax=ax,
                          edgecolors='black', linewidths=2.5)
    
    if exploration_edges:
        nx.draw_networkx_edges(G, pos, 
                              edgelist=list(exploration_edges),
                              edge_color='black',
                              arrows=True, 
                              arrowsize=25,
                              arrowstyle='->',
                              width=2.5,
                              connectionstyle='arc3,rad=0.1',
                              node_size=1500,
                              ax=ax,
                              alpha=0.8)
    
    backtrack_edge_list = list(backtrack_edges.keys())
    if backtrack_edge_list:
        nx.draw_networkx_edges(G, pos, 
                              edgelist=backtrack_edge_list,
                              edge_color='red',
                              arrows=True, 
                              arrowsize=25,
                              arrowstyle='->',
                              width=3.5,
                              style='dashed',
                              connectionstyle='arc3,rad=0.25',
                              node_size=1500,
                              ax=ax,
                              alpha=0.9)
    
    edge_labels = {}
    for edge, timesteps in backtrack_edges.items():
        if len(timesteps) == 1:
            edge_labels[edge] = f"t={timesteps[0]}"
        else:
            times_str = ','.join([f"{t}" for t in timesteps])
            edge_labels[edge] = f"t={times_str}"
    
    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels, 
                                     font_size=8, 
                                     font_color='darkred',
                                     font_weight='bold',
                                     bbox=dict(boxstyle='round,pad=0.4', 
                                              facecolor='yellow', 
                                              edgecolor='red', 
                                              alpha=0.85,
                                              linewidth=1.5),
                                     ax=ax)
    

    labels = {}
    for node in G.nodes():
        t = discovery_order[node]
        if node == start:
            labels[node] = f"{node}\nt={t}\n[START]"
        elif node == goal:
            labels[node] = f"{node}\nt={t}\n[GOAL]"
        else:
            labels[node] = f"{node}\nt={t}"
    
    nx.draw_networkx_labels(G, pos, labels, font_size=8, 
                           font_weight='bold', 
                           font_family='monospace',
                           ax=ax)
    

    title = "Real-Time Heuristic Search - Tree Structure\n"
    title += f"Nodes Explored: {len(G.nodes())} | "
    title += f"Exploration Edges: {len(exploration_edges)} | "
    title += f"Backtrack Edges: {len(backtrack_edges)}"
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    

    legend_elements = [
        patches.Patch(facecolor='#90EE90', edgecolor='black', linewidth=2, 
                     label='Start Node'),
        patches.Patch(facecolor='#FFB6C6', edgecolor='black', linewidth=2,
                     label='Goal Node'),
        patches.Patch(facecolor='#ADD8E6', edgecolor='black', linewidth=2,
                     label='Explored Node'),
        Line2D([0], [0], color='black', linewidth=2.5, 
               label='Exploration Edge (forward)'),
        Line2D([0], [0], color='red', linewidth=3.5, linestyle='--', 
               label='Backtrack Edge (stuck!)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=12, 
             framealpha=0.95, edgecolor='black')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"\n  Search graph visualization saved to {save_path}")
    print(f"  - Total nodes: {len(G.nodes())}")
    print(f"  - Exploration edges (black solid): {len(exploration_edges)}")
    print(f"  - Backtrack edges (red dashed): {len(backtrack_edges)}")
    
    if backtrack_edges:
        print(f"\n  Backtracking events:")
        for edge, timesteps in sorted(backtrack_edges.items(), 
                                     key=lambda x: x[1][0]):
            from_loc, to_loc = edge
            times = ', '.join([f"t={t}" for t in timesteps])
            print(f"    {from_loc} -> {to_loc} at {times}")
    else:
        print(f"\n  No backtracking occurred!")


def print_search_stats(database: Dict, start: Vertex, goal: Vertex, discovery_order: Dict):
    print("\n" + "-"*70)
    print("SEARCH GRAPH STATISTICS:")
    print("-"*70)
    
    total_edges = sum(len(info['incoming_edges']) for info in database.values())
    multi_parent_nodes = [(loc, info['incoming_edges']) 
                          for loc, info in database.items() 
                          if len(info['incoming_edges']) > 1]
    
    print(f"\nTotal nodes explored: {len(database)}")
    print(f"Total edges: {total_edges}")
    print(f"Nodes with multiple parents: {len(multi_parent_nodes)}")
    
    if multi_parent_nodes:
        print("\nNodes with multiple parents:")
        for loc, edges in multi_parent_nodes:
            parents = [e[0] for e in edges]
            print(f"  {loc}: {' & '.join(map(str, parents))}")
    else:
        print("\nNo loops detected")
    
    print("-"*70)


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
    else:
        print("\nFAILED - No solution found")
    
    if solver.messy_solution and len(solver.messy_solution) > 1:
        print("\n" + "-"*70)
        print("DATABASE TABLE:")
        print("-"*70)
        print(f"{'Location':<12} {'Actions Tried':<30} {'Parent':<12} {'All Edges':<25}")
        print("-" * 79)

        for loc in sorted(solver.database.keys(), key=lambda x: solver.discovery_order[x]):
            info = solver.database[loc]
            
            if info['actions_tried']:
                actions_str = ', '.join([f"{a[0][0]}{a[1]}" for a in info['actions_tried']])
            else:
                actions_str = "none"
            
            if info['incoming_edges']:
                parent = str(info['incoming_edges'][-1][0])
            else:
                parent = "⊥"
            
            if info['incoming_edges']:
                edges_str = ', '.join([f"{e[0]}→{e[1]}" for e in info['incoming_edges']])
            else:
                edges_str = "none"
            
            print(f"{str(loc):<12} {actions_str:<30} {parent:<12} {edges_str:<25}")

        print("\n" + "-"*70)
        print("TIMESTEP-BY-TIMESTEP TRACE:")
        print("-"*70)
        print(f"{'Timestep':<10} {'Location':<12} {'Action Chosen':<25} {'How Got Here':<25} {'Parent':<12} {'Edges':<30}")
        print("-" * 122)

        for snapshot in solver.timestep_snapshots:
            t = snapshot['timestep']
            loc = snapshot['location']
            db = snapshot['database_snapshot']
            
            if t == 0:
                action_chosen = "START"
            elif t < len(solver.messy_solution) - 1:
                if t < len(solver.move_types) and solver.move_types[t] == 'backtrack':
                    action_chosen = "BACKTRACK"
                else:
                    next_loc = solver.messy_solution[t + 1]
                    if next_loc == loc:
                        action_chosen = "STAY"
                    else:
                        action_chosen = f"MOVE to {next_loc}"
            else:
                action_chosen = "GOAL REACHED" if solution else "STOPPED"
            
            if t == 0:
                how_got_here = "START"
            elif t - 1 < len(solver.move_types):
                move_type = solver.move_types[t - 1]
                prev_loc = solver.messy_solution[t - 1]
                
                if move_type == 'backtrack':
                    how_got_here = f"← backtrack from {prev_loc}"
                else:
                    how_got_here = f"← from {prev_loc}"
            else:
                how_got_here = "?"
            
            if loc in db:
                edges = db[loc]['incoming_edges']
                parent = str(edges[-1][0]) if edges else "⊥"
                edges_str = ', '.join([f"{e[0]}→{e[1]}" for e in edges]) if edges else "none"
            else:
                parent = "⊥"
                edges_str = "none"
            
            print(f"t={t:<8} {str(loc):<12} {action_chosen:<25} {how_got_here:<25} {parent:<12} {edges_str:<30}")
        
        
        print_search_stats(
            solver.database,
            start,
            goal,
            solver.discovery_order
        )

        visualize_search_graph(
            solver.database,
            start,
            goal,
            solver.discovery_order,
            solver.messy_solution,
            solver.move_types,
            save_path='data/logs/search_graph.png'
        )

        animate_realtime_search(
            grid_map, 
            solver.messy_solution, 
            start, 
            goal, 
            solver.database,
            save_path='data/logs/realtime_search.gif'
        )
    
    return solution is not None

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
            if char in ['@', 'O', 'T', 'W']:
                grid_map[i, j] = 1
    
    return grid_map


def print_database_and_trace(solver, solution):
    print("\n" + "-"*70)
    print("DATABASE TABLE:")
    print("-"*70)
    print(f"{'Location':<12} {'Actions Tried':<30} {'Parent':<12} {'All Edges':<25}")
    print("-" * 79)

    for loc in sorted(solver.database.keys(), key=lambda x: solver.discovery_order[x]):
        info = solver.database[loc]
        
        if info['actions_tried']:
            actions_str = ', '.join([f"{a[0][0]}{a[1]}" for a in info['actions_tried']])
        else:
            actions_str = "none"
        
        if info['incoming_edges']:
            parent = str(info['incoming_edges'][-1][0])
        else:
            parent = "⊥"
        
        if info['incoming_edges']:
            edges_str = ', '.join([f"{e[0]}→{e[1]}" for e in info['incoming_edges']])
        else:
            edges_str = "none"
        
        print(f"{str(loc):<12} {actions_str:<30} {parent:<12} {edges_str:<25}")

    print("\n" + "-"*70)
    print("TIMESTEP-BY-TIMESTEP TRACE:")
    print("-"*70)
    print(f"{'Timestep':<10} {'Location':<12} {'Action Chosen':<25} {'How Got Here':<25} {'Parent':<12} {'Edges':<30}")
    print("-" * 122)

    for snapshot in solver.timestep_snapshots:
        t = snapshot['timestep']
        loc = snapshot['location']
        db = snapshot['database_snapshot']
        
        if t == 0:
            action_chosen = "START"
        elif t < len(solver.messy_solution) - 1:
            if t < len(solver.move_types) and solver.move_types[t] == 'backtrack':
                action_chosen = "BACKTRACK"
            else:
                next_loc = solver.messy_solution[t + 1]
                if next_loc == loc:
                    action_chosen = "STAY"
                else:
                    action_chosen = f"MOVE to {next_loc}"
        else:
            action_chosen = "GOAL REACHED" if solution else "STOPPED"
        
        if t == 0:
            how_got_here = "START"
        elif t - 1 < len(solver.move_types):
            move_type = solver.move_types[t - 1]
            prev_loc = solver.messy_solution[t - 1]
            
            if move_type == 'backtrack':
                how_got_here = f"← backtrack from {prev_loc}"
            else:
                how_got_here = f"← from {prev_loc}"
        else:
            how_got_here = "?"
        
        if loc in db:
            edges = db[loc]['incoming_edges']
            parent = str(edges[-1][0]) if edges else "⊥"
            edges_str = ', '.join([f"{e[0]}→{e[1]}" for e in edges]) if edges else "none"
        else:
            parent = "⊥"
            edges_str = "none"
        
        print(f"t={t:<8} {str(loc):<12} {action_chosen:<25} {how_got_here:<25} {parent:<12} {edges_str:<30}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Real-Time Heuristic Search for Single-Agent Pathfinding',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--map', help='Path to map file')
    parser.add_argument('--start', help='Start position (row,col)')
    parser.add_argument('--goal', help='Goal position (row,col)')
    parser.add_argument('--gif', action='store_true', help='Save GIF and visualizations')
    
    args = parser.parse_args()
    
    os.makedirs('data/logs', exist_ok=True)
    
    print("\n" + "-"*70)
    print("REAL-TIME HEURISTIC SEARCH - Single Agent Pathfinding")
    print("-"*70)
    
    if args.map and args.start and args.goal:
        grid_map = load_map_file(args.map)
        start = tuple(map(int, args.start.split(',')))
        goal = tuple(map(int, args.goal.split(',')))
        
        graph = Graph(grid_map)
        solver = RealtimeHeuristicSearch(graph, start, goal, verbose=True)
        solution = solver.solve(max_timesteps=10000)
        
        if solution:
            print(f"\nSUCCESS!")
            print(f"  Messy: {len(solution)} steps")
            print(f"  Messy path: {' -> '.join(str(p) for p in solution)}")

            clean = solver.extract_clean_solution()
            
            if clean:
                print(f"  Clean: {len(clean)} steps")
                print(f"  Clean path: {' -> '.join(str(p) for p in clean)}")
        else:
            print("\nFAILED")
        
        if solver.messy_solution and len(solver.messy_solution) > 1 and args.gif:
            print_database_and_trace(solver, solution)
            
            print_search_stats(
                solver.database,
                start,
                goal,
                solver.discovery_order
            )

            print("\nGenerating visualizations...")

            visualize_search_graph(
                solver.database,
                start,
                goal,
                solver.discovery_order,
                solver.messy_solution,
                solver.move_types, 
                save_path='data/logs/search_graph.png'
            )

            animate_realtime_search(grid_map, solver.messy_solution, 
                                   start, goal, solver.database,
                                   save_path='data/logs/realtime_search.gif')
            print("Saved to data/logs/realtime_search.gif")

    else:
        test_realtime_backtracking()
    
    print("\n" + "-"*70)
    print("Complete!")
    print("-"*70)
