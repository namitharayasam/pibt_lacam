import numpy as np
import pandas as pd
import argparse
import time
import os
from typing import List, Tuple
import sys

from pibt_lacam import (
    Graph, PIBT, LaCAM, 
    load_map_file, load_scenario_file,
    calculate_costs
)

def run_single_test(map_path: str, scen_path: str, num_agents: int, 
                   algorithm: str, max_timesteps: int = 500,
                   node_limit: int = 120000, time_limit: float = 100.0) -> dict:
    try:
        # Load map and scenario
        grid_map = load_map_file(map_path)
        graph = Graph(grid_map)
        map_name, starts, goals = load_scenario_file(scen_path, num_agents)
        
        # Run algorithm
        start_time = time.time()
        
        if algorithm == 'PIBT':
            solver = PIBT(graph, starts, goals)
            solution = solver.solve(max_timesteps=max_timesteps)
        elif algorithm == 'LaCAM':
            solver = LaCAM(graph, starts, goals)
            solution = solver.solve(node_limit=node_limit, time_limit=time_limit)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        elapsed_time = time.time() - start_time
        
        # Calculate results
        if solution is not None:
            soc, makespan = calculate_costs(solution, goals)
            return {
                'algorithm': algorithm,
                'num_agents': num_agents,
                'success': True,
                'time': elapsed_time,
                'soc': soc,
                'makespan': makespan
            }
        else:
            return {
                'algorithm': algorithm,
                'num_agents': num_agents,
                'success': False,
                'time': elapsed_time,
                'soc': -1,
                'makespan': -1
            }
    
    except Exception as e:
        print(f"Error in test: {e}")
        return {
            'algorithm': algorithm,
            'num_agents': num_agents,
            'success': False,
            'time': -1,
            'soc': -1,
            'makespan': -1
        }

def run_batch_experiments(map_path: str, scen_path: str, 
                         agent_counts: List[int],
                         num_scenarios: int = 10,
                         algorithms: List[str] = ['PIBT', 'LaCAM'],
                         output_csv: str = None) -> pd.DataFrame:

    os.makedirs('data/logs', exist_ok=True)
    
    results = []
    total_tests = len(agent_counts) * len(algorithms) * num_scenarios
    test_count = 0
    
    print(f"\n{'='*70}")
    print(f"Starting batch experiments")
    print(f"Map: {map_path}")
    print(f"Scenarios: {scen_path}")
    print(f"Agent counts: {agent_counts}")
    print(f"Algorithms: {algorithms}")
    print(f"Scenarios per agent count: {num_scenarios}")
    print(f"Total tests: {total_tests}")
    print(f"{'='*70}\n")
    
    for num_agents in agent_counts:
        for algo in algorithms:
            print(f"\nTesting {algo} with {num_agents} agents...")
            
            for scen_idx in range(num_scenarios):
                test_count += 1
                print(f"  Test {test_count}/{total_tests}: {algo}, {num_agents} agents, scenario {scen_idx+1}/{num_scenarios}")
                
                result = run_single_test(
                    map_path=map_path,
                    scen_path=scen_path,
                    num_agents=num_agents,
                    algorithm=algo
                )
                
                result['scenario_idx'] = scen_idx
                results.append(result)
                
                status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
                print(f"    {status} - Time: {result['time']:.3f}s")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV if specified
    if output_csv:
        csv_path = f"data/logs/{output_csv}"
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Batch runner for LaCAM and PIBT comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python batch_runner.py --map data/mapf-map/maze-32-32-4.map --scen data/mapf-scen-random/maze-32-32-4-random-1.scen --agents 5 10 15 20 --num_scens 10 --output results.csv
        """
    )
    
    parser.add_argument('--map', type=str, required=True,
                       help='Path to .map file')
    parser.add_argument('--scen', type=str, required=True,
                       help='Path to .scen file')
    parser.add_argument('--agents', type=int, nargs='+', required=True,
                       help='List of agent counts to test (e.g., 5 10 15 20)')
    parser.add_argument('--num_scens', type=int, default=10,
                       help='Number of scenarios per agent count')
    parser.add_argument('--algorithms', type=str, nargs='+', 
                       default=['PIBT', 'LaCAM'],
                       help='Algorithms to test')
    parser.add_argument('--output', type=str, default='batch_results.csv',
                       help='Output CSV filename')
    
    args = parser.parse_args()
    
    run_batch_experiments(
        map_path=args.map,
        scen_path=args.scen,
        agent_counts=args.agents,
        num_scenarios=args.num_scens,
        algorithms=args.algorithms,
        output_csv=args.output
    )
