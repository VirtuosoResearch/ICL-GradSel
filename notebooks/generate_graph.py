import numpy as np
import json
import os
import argparse
from collections import deque

def generate_graph(n, m):
    edges = []
    adjacency_list = {i: set() for i in range(n)}
    
    while len(edges) < 2 * m:
        u, v = np.random.choice(n, 2, replace=False)
        if v not in adjacency_list[u]:
            adjacency_list[u].add(v)
            adjacency_list[v].add(u)
            edges.append((u, v))
            edges.append((v, u))
    
    return edges, adjacency_list

def is_connected(n, adjacency_list, start_node, target_node):
    visited = set()
    queue = deque([start_node])
    visited.add(start_node)
    intermediate_steps = []
    
    while queue:
        u = queue.popleft()
        intermediate_steps.append(u)
        if u == target_node:
            return "Yes", intermediate_steps
        for v in adjacency_list[u]:
            if v not in visited:
                visited.add(v)
                queue.append(v)
    
    return "No", intermediate_steps

def main(args):
    data_size = args.data_size
    file_path = "./graphqa.jsonl"
    cnt_con = 0
    with open(file_path, "w") as f:
        for _ in range(data_size):
            n = np.random.randint(15, 21)
            m = np.random.randint(max(1, n - 3), min(n + 1, (n * (n - 1)) // 2) + 1)
            edges, adjacency_list = generate_graph(n, m)
            start_node, target_node = np.random.choice(n, 2, replace=False)
            connected, intermediate_steps = is_connected(n, adjacency_list, start_node, target_node)
            if connected== "Yes": cnt_con+=1
            instance = {
                "task": "graphqa",
                "input": f"G describes a graph among nodes {', '.join(map(str, range(n)))}.\n"
                          + "\n".join([f"Node {u} is connected to nodes {', '.join(map(str, adjacency_list[u]))}." for u in range(n)])
                          + f"\nQuestion: Is node {start_node} connected to node {target_node}?\nAnswer:",
                "output": connected,
                "options": ["Yes", "No"]
            }
            
            f.write(json.dumps(instance) + "\n")
    print(cnt_con)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_size", type=int, default=1000)
    args = parser.parse_args()
    
    main(args)
