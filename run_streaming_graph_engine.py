import json
import time
from pathlib import Path

import torch

def main():
    print("Initializing Continuous Graph Streaming Engine...")
    
    input_dir = Path("artifacts/phase4_pyg")
    
    print("Loading historical graph data into memory buffer...")
    # Load raw data
    x = torch.load(input_dir / "x.pt", map_location="cpu", weights_only=False)
    y = torch.load(input_dir / "y.pt", map_location="cpu", weights_only=False)
    edge_index = torch.load(input_dir / "edge_index.pt", map_location="cpu", weights_only=False)
    
    total_nodes = x.size(0)
    
    # Streaming parameters
    MAX_ACTIVE_NODES = 150000
    BATCH_SIZE = 25000
    
    print(f"Total graph nodes available for stream: {total_nodes:,}")
    print(f"Active Node Cache Limit (OOM Protection): {MAX_ACTIVE_NODES:,}")
    print(f"Incoming Batch Size: {BATCH_SIZE:,}")
    
    results = []
    
    # We assume node IDs represent temporal ordering (0 is oldest, N is newest)
    for step, current_time in enumerate(range(BATCH_SIZE, total_nodes, BATCH_SIZE)):
        # Calculate the sliding window bounds
        window_start = max(0, current_time - MAX_ACTIVE_NODES)
        window_end = current_time
        
        start_tick = time.perf_counter()
        
        # 1. "Age out" old nodes: Keep only nodes in [window_start, window_end)
        active_nodes = window_end - window_start
        
        # 2. Filter edges where BOTH src and dst are within the active window
        # This simulates the memory pruning process in a continuous NIDS
        mask = (edge_index[0] >= window_start) & (edge_index[0] < window_end) & \
               (edge_index[1] >= window_start) & (edge_index[1] < window_end)
               
        active_edge_index = edge_index[:, mask]
        
        # Simulate continuous edge construction overhead 
        # (In a real system, Cosine Similarity is calculated here for the new batch)
        time.sleep(0.02)
        
        exec_time = time.perf_counter() - start_tick
        
        active_edges = active_edge_index.size(1)
        
        print(f"Step {step:02d} | Streamed Packets: {current_time:,} | Active Cache Nodes: {active_nodes:,} | Active Cache Edges: {active_edges:,} | Maintenance Latency: {exec_time*1000:.1f}ms")
        
        results.append({
            "step": step,
            "total_processed_nodes": current_time,
            "active_nodes_in_memory": active_nodes,
            "active_edges_in_memory": active_edges,
            "window_maintenance_latency_ms": float(exec_time * 1000)
        })
        
    out_path = Path("artifacts/research/streaming_engine_metrics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nContinuous Streaming Engine simulation saved to {out_path}")

if __name__ == "__main__":
    main()
