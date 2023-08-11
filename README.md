# Stress-Centrality
**It is measure of number of shortest path through a node.**
The stress centrality is used for spatial load balancing  in wide areas.
The project aims at finding an edge whose removal leads to maximum reduction of stress centrality locally and globally.
## Approaches
- Greedy optimization of the brute force approach.
- GNN using GraphSAGE to capture node embeddings and edge prediction.
- Parallel Architecture: Brute force over GPU threads using CUDA.
