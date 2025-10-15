# Fast Shortest Path Distance Estimation in Networks using Community Detection with Landmarks-based Approach

This repository contains the implementation of our course project for **Social Network Analysis for Computer Scientists (SNACS ’23)** at **Leiden University**.  
The project focuses on efficiently computing shortest-path distances in large-scale networks using **community detection** and **landmark-based estimation**.

---

## 🧠 Overview

Computing shortest-path distances in massive graphs (like social networks or co-authorship graphs) is computationally expensive.  
This project proposes a **hybrid method** that integrates **Louvain community detection** with **landmark-based distance estimation**.

By first partitioning the network into communities, and then strategically selecting **landmarks** (nodes with high degree centrality) within each community, we can:
- Reduce the search space for shortest paths,
- Maintain reasonable accuracy,
- Improve computational efficiency.

---

## 🧩 Methodology

1. **Community Detection**  
   - Apply the **Louvain algorithm** to detect communities within the graph.  
   - Optimize modularity to form densely connected subgraphs.

2. **Landmark Selection**  
   - Choose key nodes within each community based on **degree centrality**.  
   - These nodes act as landmarks for distance estimation.

3. **Distance Estimation**  
   - Compute distance matrices:
     - Node → Landmark  
     - Landmark → Node  
   - Estimate the shortest path between any two nodes using:
     \[
     d(x, y) \approx d(x, L) + d(L, y)
     \]
     where \( L \) is the selected landmark minimizing total distance.

4. **Evaluation**  
   - Tested on **five real-world datasets**:
     - Scientific collaborations in physics  
     - Internet AS graph  
     - Inploid network  
     - Wikipedia link dynamics  
     - DBLP authors network  

---

## 📊 Results

- The proposed approach achieved promising results, particularly on graphs with **low clustering coefficients** (simpler, less entangled structures).
- Metrics used:
  - **MAE (Mean Absolute Error)**
  - **RMSE (Root Mean Squared Error)**

| Dataset | MAE | RMSE |
|----------|------|------|
| Astrophysics | 2.0 | 2.40 |
| DBLP | 0.9 | 1.04 |
| Inploid | 0.8 | 1.01 |
| Internet Graph | 0.1 | 0.31 |
| Wikipedia | 1.4 | 1.54 |

Graphs with lower clustering coefficients yielded more accurate estimations.

---

## ⚙️ Implementation

- **Language:** Python  
- **Main Libraries:**
  - `networkx` — for graph creation and shortest path computation  
  - `community` (python-louvain) — for Louvain community detection  
  - `numpy`, `pandas`, `matplotlib` — for data handling and visualization  

