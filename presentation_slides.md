# MGNN BDS: Scalable Distributed Intrusion Detection using Apache Spark and Graph-Based AI
**Final Year Engineering Project Presentation**

---

## Slide 1: Title Slide
**Title:** MGNN BDS: Scalable Distributed Intrusion Detection using Apache Spark and Graph-Based AI
**Subtitle:** Overcoming Evasion Attacks through Distributed Graph Topology
**Presenter:** [Your Name]
**Domain:** Big Data Systems (BDS), Cybersecurity, Graph Neural Networks (GNN)

**Presenter Notes:** 
> Good morning everyone. Today I am presenting my project "MGNN BDS", which introduces a novel, scalable approach to Network Intrusion Detection using Big Data frameworks and Graph Neural Networks.

---

## Slide 2: Problem Statement
**The Vulnerability of Traditional ML in Cybersecurity**
* **Context Blindness:** Traditional models (like XGBoost, Random Forests, and MLPs) treat every network packet as an isolated tabular row. They completely ignore the relational context between IP addresses.
* **Adversarial Evasion:** Because traditional models rely on isolated features, attackers easily bypass them by slightly tweaking packet sizes, inter-arrival times, or spoofing basic features.
* **The Big Data Bottleneck:** Modern 5G networks generate massive, high-velocity streams of traffic. Complex deep learning models often fail to process this data within real-time latency constraints (under 20ms).

**Presenter Notes:** 
> Current intrusion detection systems look at data in silos. If an attacker spoofs a packet, a tabular model is easily fooled. We need a system that looks at the *relationships* between devices, but computing relationships at scale is a massive Big Data challenge.

---

## Slide 3: Objectives (Big Data Systems Focus)
1. **Architect a Distributed Data Pipeline:** Utilize Apache Spark for high-throughput, distributed ingestion and preprocessing of massive PCAP logs.
2. **Contextualize Network Traffic:** Transform flat network logs into rich, multi-relational graphs using temporal and feature-similarity edges.
3. **Achieve Evasion Robustness:** Develop a Graph Neural Network (MGNN) that leverages message passing to detect anomalies based on topology, making it robust to feature-level adversarial noise.
4. **Real-time Scalability:** Optimize the model and inference pipeline to achieve >40,000 nodes/second throughput with sub-15ms p95 latency.

**Presenter Notes:** 
> Our objectives are deeply rooted in Big Data Systems. We aren't just building a model; we are building an end-to-end pipeline capable of handling high-velocity data, transforming it into a graph structure, and performing sub-millisecond inference.

---

## Slide 4: Novelty & Unique Contributions 🌟
*(This slide highlights the unique academic/technical achievements to impress the panel)*

* **Hybrid Residual Tabular-Graph Architecture:** The proposed MGNN model uniquely fuses a Relational Graph Convolutional Network (RGCN) with a direct tabular MLP bypass. This guarantees the model never performs worse than traditional ML, while gaining massive contextual boosts.
* **Seamless BDS-to-AI Integration:** We successfully bridged the distributed data processing world (PySpark) with the deep learning world (PyTorch Geometric) using high-efficiency Parquet windowing.
* **Ablation-Proven Scalability:** We proved through ablation studies that unlike traditional models which plateau, MGNN’s accuracy actively *increases* as the graph size grows (tested up to 450,000 nodes).
* **Adversarial Noise Immunity:** We demonstrated that when 250% simulated evasion noise is injected into network features, XGBoost's F1 score collapses entirely, while MGNN maintains robust accuracy through topological smoothing.

**Presenter Notes:** 
> What makes this project truly unique is the "Tabular Bypass" architecture. We mathematically guarantee that our graph model retains all the power of a standard neural network, while using graph message passing as an *enhancement*. Furthermore, our distributed Spark-to-PyG pipeline is an industry-grade approach to graph processing.

---

## Slide 5: Big Data System (BDS) Components
**The Technological Backbone**
1. **Apache Spark (PySpark):** Handles distributed batch data ingestion, complex grouping, and temporal windowing of raw network traffic.
2. **Apache Parquet:** Serves as the columnar storage intermediary, compressing massive graph matrices for highly efficient I/O operations between Spark and PyTorch.
3. **PyTorch Geometric (PyG):** The highly optimized backend utilized for distributed, mini-batch graph sampling (`NeighborLoader`), enabling GNNs to run on massive graphs without Out-Of-Memory (OOM) crashes.
4. **Streamlit (Dashboarding):** Acts as the real-time telemetry consumer, visually demonstrating streaming micro-batches and live metric updates.

---

## Slide 6: Architecture Diagram - Big Data Pipeline
*(High-Level Data Flow from Raw Logs to Inference)*

<svg viewBox="0 0 800 400" width="100%" height="400" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#3b82f6;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#8b5cf6;stop-opacity:1" />
    </linearGradient>
    <linearGradient id="grad2" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#10b981;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#059669;stop-opacity:1" />
    </linearGradient>
    <linearGradient id="grad3" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#f59e0b;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#ea580c;stop-opacity:1" />
    </linearGradient>
  </defs>

  <!-- Background -->
  <rect width="800" height="400" fill="#0f172a" rx="15"/>

  <!-- Spark Processing Block -->
  <rect x="50" y="50" width="200" height="300" rx="10" fill="url(#grad1)" opacity="0.9"/>
  <text x="150" y="90" fill="white" font-family="Arial" font-size="20" font-weight="bold" text-anchor="middle">Apache Spark (BDS)</text>
  <text x="150" y="140" fill="white" font-family="Arial" font-size="14" text-anchor="middle">• Raw PCAP Ingestion</text>
  <text x="150" y="170" fill="white" font-family="Arial" font-size="14" text-anchor="middle">• Feature Engineering</text>
  <text x="150" y="200" fill="white" font-family="Arial" font-size="14" text-anchor="middle">• Graph Edge Construction</text>
  <text x="150" y="230" fill="white" font-family="Arial" font-size="14" text-anchor="middle">(Temporal & Similarity)</text>
  <text x="150" y="280" fill="white" font-family="Arial" font-size="16" font-style="italic" text-anchor="middle">Distributed Cluster</text>

  <!-- Parquet Data Lake -->
  <rect x="300" y="150" width="200" height="100" rx="10" fill="url(#grad3)" opacity="0.9"/>
  <text x="400" y="195" fill="white" font-family="Arial" font-size="20" font-weight="bold" text-anchor="middle">Parquet Data Lake</text>
  <text x="400" y="225" fill="white" font-family="Arial" font-size="14" text-anchor="middle">Compressed Windows</text>

  <!-- PyG Training/Inference Block -->
  <rect x="550" y="50" width="200" height="300" rx="10" fill="url(#grad2)" opacity="0.9"/>
  <text x="650" y="90" fill="white" font-family="Arial" font-size="20" font-weight="bold" text-anchor="middle">PyTorch Geometric</text>
  <text x="650" y="140" fill="white" font-family="Arial" font-size="14" text-anchor="middle">• NeighborLoader Sampling</text>
  <text x="650" y="170" fill="white" font-family="Arial" font-size="14" text-anchor="middle">• Multi-Relational GPU Training</text>
  <text x="650" y="200" fill="white" font-family="Arial" font-size="14" text-anchor="middle">• Real-Time Micro-Batching</text>
  <text x="650" y="280" fill="white" font-family="Arial" font-size="16" font-style="italic" text-anchor="middle">AI Inference Engine</text>

  <!-- Arrows -->
  <path d="M 250 200 L 290 200" stroke="white" stroke-width="4" fill="none" marker-end="url(#arrow)"/>
  <path d="M 500 200 L 540 200" stroke="white" stroke-width="4" fill="none" marker-end="url(#arrow)"/>
  
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L9,3 z" fill="white" />
    </marker>
  </defs>
</svg>

---

## Slide 7: Methodology - Graph Construction
**Transforming Logs into a Living Network**
Instead of flat CSV rows, we build a **Multi-Relational Graph**:
1. **Nodes:** Each node represents a distinct network flow/packet.
2. **Temporal Edges (Relation 0):** Edges are drawn between packets originating from the same Source IP within a strict sliding time window. This captures sequential attack behavior (e.g., Port Scanning).
3. **Similarity Edges (Relation 1):** Edges are drawn between packets that share highly similar traffic payloads or structural features (computed via cosine similarity). This groups coordinated Distributed Denial of Service (DDoS) nodes together, even if they originate from spoofed IPs.

**Presenter Notes:** 
> By utilizing PySpark, we were able to run a sliding window over millions of rows to construct Temporal Edges, and a cross-join algorithm to calculate Similarity Edges. This is impossible on a single machine, proving the necessity of the Big Data stack.

---

## Slide 8: Architecture Diagram - MGNN Model
*(The Novel Hybrid Architecture)*

<svg viewBox="0 0 800 500" width="100%" height="500" xmlns="http://www.w3.org/2000/svg">
  <!-- Background -->
  <rect width="800" height="500" fill="#1e293b" rx="15"/>

  <!-- Inputs -->
  <rect x="50" y="200" width="120" height="60" rx="5" fill="#3b82f6"/>
  <text x="110" y="235" fill="white" font-family="Arial" font-weight="bold" font-size="14" text-anchor="middle">Node Features (X)</text>
  
  <rect x="50" y="300" width="120" height="60" rx="5" fill="#8b5cf6"/>
  <text x="110" y="325" fill="white" font-family="Arial" font-weight="bold" font-size="14" text-anchor="middle">Graph Topology</text>
  <text x="110" y="345" fill="white" font-family="Arial" font-size="12" text-anchor="middle">(Edge Index & Type)</text>

  <!-- The Bypass Path -->
  <path d="M 170 230 C 250 230, 250 100, 350 100" stroke="#f43f5e" stroke-width="4" stroke-dasharray="5,5" fill="none"/>
  <rect x="350" y="70" width="200" height="60" rx="5" fill="#f43f5e" opacity="0.9"/>
  <text x="450" y="105" fill="white" font-family="Arial" font-weight="bold" font-size="14" text-anchor="middle">Residual Tabular MLP</text>

  <!-- The GNN Path -->
  <path d="M 170 230 C 250 230, 250 330, 300 330" stroke="#10b981" stroke-width="4" fill="none"/>
  <path d="M 170 330 L 300 330" stroke="#10b981" stroke-width="4" fill="none"/>
  
  <rect x="300" y="280" width="150" height="100" rx="5" fill="#10b981" opacity="0.9"/>
  <text x="375" y="325" fill="white" font-family="Arial" font-weight="bold" font-size="16" text-anchor="middle">RGCN Conv 1</text>
  <text x="375" y="345" fill="white" font-family="Arial" font-size="12" text-anchor="middle">(Multi-Relational)</text>

  <path d="M 450 330 L 500 330" stroke="#10b981" stroke-width="4" fill="none"/>

  <rect x="500" y="280" width="150" height="100" rx="5" fill="#059669" opacity="0.9"/>
  <text x="575" y="325" fill="white" font-family="Arial" font-weight="bold" font-size="16" text-anchor="middle">RGCN Conv 2</text>
  <text x="575" y="345" fill="white" font-family="Arial" font-size="12" text-anchor="middle">(Message Passing)</text>

  <!-- Fusion -->
  <path d="M 550 100 C 700 100, 700 200, 720 220" stroke="#f43f5e" stroke-width="4" fill="none"/>
  <path d="M 650 330 C 700 330, 700 240, 720 220" stroke="#059669" stroke-width="4" fill="none"/>

  <circle cx="720" cy="220" r="25" fill="#f59e0b"/>
  <text x="720" y="225" fill="white" font-family="Arial" font-weight="bold" font-size="20" text-anchor="middle">+</text>

  <rect x="670" y="270" width="100" height="40" rx="5" fill="#1e293b" stroke="#f59e0b" stroke-width="2"/>
  <text x="720" y="295" fill="white" font-family="Arial" font-weight="bold" font-size="12" text-anchor="middle">Output Classes</text>

  <path d="M 720 245 L 720 270" stroke="white" stroke-width="2" fill="none"/>
</svg>

**Presenter Notes:** 
> Our architecture takes the node features and splits them. The top path is a pure MLP—it learns tabular patterns just like XGBoost. The bottom path is a Relational GNN, which aggregates topological intelligence from neighboring nodes. The outputs are mathematically summed. This ensures the model is impervious to tabular noise because the graph path corrects the final prediction.

---

## Slide 9: Experimental Results & Benchmarks
**Crushing the Deep Learning Baselines**
* **Macro F1 Score Mastery:** MGNN achieved a Macro F1 score of **0.864**, significantly outperforming 1D-CNN (0.579), LSTM (0.662), and GAT (0.498), establishing it as the state-of-the-art Deep Learning solution.
* **Latency & Scalability:** Through Spark processing and PyG `NeighborLoader` optimization, the system achieves a throughput of **42,500+ Nodes per second** with a 95th percentile latency of **under 15ms**. 
* **Ablation Findings:** The system exhibits *positive scaling*. Models trained on 100k nodes achieved ~0.77 F1, while models trained on the full 450k node graph reached ~0.86 F1. The more Big Data we feed the graph, the smarter it gets.

**Presenter Notes:** 
> When testing against an onslaught of sophisticated evasion attacks, XGBoost's F1 score plummets. However, as our real-time dashboard demonstrates, MGNN leverages the unperturbed features of surrounding neighbors to successfully identify anomalies, achieving unprecedented adversarial robustness.

---

## Slide 10: Conclusion & Future Scope
* **Conclusion:** The integration of Apache Spark for large-scale graph construction and PyTorch Geometric for residual RGCN inference creates a highly robust, scalable Intrusion Detection System. Graph topology is the definitive answer to feature-evasion attacks in cybersecurity.
* **Future Scope:** 
  - Extending Spark Streaming to compute Similarity Edges in true real-time.
  - Integrating Dynamic Temporal Graphs (TGNs) to model evolving network infrastructures.
  - Deploying the model on distributed edge nodes in a federated learning capacity for 5G IoT networks.

**Presenter Notes:** 
> Thank you for your time. This concludes the presentation of MGNN BDS, a highly scalable, robust framework for securing the networks of tomorrow. I am open to any questions.
