"""
Enrich documents.json with categories, subcategories, document types, publication years,
authors, topics, language, and internal citation links.
"""

import json
import os

data_path = os.path.join(os.path.dirname(__file__), "..", "data", "documents.json")
with open(data_path, "r", encoding="utf-8") as f:
    docs = json.load(f)

metadata_map = {
    1: {
        "category": "Healthcare & Medicine",
        "subcategory": "Clinical Diagnostics",
        "doc_type": "Journal Article",
        "year": 2024,
        "authors": ["Dr. Elena Rostova", "Marcus Vance", "Dr. Sarah Lin"],
        "language": "English",
        "topics": ["Artificial Intelligence", "Clinical Diagnostics", "Electronic Health Records", "Triage"],
        "internal_links": [2, 3, 9]
    },
    2: {
        "category": "Healthcare & Medicine",
        "subcategory": "Intensive Care Analytics",
        "doc_type": "Conference Paper",
        "year": 2023,
        "authors": ["Dr. Aris Thorne", "Marcus Vance", "Dr. Chloe Becker"],
        "language": "English",
        "topics": ["Machine Learning", "Intensive Care", "Patient Outcome", "Vital Signs"],
        "internal_links": [1, 8, 47]
    },
    3: {
        "category": "Healthcare & Medicine",
        "subcategory": "Radiology & Imaging",
        "doc_type": "Journal Article",
        "year": 2024,
        "authors": ["Dr. Chloe Becker", "Dr. Julian Weiss", "Dr. Elena Rostova"],
        "language": "English",
        "topics": ["Deep Learning", "Medical Imaging", "Radiomics", "Oncology"],
        "internal_links": [1, 10, 21]
    },
    4: {
        "category": "Healthcare & Medicine",
        "subcategory": "Clinical NLP & Pharmacovigilance",
        "doc_type": "Conference Paper",
        "year": 2023,
        "authors": ["Dr. Maya Patel", "Dr. Aris Thorne", "Dr. David Kim"],
        "language": "English",
        "topics": ["Natural Language Processing", "Clinical Notes", "Drug Adverse Events", "Pharmacovigilance"],
        "internal_links": [1, 11, 14]
    },
    5: {
        "category": "Healthcare & Medicine",
        "subcategory": "Federated Medical Learning",
        "doc_type": "Review Paper",
        "year": 2025,
        "authors": ["Dr. Henrik Lindqvist", "Dr. Elena Rostova", "Sophia Alvarez"],
        "language": "English",
        "topics": ["Federated Learning", "Healthcare Privacy", "Differential Privacy", "Collaborative Research"],
        "internal_links": [1, 31, 35]
    },
    6: {
        "category": "Healthcare & Medicine",
        "subcategory": "Surgical Robotics",
        "doc_type": "Journal Article",
        "year": 2023,
        "authors": ["Dr. Hiroshi Tanaka", "Dr. Julian Weiss", "Liam O'Connor"],
        "language": "English",
        "topics": ["Robotic Surgery", "Computer Vision", "Reinforcement Learning", "Suturing"],
        "internal_links": [1, 23, 28]
    },
    7: {
        "category": "Healthcare & Medicine",
        "subcategory": "Genomics & Precision Diagnostics",
        "doc_type": "Journal Article",
        "year": 2024,
        "authors": ["Dr. Sarah Lin", "Dr. David Kim", "Dr. Aris Thorne"],
        "language": "English",
        "topics": ["Genomic Sequences", "Deep Neural Networks", "Rare Diseases", "Pathogenicity"],
        "internal_links": [1, 9]
    },
    8: {
        "category": "Healthcare & Medicine",
        "subcategory": "Wearable Sensor Analytics",
        "doc_type": "Conference Paper",
        "year": 2024,
        "authors": ["Liam O'Connor", "Dr. Henrik Lindqvist", "Marcus Vance"],
        "language": "English",
        "topics": ["Wearable Sensors", "Edge AI", "Cardiovascular Monitoring", "Arrhythmia"],
        "internal_links": [1, 2, 37]
    },
    9: {
        "category": "Healthcare & Medicine",
        "subcategory": "Explainable Clinical AI",
        "doc_type": "Review Paper",
        "year": 2025,
        "authors": ["Dr. Elena Rostova", "Dr. Maya Patel", "Sophia Alvarez"],
        "language": "English",
        "topics": ["Explainable AI", "Clinical Decision Support", "Trust Calibration", "SHAP"],
        "internal_links": [1, 7, 31]
    },
    10: {
        "category": "Healthcare & Medicine",
        "subcategory": "Retinopathy & Point-of-Care",
        "doc_type": "Journal Article",
        "year": 2022,
        "authors": ["Dr. Chloe Becker", "Dr. Hiroshi Tanaka", "Dr. Sarah Lin"],
        "language": "English",
        "topics": ["Diabetic Retinopathy", "Fundus Photography", "Point of Care", "Mobile Health"],
        "internal_links": [1, 3, 21]
    },

    # Cluster 2: NLP
    11: {
        "category": "Natural Language Processing",
        "subcategory": "Text Summarization & Transformers",
        "doc_type": "Conference Paper",
        "year": 2023,
        "authors": ["Dr. David Kim", "Sophia Alvarez", "Dr. Maya Patel"],
        "language": "English",
        "topics": ["Transformers", "Self-Attention", "Text Summarization", "ROUGE"],
        "internal_links": [14, 15, 18]
    },
    12: {
        "category": "Natural Language Processing",
        "subcategory": "Machine Translation",
        "doc_type": "Journal Article",
        "year": 2024,
        "authors": ["Dr. Jean-Luc Dubois", "Dr. David Kim", "Amina Bello"],
        "language": "English",
        "topics": ["Machine Translation", "Low-Resource Languages", "Cross-Lingual Transfer", "BLEU"],
        "internal_links": [11, 19]
    },
    13: {
        "category": "Natural Language Processing",
        "subcategory": "Sentiment Analysis & Opinion Mining",
        "doc_type": "Conference Paper",
        "year": 2023,
        "authors": ["Sophia Alvarez", "Dr. Maya Patel", "Carlos Mendoza"],
        "language": "English",
        "topics": ["Sentiment Analysis", "Aspect-Based Mining", "Customer Reviews", "Graph Neural Networks"],
        "internal_links": [14, 41]
    },
    14: {
        "category": "Natural Language Processing",
        "subcategory": "Scientific Information Extraction",
        "doc_type": "Journal Article",
        "year": 2024,
        "authors": ["Dr. David Kim", "Dr. Henrik Lindqvist", "Dr. Maya Patel"],
        "language": "English",
        "topics": ["Information Extraction", "Relation Extraction", "Knowledge Graphs", "Scholarly Literature"],
        "internal_links": [11, 17, 49]
    },
    15: {
        "category": "Natural Language Processing",
        "subcategory": "Retrieval-Augmented Generation",
        "doc_type": "Conference Paper",
        "year": 2025,
        "authors": ["Sophia Alvarez", "Dr. Aris Thorne", "Dr. David Kim"],
        "language": "English",
        "topics": ["Conversational Agents", "Retrieval-Augmented Generation", "LLMs", "Factuality"],
        "internal_links": [11, 16, 17]
    },
    16: {
        "category": "Natural Language Processing",
        "subcategory": "Fact-Checking & Misinformation",
        "doc_type": "Review Paper",
        "year": 2024,
        "authors": ["Amina Bello", "Carlos Mendoza", "Sophia Alvarez"],
        "language": "English",
        "topics": ["Fact-Checking", "Misinformation Detection", "Social Media", "Natural Language Inference"],
        "internal_links": [15, 34]
    },
    17: {
        "category": "Natural Language Processing",
        "subcategory": "Question Answering",
        "doc_type": "Journal Article",
        "year": 2024,
        "authors": ["Dr. David Kim", "Dr. Jean-Luc Dubois", "Dr. Julian Weiss"],
        "language": "English",
        "topics": ["Question Answering", "Knowledge Graphs", "Multi-Hop Reasoning", "Dense Passage Retrieval"],
        "internal_links": [14, 15, 26]
    },
    18: {
        "category": "Natural Language Processing",
        "subcategory": "Parameter-Efficient Fine-Tuning",
        "doc_type": "Conference Paper",
        "year": 2024,
        "authors": ["Dr. Henrik Lindqvist", "Sophia Alvarez", "Dr. David Kim"],
        "language": "English",
        "topics": ["PEFT", "LoRA", "Large Language Models", "Adapters", "GPU Efficiency"],
        "internal_links": [11, 15]
    },
    19: {
        "category": "Natural Language Processing",
        "subcategory": "Named Entity Recognition",
        "doc_type": "Conference Paper",
        "year": 2023,
        "authors": ["Amina Bello", "Dr. Jean-Luc Dubois", "Dr. Maya Patel"],
        "language": "English",
        "topics": ["Named Entity Recognition", "Zero-Shot Transfer", "Multilingual NLP", "Subword Tokenization"],
        "internal_links": [12, 14]
    },
    20: {
        "category": "Natural Language Processing",
        "subcategory": "Semantic Parsing & SQL",
        "doc_type": "Journal Article",
        "year": 2024,
        "authors": ["Carlos Mendoza", "Dr. David Kim", "Dr. Aris Thorne"],
        "language": "English",
        "topics": ["Semantic Parsing", "Natural Language Interfaces", "SQL Generation", "Relational Graphs"],
        "internal_links": [14, 17]
    },

    # Cluster 3: Computer Vision
    21: {
        "category": "Computer Vision",
        "subcategory": "Vision Transformers & Detection",
        "doc_type": "Conference Paper",
        "year": 2024,
        "authors": ["Dr. Julian Weiss", "Dr. Hiroshi Tanaka", "Liam O'Connor"],
        "language": "English",
        "topics": ["Vision Transformers", "Object Detection", "Instance Segmentation", "Feature Pyramids"],
        "internal_links": [3, 22, 27]
    },
    22: {
        "category": "Computer Vision",
        "subcategory": "Self-Supervised Learning",
        "doc_type": "Journal Article",
        "year": 2023,
        "authors": ["Dr. Julian Weiss", "Dr. Henrik Lindqvist", "Dr. Hiroshi Tanaka"],
        "language": "English",
        "topics": ["Self-Supervised Learning", "Masked Autoencoders", "Representation Learning", "Pretraining"],
        "internal_links": [21, 26, 32]
    },
    23: {
        "category": "Computer Vision",
        "subcategory": "Autonomous Driving & Tracking",
        "doc_type": "Conference Paper",
        "year": 2024,
        "authors": ["Dr. Hiroshi Tanaka", "Marcus Vance", "Dr. Julian Weiss"],
        "language": "English",
        "topics": ["Object Tracking", "Autonomous Driving", "Kalman Filters", "Embedded Compute"],
        "internal_links": [6, 21, 28]
    },
    24: {
        "category": "Computer Vision",
        "subcategory": "Super-Resolution & Satellite Imaging",
        "doc_type": "Journal Article",
        "year": 2023,
        "authors": ["Liam O'Connor", "Dr. Julian Weiss", "Dr. Chloe Becker"],
        "language": "English",
        "topics": ["Generative Adversarial Networks", "Super-Resolution", "Satellite Imagery", "Earth Observation"],
        "internal_links": [21, 25]
    },
    25: {
        "category": "Computer Vision",
        "subcategory": "Neural Radiance Fields (NeRF)",
        "doc_type": "Conference Paper",
        "year": 2024,
        "authors": ["Dr. Julian Weiss", "Liam O'Connor", "Carlos Mendoza"],
        "language": "English",
        "topics": ["Neural Radiance Fields", "3D Scene Reconstruction", "Novel View Synthesis", "Volumetric Rendering"],
        "internal_links": [21, 26]
    },
    26: {
        "category": "Computer Vision",
        "subcategory": "Vision-Language Foundation Models",
        "doc_type": "Review Paper",
        "year": 2025,
        "authors": ["Dr. Julian Weiss", "Sophia Alvarez", "Dr. David Kim"],
        "language": "English",
        "topics": ["Visual Question Answering", "Multimodal Reasoning", "Foundation Models", "Contrastive Learning"],
        "internal_links": [15, 17, 21]
    },
    27: {
        "category": "Computer Vision",
        "subcategory": "Semantic Segmentation",
        "doc_type": "Conference Paper",
        "year": 2023,
        "authors": ["Dr. Hiroshi Tanaka", "Dr. Julian Weiss", "Dr. Chloe Becker"],
        "language": "English",
        "topics": ["Semantic Segmentation", "Weakly-Supervised Learning", "Class Activation Maps", "Pseudo-Masks"],
        "internal_links": [3, 21]
    },
    28: {
        "category": "Computer Vision",
        "subcategory": "Monocular Depth Estimation",
        "doc_type": "Journal Article",
        "year": 2024,
        "authors": ["Liam O'Connor", "Dr. Hiroshi Tanaka", "Dr. Julian Weiss"],
        "language": "English",
        "topics": ["Monocular Depth Estimation", "Mobile Robotics", "Attention-Guided CNNs", "Obstacle Avoidance"],
        "internal_links": [6, 23]
    },
    29: {
        "category": "Computer Vision",
        "subcategory": "Facial Emotion Recognition",
        "doc_type": "Conference Paper",
        "year": 2023,
        "authors": ["Dr. Chloe Becker", "Dr. Maya Patel", "Carlos Mendoza"],
        "language": "English",
        "topics": ["Facial Emotion Recognition", "Action Unit Detection", "Facial Landmark Tracking", "Affective Computing"],
        "internal_links": [21, 39]
    },
    30: {
        "category": "Computer Vision",
        "subcategory": "Spatiotemporal Video Recognition",
        "doc_type": "Journal Article",
        "year": 2024,
        "authors": ["Dr. Hiroshi Tanaka", "Liam O'Connor", "Dr. Julian Weiss"],
        "language": "English",
        "topics": ["Video Action Recognition", "Activity Detection", "Spatiotemporal Transformers", "Tubelets"],
        "internal_links": [21, 23]
    },

    # Cluster 4: Cybersecurity
    31: {
        "category": "Cybersecurity & ML Defense",
        "subcategory": "Adversarial Robustness & Attacks",
        "doc_type": "Conference Paper",
        "year": 2024,
        "authors": ["Dr. Viktor Vance", "Dr. Henrik Lindqvist", "Dr. Elena Rostova"],
        "language": "English",
        "topics": ["Adversarial Robustness", "Adversarial Defense", "Perturbations", "PGD", "Randomized Smoothing"],
        "internal_links": [9, 32, 35]
    },
    32: {
        "category": "Cybersecurity & ML Defense",
        "subcategory": "Network Intrusion & Anomaly Detection",
        "doc_type": "Journal Article",
        "year": 2023,
        "authors": ["Dr. Viktor Vance", "Carlos Mendoza", "Liam O'Connor"],
        "language": "English",
        "topics": ["Network Intrusion Detection", "Anomaly Detection", "Autoencoders", "Zero-Day Threats"],
        "internal_links": [31, 33, 47]
    },
    33: {
        "category": "Cybersecurity & ML Defense",
        "subcategory": "Malware Detection & Graph Analysis",
        "doc_type": "Conference Paper",
        "year": 2024,
        "authors": ["Dr. Viktor Vance", "Dr. Hiroshi Tanaka", "Carlos Mendoza"],
        "language": "English",
        "topics": ["Malware Detection", "Dynamic Analysis", "Graph Neural Networks", "Ransomware"],
        "internal_links": [32, 36, 41]
    },
    34: {
        "category": "Cybersecurity & ML Defense",
        "subcategory": "Phishing & URL Detection",
        "doc_type": "Journal Article",
        "year": 2023,
        "authors": ["Amina Bello", "Dr. Viktor Vance", "Marcus Vance"],
        "language": "English",
        "topics": ["Phishing Detection", "Malicious Domains", "Ensemble Learning", "URL Scoring"],
        "internal_links": [16, 32, 48]
    },
    35: {
        "category": "Cybersecurity & ML Defense",
        "subcategory": "Privacy Attacks & Membership Inference",
        "doc_type": "Conference Paper",
        "year": 2024,
        "authors": ["Dr. Henrik Lindqvist", "Dr. Viktor Vance", "Sophia Alvarez"],
        "language": "English",
        "topics": ["Membership Inference", "Privacy Attacks", "Differential Privacy", "Shadow Models"],
        "internal_links": [5, 31]
    },
    36: {
        "category": "Cybersecurity & ML Defense",
        "subcategory": "Code Vulnerability Detection",
        "doc_type": "Journal Article",
        "year": 2024,
        "authors": ["Dr. Viktor Vance", "Dr. David Kim", "Carlos Mendoza"],
        "language": "English",
        "topics": ["Vulnerability Detection", "Source Code Analysis", "Code Property Graphs", "CVE"],
        "internal_links": [14, 33, 40]
    },
    37: {
        "category": "Cybersecurity & ML Defense",
        "subcategory": "Hardware Trojan Detection",
        "doc_type": "Conference Paper",
        "year": 2023,
        "authors": ["Liam O'Connor", "Dr. Viktor Vance", "Dr. Hiroshi Tanaka"],
        "language": "English",
        "topics": ["Hardware Trojans", "Security Verification", "Side-Channel Power", "Semiconductor Security"],
        "internal_links": [8, 38]
    },
    38: {
        "category": "Cybersecurity & ML Defense",
        "subcategory": "Cryptographic Side-Channels",
        "doc_type": "Journal Article",
        "year": 2024,
        "authors": ["Dr. Henrik Lindqvist", "Liam O'Connor", "Dr. Viktor Vance"],
        "language": "English",
        "topics": ["Cryptographic Key Recovery", "Side-Channel Attacks", "Deep Learning", "AES Encryption"],
        "internal_links": [37]
    },
    39: {
        "category": "Cybersecurity & ML Defense",
        "subcategory": "Behavioral Biometrics & Zero-Trust",
        "doc_type": "Review Paper",
        "year": 2025,
        "authors": ["Dr. Viktor Vance", "Dr. Chloe Becker", "Marcus Vance"],
        "language": "English",
        "topics": ["Zero-Trust Architecture", "Behavioral Biometrics", "Continuous Authentication", "Keystroke Dynamics"],
        "internal_links": [29, 32]
    },
    40: {
        "category": "Cybersecurity & ML Defense",
        "subcategory": "Smart Contract Auditing",
        "doc_type": "Conference Paper",
        "year": 2024,
        "authors": ["Carlos Mendoza", "Dr. Viktor Vance", "Sophia Alvarez"],
        "language": "English",
        "topics": ["Smart Contracts", "Symbolic Execution", "Blockchain Security", "Ethereum", "Vulnerability Auditing"],
        "internal_links": [36]
    },

    # Cluster 5: Data Mining
    41: {
        "category": "Data Mining & Recommender Systems",
        "subcategory": "Collaborative Filtering & Graph Models",
        "doc_type": "Conference Paper",
        "year": 2024,
        "authors": ["Dr. Aris Thorne", "Carlos Mendoza", "Amina Bello"],
        "language": "English",
        "topics": ["Graph Neural Networks", "Collaborative Filtering", "Recommender Systems", "Bipartite Graphs"],
        "internal_links": [13, 42, 49]
    },
    42: {
        "category": "Data Mining & Recommender Systems",
        "subcategory": "Session-Based Recommenders",
        "doc_type": "Journal Article",
        "year": 2023,
        "authors": ["Dr. Aris Thorne", "Dr. Maya Patel", "Carlos Mendoza"],
        "language": "English",
        "topics": ["Session-Based Recommendation", "Self-Attention", "E-Commerce", "Intent Prediction"],
        "internal_links": [11, 41, 49]
    },
    43: {
        "category": "Data Mining & Recommender Systems",
        "subcategory": "Multi-Armed Bandits & Ad Tech",
        "doc_type": "Conference Paper",
        "year": 2024,
        "authors": ["Marcus Vance", "Dr. Aris Thorne", "Sophia Alvarez"],
        "language": "English",
        "topics": ["Multi-Armed Bandits", "Exploration-Exploitation", "Online Advertising", "Thompson Sampling"],
        "internal_links": [41, 48]
    },
    44: {
        "category": "Data Mining & Recommender Systems",
        "subcategory": "Association Rule Mining",
        "doc_type": "Journal Article",
        "year": 2023,
        "authors": ["Dr. Aris Thorne", "Liam O'Connor", "Amina Bello"],
        "language": "English",
        "topics": ["Frequent Itemset Mining", "Association Rules", "FP-Growth", "High-Dimensional Data"],
        "internal_links": [41, 46]
    },
    45: {
        "category": "Data Mining & Recommender Systems",
        "subcategory": "Spatiotemporal & Traffic Forecasting",
        "doc_type": "Conference Paper",
        "year": 2024,
        "authors": ["Carlos Mendoza", "Dr. Hiroshi Tanaka", "Dr. Aris Thorne"],
        "language": "English",
        "topics": ["Traffic Flow Forecasting", "Graph Convolutional Networks", "Spatiotemporal Modeling", "Intelligent Transportation"],
        "internal_links": [23, 41, 47]
    },
    46: {
        "category": "Data Mining & Recommender Systems",
        "subcategory": "Scalable Clustering",
        "doc_type": "Journal Article",
        "year": 2024,
        "authors": ["Dr. Henrik Lindqvist", "Dr. Aris Thorne", "Marcus Vance"],
        "language": "English",
        "topics": ["Clustering", "K-Means", "Density Peaks", "Massive Datasets", "Approximate Nearest Neighbors"],
        "internal_links": [44, 47]
    },
    47: {
        "category": "Data Mining & Recommender Systems",
        "subcategory": "Time-Series Outlier Detection",
        "doc_type": "Conference Paper",
        "year": 2023,
        "authors": ["Dr. Aris Thorne", "Dr. Viktor Vance", "Carlos Mendoza"],
        "language": "English",
        "topics": ["Outlier Detection", "Time Series", "Isolation Forests", "Autoencoders", "Sensor Telemetry"],
        "internal_links": [2, 32, 46]
    },
    48: {
        "category": "Data Mining & Recommender Systems",
        "subcategory": "Customer Churn Analytics",
        "doc_type": "Journal Article",
        "year": 2024,
        "authors": ["Amina Bello", "Dr. Aris Thorne", "Marcus Vance"],
        "language": "English",
        "topics": ["Customer Churn", "Lifetime Value", "Gradient Boosted Trees", "Survival Analysis"],
        "internal_links": [34, 43, 50]
    },
    49: {
        "category": "Data Mining & Recommender Systems",
        "subcategory": "Cold-Start & Cross-Domain Recommenders",
        "doc_type": "Conference Paper",
        "year": 2025,
        "authors": ["Dr. Aris Thorne", "Dr. David Kim", "Sophia Alvarez"],
        "language": "English",
        "topics": ["Cold-Start Recommendations", "Knowledge Graphs", "Cross-Domain Transfer", "Translational Embeddings"],
        "internal_links": [14, 41, 42]
    },
    50: {
        "category": "Data Mining & Recommender Systems",
        "subcategory": "AutoML & Feature Engineering",
        "doc_type": "Review Paper",
        "year": 2025,
        "authors": ["Dr. Aris Thorne", "Dr. Henrik Lindqvist", "Carlos Mendoza"],
        "language": "English",
        "topics": ["Automated Feature Engineering", "Model Selection", "Tabular Mining", "AutoML", "Bayesian Optimization"],
        "internal_links": [48, 49]
    }
}

for doc in docs:
    d_id = doc["id"]
    if d_id in metadata_map:
        doc.update(metadata_map[d_id])

with open(data_path, "w", encoding="utf-8") as f:
    json.dump(docs, f, indent=2, ensure_ascii=False)

print(f"Successfully enriched {len(docs)} documents in {data_path}")
