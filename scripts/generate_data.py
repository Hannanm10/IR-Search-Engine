import json
import os

docs = [
    # Cluster 1: AI in Healthcare (1-10)
    {
        "id": 1,
        "title": "Artificial Intelligence in Healthcare and Clinical Diagnostics",
        "abstract": "Artificial intelligence models can assist clinicians in detecting disease patterns from medical records and imaging. Machine learning algorithms improve diagnostic accuracy and reduce diagnostic turnaround time. This study evaluates AI integration within hospital health systems.",
        "methodology": "We evaluated deep neural networks on electronic health records across four metropolitan hospitals. Data was normalized and preprocessed to handle missing patient attributes. Cross-validation was performed to avoid clinical bias.",
        "results": "The automated diagnostic pipeline achieved an AUC of 0.94 across major diagnostic categories. Clinicians reported a thirty percent reduction in triage evaluation latency."
    },
    {
        "id": 2,
        "title": "Machine Learning for Patient Outcome Prediction in Intensive Care",
        "abstract": "Predicting patient deterioration in intensive care units is vital for timely clinical interventions. Supervised machine learning algorithms analyze continuous physiological telemetry to forecast adverse events. Early alerts allow intensive care teams to optimize emergency resource allocation.",
        "methodology": "Gradient boosting models and recurrent networks were trained on multi-parameter vital signs collected every minute. Feature engineering captured heart rate variability and arterial pressure trends. Missing sensor readings were imputed using spline interpolation.",
        "results": "The predictive model detected clinical decompensation six hours prior to onset with 88 percent sensitivity. False alarm rates decreased by 24 percent compared to standard scoring systems."
    },
    {
        "id": 3,
        "title": "Deep Learning Applications in Medical Imaging and Radiomics",
        "abstract": "Automated radiomic feature extraction from computed tomography scans enables early cancer detection. Deep convolutional networks identify subtle pathological lesions that may escape human visual screening. This paper outlines an end-to-end framework for radiomic classification.",
        "methodology": "A 3D convolutional neural network architecture was trained on over twenty thousand annotated thoracic CT scans. Data augmentation included elastic deformations and affine transformations. Gradient-weighted class activation mapping was used for visual interpretability.",
        "results": "The network demonstrated a sensitivity of 92.4 percent and a specificity of 91.1 percent on independent validation sets. Heatmaps accurately localized pulmonary nodules matching board-certified radiologist markings."
    },
    {
        "id": 4,
        "title": "Natural Language Processing on Clinical Notes for Drug Adverse Event Detection",
        "abstract": "Unstructured electronic clinical notes contain critical signals of adverse drug reactions. Applying natural language processing allows automated extraction of drug-symptom associations from doctor narratives. This accelerates pharmacovigilance and enhances patient safety surveillance.",
        "methodology": "We fine-tuned domain-specific transformer models on anonymized medical discharge summaries. Named entity recognition identified medications and physiological symptoms simultaneously. Relation extraction classified causal dependencies between treatments and reactions.",
        "results": "The system extracted adverse drug reactions with an F1-score of 0.86, outperforming legacy rule-based parsers. The model identified twelve previously underreported adverse events in post-market monitoring."
    },
    {
        "id": 5,
        "title": "Federated Learning for Privacy-Preserving Collaborative Medical Research",
        "abstract": "Sharing sensitive patient health records across institutions is heavily constrained by privacy regulations. Federated learning enables multi-center machine learning model training without centralizing patient data. This research assesses federated optimization across heterogeneous medical nodes.",
        "methodology": "We implemented federated averaging across five simulated hospital clusters with differential privacy guarantees. Models shared encrypted gradient updates rather than raw patient records. Secure aggregation protocols prevented gradient leakage attacks.",
        "results": "The federated model achieved 98 percent of the accuracy of a centrally trained benchmark model. Communication overhead was reduced by forty percent using selective parameter quantization."
    },
    {
        "id": 6,
        "title": "Robotic Surgery Assistance using Computer Vision and Reinforcement Learning",
        "abstract": "Autonomous surgical robotic instruments require real-time scene understanding and millimeter-precision motor control. Combining computer vision with reinforcement learning enables robots to assist surgeons during minimally invasive laparoscopy. Visual tracking guides suture placement safely.",
        "methodology": "Stereo endoscopic video was analyzed using optical flow and spatial attention networks to segment surgical tools. Reinforcement learning agents learned policy trajectories in simulated soft-tissue environments. Trajectory smoothness was enforced using dynamic constraint penalties.",
        "results": "Autonomous suturing accuracy improved by 35 percent compared to manual teleoperation under simulated latency. Tissue trauma indicators decreased significantly during robotic needle insertion trials."
    },
    {
        "id": 7,
        "title": "Genomic Sequence Analysis with Deep Neural Networks for Rare Disease Discovery",
        "abstract": "Interpreting non-coding genomic variants remains a major bottleneck in rare genetic disease diagnosis. Deep neural networks learn regulatory motifs directly from raw DNA sequences. Automated variant pathogenicity scoring improves precision medicine pipelines.",
        "methodology": "Convolutional and dilated causal architectures were trained on whole-genome sequencing cohorts from rare disease consortia. Epigenomic markers and chromatin accessibility assays served as multi-task prediction targets. In silico mutagenesis evaluated nucleotide substitution impacts.",
        "results": "The variant classifier discovered twenty-three candidate pathogenic non-coding mutations validated by functional assays. Precision on pathogenic single nucleotide variants reached 89 percent."
    },
    {
        "id": 8,
        "title": "Wearable Sensor Analytics and Edge AI for Remote Cardiovascular Monitoring",
        "abstract": "Continuous cardiac telemetry using lightweight wearable sensors enables proactive management of cardiovascular disease. Deploying compact neural network models directly onto wearable microcontrollers allows real-time arrhythmia detection. Local processing preserves patient confidentiality and battery life.",
        "methodology": "Quantized one-dimensional convolutional networks were deployed onto ARM Cortex processors. Continuous electrocardiogram streams were filtered to eliminate baseline wander and motion artifacts. An adaptive threshold triggered event-driven data transmission to cloud backends.",
        "results": "The edge device detected atrial fibrillation with 96.2 percent accuracy while consuming under 15 milliwatts of operating power. Battery lifespan on continuous telemetry extended to fourteen days."
    },
    {
        "id": 9,
        "title": "Explainable Artificial Intelligence for Clinical Decision Support Systems",
        "abstract": "Opaque black-box models face resistance from healthcare providers demanding transparent rationales. Explainable artificial intelligence frameworks generate human-interpretable justifications for clinical recommendations. Trust calibration between clinicians and AI systems is essential for adoption.",
        "methodology": "We combined tree-based ensemble models with Shapley additive explanations and counterfactual reasoning algorithms. Clinicians engaged in blind user studies comparing standard model scores against explained score breakdowns. Qualitative feedback assessed confidence in diagnostic support.",
        "results": "Clinical decision adoption rates increased by 42 percent when explainability interfaces were presented alongside predictions. Clinicians detected and corrected erroneous edge-case model recommendations in ninety percent of test cases."
    },
    {
        "id": 10,
        "title": "Automated Diabetic Retinopathy Screening using Mobile Fundus Photography",
        "abstract": "Diabetic retinopathy is a leading cause of preventable blindness worldwide, especially in underserved rural regions. Mobile fundus cameras combined with on-device computer vision enable accessible point-of-care screening. Rapid triage identifies patients requiring urgent ophthalmological intervention.",
        "methodology": "A lightweight MobileNet architecture was fine-tuned on sixty thousand retinal fundus images graded by expert ophthalmologists. Images underwent color normalization and macular centering. An offline mobile application was deployed to rural community health clinics.",
        "results": "Field trials demonstrated 93.8 percent sensitivity for referable diabetic retinopathy. Screening throughput in rural clinics increased threefold without requiring on-site specialist presence."
    },

    # Cluster 2: Natural Language Processing (11-20)
    {
        "id": 11,
        "title": "Transformer Architectures and Self-Attention Mechanisms for Text Summarization",
        "abstract": "Abstractive document summarization requires synthesizing core narrative arguments while maintaining factual consistency. Transformer models utilizing multi-head self-attention capture long-range contextual relationships across extensive documents. This paper examines architectural variations for factual summarization.",
        "methodology": "We evaluated encoder-decoder transformers equipped with sparse attention across news article benchmarks. Contrastive loss penalties were introduced to penalize hallucinated statements not present in the source text. Factuality was verified using dependency tree alignment.",
        "results": "The factual consistency score of generated abstracts improved by 28 percent over standard seq2seq models. ROUGE-1 and ROUGE-L metrics reached 47.2 and 43.8 respectively on benchmark corpora."
    },
    {
        "id": 12,
        "title": "Low-Resource Machine Translation via Cross-Lingual Knowledge Transfer",
        "abstract": "Neural machine translation performs exceptionally well for language pairs with abundant parallel corpora but fails on low-resource languages. Cross-lingual transfer learning leverages multilingual representations to bridge data scarcity. Shared semantic representations allow zero-shot and few-shot translation.",
        "methodology": "We pretrained a massively multilingual sequence-to-sequence model on one hundred languages using masked language modeling. Target low-resource language pairs were fine-tuned with back-translation and lexical adapter layers. Word alignment was regularized using semantic similarity objectives.",
        "results": "BLEU scores for five under-resourced African languages increased by an average of 7.4 points over baseline bilingual baselines. Zero-shot transfer yielded intelligible translations on colloquial phrase datasets."
    },
    {
        "id": 13,
        "title": "Aspect-Based Sentiment Analysis and Opinion Mining on Customer Reviews",
        "abstract": "Fine-grained aspect-based sentiment analysis identifies distinct sentiments associated with specific product features rather than overall polarity. Understanding customer sentiment at the attribute level provides actionable product insights for manufacturers. Graph neural networks model syntactic dependencies between aspect targets and opinion terms.",
        "methodology": "Dependency parse trees were converted into graph convolutional networks combined with contextualized embeddings. Aspect boundaries and polarity terms were jointly tagged using sequential conditional random fields. Experiments spanned restaurant and consumer electronic review corpora.",
        "results": "The graph-augmented model reached 84.5 percent aspect classification accuracy across benchmark datasets. Error analysis demonstrated superior handling of complex sentences with contrasting multi-aspect polarities."
    },
    {
        "id": 14,
        "title": "Information Extraction and Relation Extraction from Scientific Literature",
        "abstract": "The exponential growth of scholarly literature makes automated extraction of scientific discoveries imperative for researchers. We present an end-to-end information extraction system that extracts scientific entities, experimental metrics, and scholarly relations. Structured knowledge graphs are constructed automatically from scientific PDFs.",
        "methodology": "We parsed scientific LaTeX and PDF sources into structured sections and applied entity linking against academic ontologies. A transformer-based span classifier predicted directional relations including method-used and metric-evaluated. Global graph consistency was enforced via integer linear programming.",
        "results": "The scientific relation extraction achieved an F1-score of 81.3 percent on domain benchmarks. Over five hundred thousand structured relationship triples were populated into an open scholarly knowledge base."
    },
    {
        "id": 15,
        "title": "Conversational Dialogue Agents with Retrieval-Augmented Generation",
        "abstract": "Large language models often generate outdated or hallucinated information when responding to specialized user queries. Retrieval-augmented generation combines neural information retrieval with parametric language generation to anchor conversations in factual truth. Grounded dialogue systems improve user trust and knowledge verification.",
        "methodology": "A dense vector index over an authoritative knowledge corpus was coupled to an instruction-tuned generative model. Top documents were dynamically retrieved using approximate nearest neighbor search and injected into prompt contexts. Fact verification filters pruned unsubstantiated generative outputs.",
        "results": "Human evaluation confirmed a 46 percent reduction in factual hallucinations compared to pure parametric generative models. Dialogue coherence and conversational engagement metrics remained equally high."
    },
    {
        "id": 16,
        "title": "Automated Fact-Checking and Misinformation Detection on Social Media",
        "abstract": "The viral spread of misinformation across social media networks poses significant societal challenges. Automated fact-checking systems verify claims against reliable journalistic repositories and authoritative archives. Multi-modal evidence synthesis strengthens classification of misleading textual and visual posts.",
        "methodology": "We designed a claim verification architecture that retrieves supporting and refuting evidence from verified fact-checking portals. Natural language inference models determined entailment or contradiction between retrieved evidence passages and contested claims. Graph neural networks modeled social propagation dynamics.",
        "results": "The automated verification pipeline achieved an accuracy of 87.1 percent in identifying debunked rumors within two hours of first emergence. Early detection prevented further viral amplification in controlled tests."
    },
    {
        "id": 17,
        "title": "Question Answering over Knowledge Graphs and Unstructured Documents",
        "abstract": "Complex question answering requires multi-hop reasoning across both structured knowledge graphs and unstructured textual paragraphs. Hybrid retrieval architectures bridge the gap between symbolic triple queries and vector semantic text search. This framework answers complex comparative and quantitative queries.",
        "methodology": "Questions were parsed into latent logical execution plans while simultaneously encoding semantic question embeddings. Candidate entity nodes in the knowledge graph were traversed in parallel with dense passage retrieval from associated Wikipedia articles. A joint neural re-ranker scored candidate answer spans.",
        "results": "On complex multi-hop benchmark benchmarks, the hybrid system scored 73.4 percent exact match accuracy. The approach outperformed standalone knowledge graph parsers by 14 percent on questions requiring recent textual context."
    },
    {
        "id": 18,
        "title": "Parameter-Efficient Fine-Tuning of Large Language Models via Adapters and LoRA",
        "abstract": "Fine-tuning billion-parameter language models for downstream domain tasks is computationally prohibitive for most academic organizations. Parameter-efficient fine-tuning techniques freeze base weights and optimize low-rank adapter matrices. This democratizes language model adaptation while preserving base capabilities.",
        "methodology": "We systematically benchmarked Low-Rank Adaptation (LoRA), prefix tuning, and bottleneck adapters across eight NLP benchmarks. Memory footprints and training throughput were logged across diverse GPU hardware configurations. Catastrophic forgetting was quantified using general benchmark suites.",
        "results": "LoRA retained 99.2 percent of full fine-tuning performance while updating less than 0.5 percent of total model parameters. Memory requirements during training decreased by sixty-five percent, enabling execution on commodity hardware."
    },
    {
        "id": 19,
        "title": "Cross-Lingual Named Entity Recognition using Zero-Shot Transformer Models",
        "abstract": "Recognizing named entities such as people, organizations, and locations across multilingual corpora is essential for global intelligence systems. Training dedicated models for every language is impractical due to annotated dataset deficits. Zero-shot cross-lingual transfer enables models trained in English to extract entities in foreign languages.",
        "methodology": "We fine-tuned multilingual transformer representations on English entity corpora using subword tokenization and token classification heads. Target languages without training annotations were evaluated directly using zero-shot inference. Feature representations across multilingual layers were inspected via canonical correlation analysis.",
        "results": "The zero-shot system obtained an average F1-score of 78.6 percent across twelve typologically diverse target languages. Languages sharing syntactic typologies with English showed minimal transfer degradation."
    },
    {
        "id": 20,
        "title": "Semantic Parsing for Natural Language Interfaces to Databases and SQL",
        "abstract": "Translating conversational natural language queries into executable SQL queries allows non-technical users to access relational databases. Semantic parsing models must handle complex schema layouts, nested subqueries, and database join conditions. Schema-aware encoding ensures syntactically valid and executable database operations.",
        "methodology": "We implemented a relational graph transformer that jointly encodes database table schemas and natural language question tokens. Constrained decoding grammars enforced SQL syntactic validity during autoregressive generation. Intermediate representation representations bridged dialect disparities.",
        "results": "Execution accuracy reached 79.8 percent on the challenging Spider benchmark dataset. Generated SQL queries demonstrated zero syntax errors due to grammar-guided decoding constraints."
    },

    # Cluster 3: Computer Vision (21-30)
    {
        "id": 21,
        "title": "Vision Transformers for High-Resolution Object Detection and Segmentation",
        "abstract": "Vision transformers have emerged as a formidable alternative to standard convolutional neural networks for visual recognition tasks. Applying self-attention across image patch sequences captures global contextual relationships effectively. This study explores hierarchical vision transformers for dense pixel-level object segmentation.",
        "methodology": "A hierarchical vision transformer backbone with shifted window self-attention was integrated with a feature pyramid network. Multi-scale feature representations were trained on public benchmark image datasets with bounding box and instance mask annotations. Focal loss addressed extreme foreground-background class imbalance.",
        "results": "The proposed architecture achieved 54.2 box mean Average Precision (mAP) and 48.1 mask mAP on standard benchmarks. Inference speed improved by twenty percent compared to standard non-hierarchical vision transformer baselines."
    },
    {
        "id": 22,
        "title": "Self-Supervised Visual Representation Learning with Masked Autoencoders",
        "abstract": "Supervised training of visual models requires massive annotated image datasets that are expensive and labor-intensive to curate. Masked autoencoders provide a scalable self-supervised learning paradigm by reconstructing randomly masked visual patches. Latent representations learned through visual reconstruction transfer effectively to downstream perception tasks.",
        "methodology": "We masked seventy-five percent of input image patches and trained an asymmetric autoencoder to reconstruct missing pixel values. The encoder operated only on visible patches, greatly reducing computational overhead. Pretrained representations were fine-tuned for classification, detection, and segmentation.",
        "results": "Models pretrained with masked autoencoders reached 85.9 percent top-1 accuracy on ImageNet without external data. Fine-tuning convergence on downstream tasks was three times faster than training from random weight initialization."
    },
    {
        "id": 23,
        "title": "Real-Time Object Tracking in Autonomous Driving using Kalman Filters and Deep Features",
        "abstract": "Robust multiple object tracking in dynamic urban traffic environments is essential for collision avoidance in autonomous vehicles. Tracking systems must maintain continuous object identities despite occlusions, abrupt motion, and sensor noise. Combining deep appearance features with kinematic Kalman filtering ensures tracking accuracy and real-time execution.",
        "methodology": "A lightweight object detector generated bounding boxes from high-frame-rate automotive camera streams. Deep metric embeddings characterized target appearance while an extended Kalman filter estimated kinematic velocities. Bipartite graph matching resolved data association across successive frames.",
        "results": "The tracking pipeline operated at 42 frames per second on automotive embedded compute units. Multiple Object Tracking Accuracy (MOTA) reached 76.5 percent on public autonomous driving benchmarks with minimal ID switches."
    },
    {
        "id": 24,
        "title": "Generative Adversarial Networks for Super-Resolution in Satellite Imagery",
        "abstract": "Low-resolution Earth observation imagery limits the effectiveness of environmental monitoring, disaster response, and urban planning. Super-resolution techniques synthesize high-frequency spatial textures and sharp object boundaries from low-resolution aerial captures. Generative adversarial networks produce photorealistic aerial imagery enhancements.",
        "methodology": "We trained a conditional generative adversarial network featuring residual-in-residual dense blocks and perceptual loss terms. The discriminator evaluated realism at multiple spatial scales against paired high-resolution aerial reference tiles. Spectral consistency loss prevented radiometric distortions.",
        "results": "The model achieved a 4x spatial resolution increase while preserving geographic features and road networks. Structural similarity index (SSIM) improved by 18 percent compared to bicubic interpolation and standard mean-squared error baselines."
    },
    {
        "id": 25,
        "title": "3D Scene Reconstruction and Novel View Synthesis with Neural Radiance Fields",
        "abstract": "Synthesizing novel camera perspectives of complex 3D scenes from sparse sets of 2D photographs is a foundational graphics challenge. Neural Radiance Fields (NeRF) represent continuous volumetric scenes as coordinate-based neural functions. Volumetric rendering equations synthesize photorealistic views with accurate lighting and reflections.",
        "methodology": "Multilayer perceptrons mapped 5D spatial coordinates and viewing angles to volume density and emitted radiance. Hierarchical volume sampling allocated network queries efficiently along camera rays. Positional encoding enabled the network to capture high-frequency geometry and surface textures.",
        "results": "The neural rendering framework synthesized novel views with peak signal-to-noise ratio (PSNR) exceeding 32 decibels. Rendering artifacts around complex semi-transparent geometry and reflective surfaces were visibly suppressed."
    },
    {
        "id": 26,
        "title": "Visual Question Answering and Multimodal Reasoning using Vision-Language Foundation Models",
        "abstract": "Visual question answering demands that computer vision models reason over both image content and complex natural language inquiries. Vision-language foundation models align visual feature embeddings with linguistic representations in a shared semantic space. This enables cross-modal reasoning, visual commonsense inference, and open-ended dialogue.",
        "methodology": "We pre-trained a dual-encoder transformer architecture on hundreds of millions of image-text pairs using contrastive learning. A multimodal fusion decoder performed cross-attention between linguistic query tokens and visual patch representations. The model was evaluated on standard benchmark visual reasoning tasks.",
        "results": "Accuracy on the VQA v2 benchmark reached 78.4 percent, surpassing previous modular neuro-symbolic baselines. The model exhibited impressive zero-shot generalization when answering visual questions about unfamiliar object categories."
    },
    {
        "id": 27,
        "title": "Weakly-Supervised Semantic Segmentation with Class Activation Maps",
        "abstract": "Pixel-level semantic segmentation annotations are exceptionally costly to collect, limiting dataset scalability for computer vision projects. Weakly-supervised approaches leverage image-level category labels to infer dense semantic boundaries. Class activation mapping generates initial pseudo-masks that guide iterative segmentation refinement.",
        "methodology": "We extracted class activation maps from classification convolutional networks and refined them using affinity matrices derived from image features. Saliency maps and random walk propagation expanded seed regions into full object masks. A segmentation network was then trained on the synthesized pseudo-ground truth.",
        "results": "The weakly-supervised pipeline reached 68.7 percent mean Intersection over Union (mIoU) on benchmark segmentation sets. The performance reached 91 percent of the accuracy of fully-supervised models trained on manual pixel annotations."
    },
    {
        "id": 28,
        "title": "Monocular Depth Estimation for Mobile Robotics using Attention-Guided CNNs",
        "abstract": "Estimating pixel-wise metric depth from a single RGB camera is a cost-effective alternative to expensive LiDAR sensors for mobile robots. Monocular depth estimation is an ill-posed problem requiring strong geometric and semantic scene priors. Attention-guided convolutional networks capture global context to resolve depth ambiguities.",
        "methodology": "An encoder-decoder architecture with channel and spatial attention modules was trained on indoor and outdoor robotics datasets. Scale-invariant loss and edge-aware smoothness penalties enforced depth consistency across flat planes and object edges. The model was optimized for deployment on embedded edge robots.",
        "results": "Absolute relative error dropped to 0.089 on benchmark indoor datasets. The depth prediction pipeline ran at 30 frames per second on an onboard robotic processor, enabling obstacle avoidance in real time."
    },
    {
        "id": 29,
        "title": "Facial Emotion Recognition and Action Unit Detection in Natural Environments",
        "abstract": "Automated recognition of facial expressions in uncontrolled natural environments is complicated by head pose variations, lighting shifts, and partial occlusions. Recognizing facial action units according to anatomical facial coding systems enables objective emotion classification. Deep spatial-temporal networks capture subtle facial muscle movements.",
        "methodology": "We built a multi-task convolutional network that simultaneously detects facial landmarks, action units, and discrete emotional states. Temporal recurrent units modeled dynamic expression transitions across continuous video sequences. Training incorporated face alignment and illumination equalization preprocessing.",
        "results": "Recognition accuracy on seven basic emotion classes reached 88.2 percent on in-the-wild facial benchmarks. Action unit detection F1-score averaged 74.5 percent under challenging illumination and dynamic head movement conditions."
    },
    {
        "id": 30,
        "title": "Video Action Recognition and Activity Detection with Spatiotemporal Transformers",
        "abstract": "Recognizing human activities in continuous video streams requires modeling both spatial posture cues and temporal motion sequences. Traditional 3D convolutional networks are computationally demanding and struggle to capture distant temporal dependencies. Spatiotemporal transformers process video tubelets with factorized space-time self-attention.",
        "methodology": "Input video clips were tokenized into non-overlapping spatiotemporal patches. The architecture decoupled spatial attention across frames and temporal attention across time steps to minimize compute requirements. Pretraining on large video datasets preceded task-specific activity classification.",
        "results": "Top-1 accuracy on the Kinetics-400 video benchmark reached 82.5 percent, exceeding heavier 3D CNN architectures. Model inference required forty percent fewer floating-point operations during batch processing."
    },

    # Cluster 4: Cybersecurity & ML (31-40)
    {
        "id": 31,
        "title": "Adversarial Robustness and Defense Mechanisms in Deep Learning Classifiers",
        "abstract": "Deep neural networks are vulnerable to imperceptible adversarial perturbations that cause erroneous classifications with high confidence. Designing certified defense mechanisms is paramount for deploying machine learning models in security-critical environments. Adversarial training regularizes decision boundaries against worst-case input perturbations.",
        "methodology": "We investigated projected gradient descent (PGD) and fast gradient sign method (FGSM) attacks across deep classifiers. Models were fortified using robust optimization objectives and randomized smoothing techniques. Certified robustness bounds were analytically computed using convex relaxations.",
        "results": "Robust classification accuracy under white-box PGD attacks improved from 12 percent to 67 percent across image benchmarks. Certified bounds guaranteed non-evasive predictions within defined perturbation radii for ninety percent of inputs."
    },
    {
        "id": 32,
        "title": "Network Intrusion Detection and Anomaly Classification via Autoencoders",
        "abstract": "Modern enterprise networks face sophisticated zero-day cyber threats that evade traditional signature-based detection systems. Machine learning anomaly detection flags abnormal network traffic patterns indicative of intrusions. Unsupervised autoencoders learn baseline traffic profiles without requiring historical attack labels.",
        "methodology": "Deep variational autoencoders were trained on normal NetFlow connection records containing packet counts, duration, and protocol flags. High reconstruction errors during testing signaled anomalous network behavior. An ensemble of isolation forests cross-verified high-reconstruction alerts.",
        "results": "The anomaly detection framework achieved a 95.8 percent true positive detection rate on modern intrusion benchmarks. False positive rates remained below 1.2 percent, preventing security analyst alert fatigue."
    },
    {
        "id": 33,
        "title": "Malware Detection and Dynamic Analysis using Graph Neural Networks",
        "abstract": "Polymorphic malware authors evade static file signature scanners through obfuscation, packing, and encryption techniques. Constructing behavioral control flow and function call graphs enables robust dynamic malware classification. Graph neural networks classify executable intent independently of low-level syntactic obfuscation.",
        "methodology": "Binaries were executed inside instrumented sandboxes to log system calls, API invocations, and memory operations. Execution traces were converted into directed behavior graphs with system calls as nodes and temporal invocations as edges. Graph convolutional networks classified malware families.",
        "results": "Classification accuracy across twenty distinct malware families reached 97.4 percent. The graph approach successfully identified sixty obfuscated zero-day ransomware samples that bypassed commercial antivirus software."
    },
    {
        "id": 34,
        "title": "Phishing URL and Malicious Domain Identification with Ensemble Learning",
        "abstract": "Phishing attacks remain a prevalent vector for credential theft, corporate espionage, and financial fraud across the web. Identifying malicious URLs before users interact with fraudulent pages is vital for browser security. Feature engineering combined with ensemble machine learning identifies suspicious domain patterns.",
        "methodology": "We extracted lexical URL characteristics, DNS registration metadata, WHOIS records, and SSL certificate features. Random forests and gradient-boosted decision trees were trained on millions of benign and verified phishing URLs. SHAP analysis quantified feature importance in domain scoring.",
        "results": "The ensemble detector reached an AUC of 0.991 on an active live URL stream, detecting phishing sites within seconds of registration. Lexical features and domain registrar age proved to be the most influential predictive indicators."
    },
    {
        "id": 35,
        "title": "Privacy Attacks and Membership Inference in Deep Neural Networks",
        "abstract": "Machine learning models trained on confidential data can inadvertently leak private training records through model outputs. Membership inference attacks determine whether a specific individual's data was included in a target model's training set. Quantifying privacy leakage informs the deployment of differential privacy safeguards.",
        "methodology": "Shadow models were trained on independent dataset partitions to mimic target model prediction distributions. Attack classifiers were trained to distinguish target predictions on member versus non-member samples. Differential privacy noise addition was evaluated as a mitigation defense.",
        "results": "Unmitigated machine learning classifiers exhibited up to 78 percent attack accuracy, confirming substantial membership leakage. Injecting differential privacy noise with epsilon equal to two reduced attack success to near random guessing."
    },
    {
        "id": 36,
        "title": "Automated Vulnerability Detection in Source Code using Graph Representations",
        "abstract": "Software security bugs and vulnerabilities can lead to catastrophic exploits if not remediated before software deployment. Traditional static analysis tools produce excessive false alarms and fail to capture complex semantic bugs. Code property graphs combined with deep neural networks identify vulnerable source code functions.",
        "methodology": "Source code functions were parsed into combined abstract syntax trees, control flow graphs, and data dependence graphs. Gated graph neural networks propagated vulnerability features across token and node representations. The model was trained on thousands of open-source CVE vulnerability patches.",
        "results": "The graph model achieved an F1-score of 82.6 percent in vulnerability detection, outperforming commercial static analyzers. The tool detected eighteen previously unknown vulnerabilities in popular open-source software libraries."
    },
    {
        "id": 37,
        "title": "Hardware Trojan Detection and Security Verification using Machine Learning",
        "abstract": "Malicious modifications inserted into integrated circuits during semiconductor manufacturing can compromise hardware security. Detecting stealthy hardware Trojans using standard electrical testing is difficult due to their dormant nature. Machine learning analyzes side-channel power dissipation and electromagnetic emissions to uncover Trojans.",
        "methodology": "Transient power profiles and electromagnetic radiation traces were captured across authentic and Trojan-infected chip designs. Principal component analysis reduced side-channel trace dimensionality prior to support vector machine classification. Supervised models separated genuine circuit variations from Trojan activations.",
        "results": "Classification accuracy reached 98.2 percent for detecting dormant hardware Trojans across diverse ASIC benchmarks. Trojans occupying less than 0.1 percent of total circuit gate area were reliably identified."
    },
    {
        "id": 38,
        "title": "Cryptographic Key Recovery Attacks using Deep Learning and Side-Channel Analysis",
        "abstract": "Side-channel attacks exploit physical emissions such as electromagnetic radiation and power consumption to recover secret cryptographic keys. Traditional template attacks require strict statistical assumptions and manual trace alignment. Deep learning networks learn non-linear leakage patterns without requiring prior trace alignment.",
        "methodology": "Convolutional neural networks were trained on electromagnetic radiation traces measured during AES encryption operations. Data augmentation introduced artificial jitter and noise to enhance model generalization against hardware countermeasures. Key hypothesis rankings evaluated attack efficiency.",
        "results": "The deep learning attack recovered the full 128-bit secret key using fewer than fifty power traces, outperforming classical template attacks. Keys were successfully extracted even in the presence of random delay countermeasures."
    },
    {
        "id": 39,
        "title": "Zero-Trust Architecture and Behavioral Biometrics for Continuous Authentication",
        "abstract": "Static passwords and one-time multi-factor authentication tokens are insufficient for continuous zero-trust security postures. Behavioral biometrics continuously verify user identity by analyzing keystroke rhythms, mouse trajectories, and touchscreen dynamics. Anomaly detection models detect account takeover events in real time.",
        "methodology": "Continuous user telemetry was logged during standard enterprise workstation usage sessions. Recurrent neural networks modeled user-specific typing timing and cursor motion distributions. An adaptive risk engine escalated verification challenges when behavioral scores deviated from baselines.",
        "results": "The continuous authentication system achieved an Equal Error Rate (EER) of 2.1 percent across enterprise users. Unauthorized impersonation attempts were detected within forty-five seconds of user seat switching."
    },
    {
        "id": 40,
        "title": "Smart Contract Vulnerability Auditing with Symbolic Execution and Machine Learning",
        "abstract": "Flaws in decentralized blockchain smart contracts have resulted in hundreds of millions of dollars in financial losses. Identifying reentrancy bugs, integer overflows, and unhandled exceptions requires rigorous automated auditing. Integrating symbolic execution with machine learning accelerates smart contract security analysis.",
        "methodology": "EVM bytecode was analyzed to extract control flow graphs and symbolic execution constraints. Machine learning classifiers prioritized vulnerable execution paths to reduce state space explosion. Automated test suites validated candidate exploit vectors on local testnet forks.",
        "results": "The hybrid analyzer uncovered 94.3 percent of known smart contract vulnerabilities on Ethereum mainnet samples. Audit execution times decreased by 60 percent compared to pure symbolic execution engines."
    },

    # Cluster 5: Data Mining & Recommender Systems (41-50)
    {
        "id": 41,
        "title": "Graph Neural Networks for Collaborative Filtering in Recommender Systems",
        "abstract": "Recommender systems utilize collaborative filtering to model user preferences and suggest relevant commercial items. Standard matrix factorization struggles to capture high-order relational connectivity between users and items. Graph neural networks propagate user-item interaction signals across bipartite graphs for superior recommendations.",
        "methodology": "We constructed a user-item bipartite interaction graph and applied neighborhood aggregation with attention weights. Node embeddings were iteratively refined across multiple propagation layers. Bayesian personalized ranking loss was utilized to optimize relative preference ordering.",
        "results": "The graph collaborative filtering model achieved a 12.3 percent improvement in Recall@20 over matrix factorization benchmarks. The model exhibited remarkable resilience to cold-start user and item conditions."
    },
    {
        "id": 42,
        "title": "Session-Based Recommendation Systems with Self-Attention and Recurrent Networks",
        "abstract": "In many e-commerce environments, user profiles and long-term historical records are unavailable due to anonymous guest browsing. Session-based recommenders must infer user purchasing intent solely from clicks within the current browsing session. Self-attention networks capture sequential item transitions and dynamic session context.",
        "methodology": "User click sequences within single browsing sessions were tokenized and embedded into continuous vector representations. Self-attention layers computed relevance weights between historical session clicks and the latest item interaction. A softmax layer generated probability distributions over next-item clicks.",
        "results": "Hit Rate@10 and Mean Reciprocal Rank (MRR@10) improved by 8.7 percent and 6.4 percent respectively on e-commerce benchmarks. The recommender responded to real-time user intent shifts within single browsing sessions."
    },
    {
        "id": 43,
        "title": "Multi-Armed Bandits for Exploration and Exploitation in Online Advertising",
        "abstract": "Displaying online advertisements requires balancing the exploitation of proven high-click banners with the exploration of new creative designs. Contextual multi-armed bandit algorithms optimize click-through rates by adapting to dynamic audience responses. Regret minimization bounds ensure commercial revenue stability.",
        "methodology": "We implemented Thompson Sampling and Upper Confidence Bound (UCB) algorithms conditioned on user demographic and contextual features. Policy updates occurred in real-time streaming pipelines processing millions of impression events per hour. Counterfactual evaluation evaluated offline historical logs.",
        "results": "The contextual bandit system delivered a 19.4 percent increase in click-through rates compared to heuristic A/B testing. Cumulative regret converged logarithmically over a multi-week enterprise deployment."
    },
    {
        "id": 44,
        "title": "Frequent Itemset Mining and Association Rule Discovery in High-Dimensional Transaction Data",
        "abstract": "Discovering hidden associations among purchased products provides foundational intelligence for retail inventory management and cross-selling. Traditional Apriori algorithms face severe computational bottlenecks when analyzing high-dimensional sparse transaction datasets. FP-growth and prefix-tree structures accelerate frequent pattern discovery.",
        "methodology": "We developed a parallelized FP-growth algorithm optimized for distributed memory clusters and disk-spill caching. Transaction databases were compressed into compact frequent pattern trees without candidate generation. Association rules were filtered using confidence, support, and lift thresholds.",
        "results": "Processing speed was 15 times faster than classic Apriori implementations on grocery transaction datasets containing over five million records. The system generated actionable cross-selling rules that improved basket value by eight percent."
    },
    {
        "id": 45,
        "title": "Spatial-Temporal Traffic Flow Forecasting using Graph Convolutional Networks",
        "abstract": "Accurate prediction of urban traffic congestion is vital for intelligent transportation systems and navigation routing engines. Traffic patterns exhibit intricate spatial road-network dependencies and non-stationary temporal dynamics. Graph convolutional networks combined with temporal gating mechanisms model complex traffic flows.",
        "methodology": "Urban road networks were modeled as spatial graph structures with sensor stations as nodes and road connectivity as weighted edges. Spatiotemporal graph convolutional blocks extracted simultaneous spatial topological features and temporal recurrent patterns. Models were evaluated against citywide highway traffic datasets.",
        "results": "Mean Absolute Error (MAE) for 60-minute traffic flow forecasts was reduced by 14.2 percent compared to ARIMA and LSTM baselines. Predictions remained accurate even during severe traffic incidents and adverse weather conditions."
    },
    {
        "id": 46,
        "title": "Scalable Clustering for Massive Datasets using K-Means and Density Peaks",
        "abstract": "Partitioning millions of high-dimensional data points into coherent clusters is a fundamental unsupervised data mining challenge. Traditional clustering algorithms suffer from sensitivity to initialization, outliers, and quadratic computational complexity. Approximate nearest neighbor indexing and density peak finding enable scalable dataset clustering.",
        "methodology": "We integrated mini-batch K-Means with fast density peak estimation over hierarchical navigable small-world graphs. Cluster centers were identified at local density maxima surrounded by lower density regions. Outliers were detected and isolated via kernel density thresholds.",
        "results": "Clustering throughput exceeded one million vectors per minute on standard workstation hardware. Adjusted Rand Index (ARI) surpassed standard K-Means and DBSCAN across non-convex cluster benchmark datasets."
    },
    {
        "id": 47,
        "title": "Outlier Detection in High-Dimensional Time Series using Isolation Forests and Autoencoders",
        "abstract": "Identifying unexpected anomalies in multivariate industrial sensor streams prevents costly equipment breakdowns and operational failures. High data dimensionality and temporal correlations complicate traditional statistical anomaly detection. Ensembles combining Isolation Forests with temporal autoencoders isolate complex outlier events.",
        "methodology": "Sensory telemetry from gas turbines and manufacturing machinery was segmented using sliding temporal windows. An LSTM autoencoder captured normal temporal progression while Isolation Forests flagged anomalies in reconstruction residual vectors. An adaptive threshold accommodated seasonal operational shifts.",
        "results": "The hybrid outlier detector detected ninety-four percent of mechanical failures ahead of catastrophic breakdown. False alarm rates declined by over fifty percent compared to single-model anomaly detection baselines."
    },
    {
        "id": 48,
        "title": "Customer Churn Prediction and Lifetime Value Estimation using Gradient Boosted Decision Trees",
        "abstract": "Retaining existing customers is significantly more cost-effective for subscription businesses than acquiring new subscribers. Machine learning models predict churn propensity and estimate customer lifetime value from transactional engagement histories. Targeted customer retention strategies preserve recurring revenue.",
        "methodology": "We engineered behavioral features including login frequencies, feature usage drops, customer support tickets, and billing histories. Extreme Gradient Boosting (XGBoost) and LightGBM models were trained using cost-sensitive learning to account for class imbalance. Survival analysis modeled retention probabilities over time.",
        "results": "The customer churn prediction pipeline achieved an AUC of 0.892 and a top-decile lift of 4.2x. Proactive retention offers triggered by model alerts reduced quarterly subscription churn by 22 percent."
    },
    {
        "id": 49,
        "title": "Cold-Start Recommendations using Knowledge Graphs and Cross-Domain Transfer",
        "abstract": "Recommender systems suffer severe accuracy degradation when new items or users enter the platform without prior interaction records. Knowledge graphs provide rich auxiliary information encompassing item attributes, creators, and semantic hierarchies. Cross-domain transfer learning propagates preference patterns from mature domains to cold items.",
        "methodology": "An entity knowledge graph was linked to catalog items and embedded using translational graph embedding models. A cross-domain mapping network aligned latent preference spaces between data-rich source domains and target cold domains. Attention mechanisms weighed knowledge graph path relevance.",
        "results": "Recommendation accuracy for brand new catalog items increased by 31 percent compared to standard collaborative filtering. User engagement with cold items matched mature item click-through benchmarks within two days of catalog entry."
    },
    {
        "id": 50,
        "title": "Automated Feature Engineering and Model Selection for Tabular Data Mining",
        "abstract": "Curating high-performing predictive models on tabular business databases is an iterative process requiring manual feature engineering and hyperparameter tuning. Automated machine learning (AutoML) frameworks systematically discover informative feature transformations and optimal model ensembles. This accelerates enterprise data science workflows.",
        "methodology": "We implemented an automated feature engineering pipeline that evaluated mathematical combinations, aggregations, and target encodings. Bayesian optimization searched model architectures including gradient boosting, neural networks, and linear baselines. Multi-model stacking blended predictions for final outputs.",
        "results": "The automated data mining pipeline outperformed human data science baselines on eight out of ten Kaggle tabular competitions. Model development cycles were reduced from weeks to under two hours of automated compute."
    }
]

os.makedirs("c:/Users/hp/Desktop/7th/IR/IR Modules/data", exist_ok=True)
with open("c:/Users/hp/Desktop/7th/IR/IR Modules/data/documents.json", "w", encoding="utf-8") as f:
    json.dump(docs, f, indent=2, ensure_ascii=False)

print(f"Successfully generated {len(docs)} documents in data/documents.json")
