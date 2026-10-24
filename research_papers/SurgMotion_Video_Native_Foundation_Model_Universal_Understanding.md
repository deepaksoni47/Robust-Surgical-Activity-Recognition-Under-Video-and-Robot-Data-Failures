# **SurgMotion: A Video-Native Foundation Model for Universal Understanding of Surgical Videos** 

Jinlin Wu<sup>1,6</sup> , Felix Holm<sup>3</sup> , Chuxi Chen<sup>1</sup> , An Wang<sup>4</sup> , Yaxin Hu<sup>1</sup> , Xiaofan Ye<sup>7</sup> , Zelin Zang 1, Miao Xu1,5,6, Lihua Zhou1, Huarui Wu11, Huai Liao8, Danny T. M. CHAN9, Ming Feng 10, Wai S. Poon7, Hongliang Ren4, Dong Yi1, Nassir Navab3, Gaofeng Meng1,5,6, Jiebo Luo2, Hongbin Liu<sup>1,6</sup> , and Zhen Lei<sup>*1,5,6</sup> 

1Center for Artificial Intelligence and Robotics, Hong Kong Institute of Science and Innovation, Chinese Academy of Sciences, Hong Kong, China 

2Hong Kong Institute of Science and Innovation, Chinese Academy of Sciences, Hong Kong, China 

3Computer Aided Medical Procedures, Technical University of Munich, Munich, Germany 

4Electronic Engineering Department, The Chinese University of Hong Kong, Hong Kong, China 

5University of Chinese Academy of Sciences, Beijing, China 

6State Key Laboratory of Multimodal Artificial Intelligence Systems, Institute of Automation, Chinese Academy of Sciences, Beijing, China 

7Neuromedical Centre, Hong Kong University Shenzhen Hospital, Shenzhen, China 

8Department of Respiratory Medicine, The First Affiliated Hospital of Sun Yat-sen University, Guangzhou, China 

9Department of Surgery, The Chinese University of Hong Kong, Hong Kong, China 

10Department of Neurosurgery, China Pituitary Disease Registry Center, Peking Union Medical College Hospital, Chinese Academy of Medical Sciences and Peking Union Medical College, Beijing, China 

11United Imaging Surgical Co., Ltd. 

September 23, 2026 

##### **Abstract** 

While foundation models have advanced surgical video analysis, current approaches rely predominantly on pixel-level reconstruction objectives that waste model capacity on low-level visual details—such as smoke, specular reflections, and fluid motion—rather than semantic structures essential for surgical understanding. We present SurgMotion, a video-native foundation model that shifts the learning paradigm from pixel-level reconstruction to latent motion prediction. Built on the Video Joint Embedding Predictive Architecture (V-JEPA), SurgMotion introduces three key technical innovations tailored to surgical videos: 1) motion-guided latent masked prediction to prioritize semantically meaningful regions, 2) spatiotemporal affinity self-distillation to enforce relational consistency, and 3) spatiotemporal feature diversity regularization (SFDR) to prevent representation collapse in texture-sparse surgical scenes. To enable large-scale pretraining, we curate SurgMotion-15M, the largest surgical video dataset to date, comprising 3,658 hours of video from 50 sources across 13 organs. Extensive experiments across 17 benchmarks demonstrate that SurgMotion significantly outperforms state-of-the-art methods on surgical workflow recognition (+14.6% F1 on EgoSurgery, +10.3% on PitVis), action triplet recognition (39.54% mAP-IVT on CholecT50), skill assessment, polyp segmentation, and depth estimation. These results establish SurgMotion as a new standard for universal, motion-oriented surgical video understanding. Code and project resources are available at GitHub and the project website. 

## **1 Introduction** 

The development of foundation models has transformed computer vision, with self-supervised pretraining enabling powerful representations that generalize across diverse tasks [1, 2, 3]. In surgical AI, this paradigm shift has led to models such as EndoViT [4], EndoFM [5], and GSViT [6], which leverage large-scale pretraining for 

*Corresponding author. 

1 



<!-- Start of picture text -->
a b<br>Spine 1843 videos, 22288.7 h Brain<br>460 videos, 572.2 h<br>Skill Training 206 videos, 5.3 h ae 1969 videos, 230.76 h Ophthalmic ® Prvateccoooonoseopy f Es m nt<br>swoetoon,  |] 0 So<br>13+ Organs<br>General surgery Breast i —— _<br>10874 videos, 125.2 h 70+ Anatomies 21 videos, 10.6 h<br>100+ Procedures<br>121 videos, 20.5 h Gallbladder eo) 45,757 Videos 63 videos, 50.3 h Bronchi #%| peedseats  {eo — i e  sta<br>3,658 hours<br>+ Liver Stomach<br>50 videos, 6.95 h 180 videos, 252.1 h<br>Uterus<br>8 videos, 20.45 h Prostate ba 30000 videos, 45.4 h J Colon 25 videos, 23.12 h ‘Yyc e | =E=enaoem :sen enaor  pevaingase Oxatin as)<br>c d<br>Node PeformancePe vorrg Oat Se<br>Phase Recognition<br>Microscope<br>Action Recognition<br>Open-Surgery SurgMotion<br>E wo — aeeT . omy1~  Gbyss© £) eu@.<br>Triplet Recognition<br>Laparoscopy Latent Space Prediction<br>Surgical Segmentation<br>Be] Be Osea Pe e etuen a ‘aur eeg= a ® Gastroscopy | §C ——a Surgical Segmentation am hd -|}<br>Pantnr<br>e<br>——— Ooh onc ren ere cerns<br>ns// " aa mae SS ren AN om,J Y Ww =<br>Fer!\ eyi 1/en \ at ws] = /y= ( NIN)  Ay /)<br>ow = ha ee<br>ee) eee) e al =<br><!-- End of picture text -->

Figure 1: **Overview of SurgMotion.** (a) SurgMotion-15M is a large-scale surgical video dataset with 3,658 hours from 50 sources, spanning 12+ organs, 70+ anatomical regions, and 100+ procedures. (b) SurgMotion greatly expands the scale of surgical pre-training data. (c) Larger pre-training data leads to better workflow recognition performance; marker size denotes model size. (d) SurgMotion learns universal representations through motion-guided latent prediction, spatiotemporal self-distillation, and feature diversity regularization. (e) The learned representations transfer effectively across diverse downstream tasks and specialties. 

2 

improved performance on downstream tasks. However, despite these advances, fundamental gaps persist that limit the effectiveness of current surgical foundation models and prevent them from achieving robust, universal surgical understanding. 

First, current self-supervised pretraining approaches face distinct technical challenges. Current approaches rely predominantly on pixel-level reconstruction objectives borrowed from general vision models such as MAE [2] or VideoMAE [7, 8]. In surgical scenes, this creates a fundamental misalignment: models spend capacity learning to reconstruct high-frequency, non-semantic details instead of focusing on the critical yet sparse semantics, such as motion cues of instrument-tissue interactions. Furthermore, surgical videos exhibit strong temporal continuity and structured relationships between anatomical regions and tools. However, many pretraining pipelines lack explicit constraints to enforce relational stability across time [9]. This absence often leads to feature drift and reduced robustness when the model faces significant motion or viewpoint changes, which are common in real surgical procedures. Compounding these challenges are texture-sparse and visually homogeneous regions, such as exposed adipose tissue or smooth organ surfaces. In self-supervised learning, this increases the risk of low-variance features and channel redundancy. Preventing representation collapse while maintaining diverse, informative feature channels in these monotonous scenarios remains a significant challenge. 

Second, beyond architectural limitations, current surgical video models are predominantly confined to small-scale, single-procedure datasets. Common benchmarks focus only on a single type of procedure, such as Cholec80 [10] for laparoscopic cholecystectomy or PitVis [11] for endonasal pituitary surgery. This narrow exposure prevents models from learning universal representations capable of handling the diverse variability of real-world clinical environments. Consequently, existing models struggle to generalize across different anatomical regions, surgical modalities, and institutions [12], failing to leverage the potential benefits of scaling up data diversity. Taken together, these challenges call for a video-native surgical foundation model that (i) emphasizes motion-centric semantics beyond pixel reconstruction, (ii) enforces stable spatiotemporal relations, (iii) maintains feature diversity in texture-sparse scenes, and (iv) scales to substantially broader data. Figure 1 provides an overview of our solution: the scale and diversity of SurgMotion-15M (a–c), the proposed pretraining design (d), and the resulting improvements across tasks and specialties (e). 

To address these fundamental limitations, we present **SurgMotion** , a video-native foundation model designed for universal surgical understanding. Guided by the overview in Figure 1(d), our approach introduces three key technical innovations and one dataset contribution that directly target the identified gaps: 

- **Motion-Guided Latent Masked Prediction.** To address the mismatch between objectives and semantics, we shift masked prediction from pixel space to latent space, building upon the Video Joint Embedding Predictive Architecture (V-JEPA) [13, 14]. We introduce a motion saliency signal based on spatiotemporal gradients to reweight the prediction loss. This strategy effectively reduces sensitivity to low-level visual disturbances and prioritizes motion-informative regions, ensuring that the model focuses on critical tool dynamics and tissue deformations. 

- **Spatiotemporal Consistency via Affinity Self-Distillation.** To enforce relational stability and provide useful coherence constraints, we introduce a Global Token Affinity Matrix that models the pairwise relationships between all latent tokens. We employ a self-distillation mechanism that aligns the affinity structures of the student network with a momentum-updated teacher via exponential moving average (EMA) [15], explicitly imposing spatiotemporal coherence and robustness against augmentations without requiring manual labels. 

- **Spatiotemporal Feature Diversity Regularization (SFDR).** To mitigate representation collapse in homogeneous scenes, we introduce SFDR, which operates on two levels: (i) regularizing the variance of pairwise feature similarities to maintain diversity, and (ii) minimizing inter-channel covariance to reduce redundancy. This encourages the network to learn complementary and informative feature channels even in texture-sparse surgical environments. 

- **The SurgMotion-15M dataset.** To overcome the generalization limitations of narrow datasets, we compile SurgMotion-15M, a massive dataset designed to facilitate large-scale self-supervised learning. Aggregating 3,658 hours of video from 50 distinct sources and 13 anatomical regions, this corpus exposes 

3 

the model to unprecedented variability. This extensive pretraining enables the model to learn universal representations, yielding superior performance across a wide range of downstream validation benchmarks. 

Extensive experiments demonstrate that SurgMotion significantly outperforms existing surgical foundation models across a comprehensive range of tasks. On surgical workflow recognition, SurgMotion improves F1scores by +14.6% on EgoSurgery and +10.3% on PitVis compared to the best prior methods. On action triplet recognition, SurgMotion achieves 39.54% mAP-IVT, setting a new state-of-the-art on CholecT50. On dense prediction tasks, SurgMotion attains superior polyp segmentation under domain shift and best-in-class depth estimation on C3VD. These results establish SurgMotion as a new standard for universal, motion-aware surgical scene understanding. A consolidated comparison across benchmarks and specialties is summarized in Figure 1(e). 

## **2 Related Work** 

**General Vision Foundation Models.** Foundation models have transformed computer vision by shifting the paradigm from supervised learning on curated datasets to large-scale self-supervised pretraining on unlabeled data [1, 2]. The dominant approach has been masked modeling, where portions of the input are masked and reconstructed by the model. In the image domain, MAE [2] demonstrated that Vision Transformers can learn strong representations by reconstructing masked patches. This principle was extended to video through VideoMAE [7] and VideoMAE V2 [8], which reconstruct masked spatiotemporal cubes, achieving impressive results on action recognition benchmarks. More recently, the InternVideo family [16, 17] combines masked video modeling with cross-modal contrastive learning and scales to billions of parameters, achieving state-of-the-art results on general video understanding tasks. 

However, pixel-level reconstruction has inherent limitations: models are forced to predict high-frequency visual details that may be stochastic, irrelevant, or fundamentally unpredictable from the available context. This is particularly problematic in domains with high visual noise, such as surgical video, where smoke, specular reflections, and fluid motion introduce unpredictable pixel variations. To address this, Joint Embedding Predictive Architectures (JEPAs) propose predictions in latent space rather than pixel space [13, 14]. A key challenge with latent prediction is representational collapse, where the model learns trivial solutions. Effective strategies to prevent collapse include teacher-student architectures with exponential moving average (EMA) updates, as employed in DINO [18], DINOv2 [1], DINOv3 [19], BYOL [15], and V-JEPA [13, 14]. V-JEPA is particularly relevant for video understanding, as it learns to predict the latent representations of masked spatiotemporal regions, encouraging the model to capture abstract semantic structures and long-range temporal dynamics rather than low-level pixel statistics. 

**Surgical Foundation Models.** The success of foundation models in general vision has motivated significant efforts to develop domain-specific models for surgery. Early works focused on static frame analysis. EndoViT [4] applies masked image modeling to the Endo700k dataset, demonstrating strong transfer to segmentation tasks. GastroNet [20] scales image-based pretraining to 5 million gastrointestinal endoscopy images, achieving robustness across different imaging devices and institutions. However, these image-based approaches fundamentally ignore the temporal dynamics that are central to understanding surgical procedures. 

To capture procedural dynamics, several video-based surgical foundation models have emerged. EndoFM [5] pretrains a Video Transformer on 33,000 endoscopic clips using masked reconstruction with spatialtemporal attention mechanisms. EndoSSL [21] explores Masked Siamese Networks for self-supervised video analysis in colonoscopy and laparoscopy. More recently, EndoMamba [22] leverages State Space Models for efficient hierarchical pretraining, enabling real-time video understanding. GSViT [6] introduces a generative approach using forward video prediction (next-frame reconstruction) for general surgery. SurgeNetXL [23] scales DINO-based self-supervised learning to 4.7 million frames across multiple procedures, demonstrating the benefits of increased data diversity. Beyond pure vision, SurgVLP [24] integrates language supervision by aligning surgical video clips with transcribed text from lecture videos, enabling zero-shot recognition capabilities. The SelfSupSurg benchmark [9] provides a systematic comparison of self-supervised methods (MoCo 

4 

v2, SimCLR, SwAV, DINO) specifically for surgical computer vision, establishing important baselines for the field. 

Despite this progress, existing surgical foundation models face persistent limitations. Most rely on pixellevel reconstruction objectives, which are inefficient for the high-noise surgical environment where models waste capacity on unpredictable visual artifacts such as smoke, specular reflections, and fluid motion. **SurgMotion** addresses these limitations by building upon the V-JEPA framework, adopting its paradigm of predicting in latent space rather than pixel space, while introducing surgical-specific innovations including motion-guided masking, spatiotemporal consistency constraints, and feature diversity regularization tailored to the unique challenges of surgical video. 

**Surgical Video Datasets.** The development of surgical AI has been fundamentally constrained by data availability. Unlike natural image datasets that can be scraped from the internet, surgical data requires institutional partnerships, ethics approval, and careful de-identification to address privacy regulations. Furthermore, meaningful annotations require domain expertise from surgeons or trained medical personnel, making large-scale labeling prohibitively expensive. 

Existing surgical datasets are typically isolated efforts targeting specific procedures and tasks. For laparoscopic surgery, Cholec80 [10], M2CAI [25], and CholecT50 [26] focus on cholecystectomy, while AutoLaparo [27] and PmLR50 [28] cover hysterectomy and liver resection respectively. Other procedures are represented by PitVis [11] for pituitary surgery and OphNet [29] for ophthalmic procedures. Open surgery datasets include EgoSurgery [12], AVOS [30], and skill assessment benchmarks such as JIGSAWS [31] and AIxSuture [32]. Additional benchmarks include SurgicalActions160 [33] for laparoscopic action recognition and PolypDiag [34] for polyp detection in colonoscopy. Colonoscopy datasets [35, 36, 37] provide annotations for polyp segmentation, depth estimation, and 3D reconstruction. 

While individually valuable, these datasets create fragmented “data silos” that prevent models from learning universal surgical concepts. Each dataset covers a single site, procedure type, and annotation schema, making aggregation non-trivial. Our **SurgMotion-15M** dataset addresses this fragmentation by systematically aggregating 3,658 hours of video from 50 distinct sources spanning 13 anatomical regions, providing the scale and diversity necessary for a truly generalizable surgical foundation model. 

## **3 Methodology** 

### **3.1 The SurgMotion-15M Dataset** 

We pretrain on **SurgMotion-15M** , a large-scale multi-specialty surgical video collection aggregated from both public datasets and private institutional sources (Table 1). SurgMotion-15M covers diverse anatomical regions and procedures, and is used to learn transferable spatiotemporal representations under strong domain redundancy and long-duration recordings. 

### **3.2 Overview** 

Our approach builds on the masked latent prediction paradigm of VJEPA, with an online branch predicting representations of masked regions and a target (EMA) branch providing stable learning targets. We introduce three components tailored to surgical videos: (i) a **motion-guided** latent masked prediction loss that emphasizes dynamically informative regions (e.g., tools and tool–tissue interaction), (ii) a spatiotemporal affinity self-distillation loss that preserves relational structure without being dominated by visible/background tokens, and (iii) a **spatiotemporal feature diversity regularization** (SFDR) loss to mitigate collapse under visually homogeneous scenes. 

Figure 2 illustrates the overall framework. During pre-training, the online encoder processes only visible tubes, while masked tokens are fed to the predictor to infer latent representations for missing regions. In parallel, an EMA target encoder observes the full clip and provides stop-gradient latent targets. The three proposed objectives act on the predicted masked tokens from complementary perspectives: The motion-guided prediction loss prioritizes dynamically salient masked tubes, The affinity self-distillation loss preserves the 

5 

|Specialty|Region / Procedure|Sources|#Videos|Hours|Public/Private|
|---|---|---|---|---|---|
|Neurosurgery|Endonasal pituitary;<br>cranial; spine|PitVis [11]; Web+<br>Institutional|2,303|2,860.85|Mixed|
|General surgery|Abdominal<br>(various)|AVOS [30]|10,874|125.16|Public|
|Ophthalmology|Cataract /<br>ophthalmic|OphNet [29];<br>CATARACTS [38]|1,969|230.76|Public|
|Gastric surgery|Gastric bypass;<br>ESD|MultiBypass140 [39];<br>CoPESD [40]|180|252.10|Public|
|Laparoscopic cholecystectomy|Cholecystectomy|Cholec80 [10];<br>M2CAI-2016 [25]|121|77.51|Public|
|Colonoscopy|Endoscopy|EndoFM [5]|30,000|45.00|Public|
|Gynecology|Hysterectomy|AutoLaparo [27]|25|23.12|Public|
|Urology|Robot-assisted<br>prostatectomy|PSI-AVA [41]|8|20.45|Public|
|Hepatic surgery|Laparoscopic liver<br>resection|PmLR50 [28]|50|6.95|Public|
|Skill training|Bench-top / training|JIGSAWS [31];<br>AIxSuture [32]|206|5.27|Public|
|Bronchoscopy|Endoscopy|Institutional|60|50.00|Private|
|Breast surgery|Egocentric / OR|EgoSurgery [12]|21|10.55|Mixed|



Table 1: Overview of data sources and scale for SurgMotion-15M. The dataset features a comprehensive collection spanning 12 diverse medical specialties, 13 organs, covering multiple surgical procedures and anatomical regions. It features a massive scale in terms of video count and duration (e.g., over 2,800 hours for Neurosurgery), possessing a high degree of significant diversity and temporal extent. 

relational structure among masked tokens, and the diversity regularization prevents feature collapse in visually repetitive surgical scenes. After pre-training, we freeze the encoder and evaluate the learned representation with lightweight probing heads on diverse downstream tasks, demonstrating its transferability across surgical video understanding settings. 

### **3.3 Architecture** 

As shown in Fig. 2, our model follows a teacher–student style masked latent prediction framework with an online branch and an EMA target branch. The online branch receives only visible tubes and predicts latent representations for masked regions, whereas the EMA branch encodes the full clip to produce stable training targets. 

**Tube tokenization.** Given a clip **X** _∈_ R<sup>_T×H×W×C_</sup> , each frame is partitioned into non-overlapping patches of size 16 _×_ 16. A _tube_ corresponds to a fixed spatial patch location across all _T_ frames, yielding 



tube tokens per clip. Let _Im_ and _Iv_ denote masked and visible tube indices. 

6 



<!-- Start of picture text -->
Pre-Training Downstream Probing evaluation<br>Motion-Guided<br>= = = * ae<br>Sam jetor<br>—> Encoder —> Predictor ~~~7 EE === in° WwW aa=<br>a1<br>\‘Affinity Self-Distillation )<br>Mask tokens L-- xu Attention Probing Head<br>EMA © |1 "<br>1 1 ‘a<br>| i: 4 OEPa aEat etas Ecf<br>— fom1 [Diversity Regularization Encoder<br>ipea ema) SP Gradient , aa<br>{ Encoder foo We a[ tos | | |e Bee ——_t _____<br>= pete tt<br>" cicos | | Nia<br><!-- End of picture text -->

Figure 2: Overview of our framework. **Left: Pre-training.** A masked surgical video clip is processed by an online encoder and predictor to infer latent representations for masked tubes, while an EMA target encoder encodes the full clip to provide stable stop-gradient targets. Training is driven by three complementary objectives: motion-guided latent prediction, spatiotemporal affinity self-distillation, and spatiotemporal feature diversity regularization. **Right: Downstream probing evaluation.** After pre-training, the encoder is frozen and evaluated with lightweight probing heads on multiple surgical video understanding tasks. 

**Online branch.** The context encoder _cθ_ encodes only visible tubes to produce visible latents 



For each masked tube _i ∈ Im_ , we initialize a learnable masked token 



where **q** _∈_ R<sup>_D_</sup> is a shared learnable query and **e** _i_ is the positional embedding. A masked token predictor _pφ_ then takes the visible latents and masked tokens as input and outputs predicted latents for masked tubes: 



**Target branch (EMA).** An EMA encoder _c_ ¯ _θ_ (with the same architecture as _cθ_ ) encodes the full clip to produce target latents 



All targets use stop-gradient, denoted sg( _·_ ). This branch provides the stable targets shown in Fig. 2, preventing representation drift during pre-training. 

### **3.4 Training Objectives** 

As depicted in Fig. 2, the three losses supervise the predicted masked tokens from complementary aspects: _L_ motion focuses learning on dynamic surgical content, _L_ st preserves pairwise spatiotemporal relations among masked tokens, and _L_ var encourages a non-collapsed feature space. 

#### **3.4.1 Motion-Guided Latent Masked Prediction** 

Surgical videos are temporally redundant; clinically meaningful changes are often concentrated around tools and interaction regions. We therefore upweight masked tubes that exhibit stronger spatiotemporal motion change in pixel space, corresponding to the _Motion-Guided Prediction_ block in Fig. 2. 

7 

**Motion gradient.** Let **x** _t,i_ be the vectorized pixel patch of tube _i_ at time _t_ . Let _N_ ( _i_ ) be the 4-neighborhood of tube _i_ on the spatial patch grid. We compute a motion score _gi_ as 



**Top-** _K_ **emphasis among masked tubes.** Among masked tubes, we select the top _K_ = 3 by motion gradient: _Ih_ = TopK� _{gi}i∈Im, K_ � _._ 

We assign soft weights with _γ_ = 2: 



**Motion-guided prediction loss.** For masked tube _i_ , let ˆ **z** _i_ be the predicted latent and **z**<sup>_E_</sup> _i_<sup>the EMA target latent.</sup> We use a weighted _ℓ_ 1 regression: 



This design biases learning toward masked regions with stronger motion cues, which in surgical videos often correspond to active tools or tool–tissue interactions. 

#### **3.4.2 Spatiotemporal Affinity Self-Distillation** 

To preserve relational structure while avoiding dominance by visible/background tokens, we distill tube-to-tube affinity distributions _within masked tubes only_ , as illustrated by the _Affinity Self-Distillation_ block in Fig. 2. Let norm( **u** ) = **u** _/_ max( _∥_ **u** _∥_ 2 _, ε_ ). Define normalized masked latents 



For each masked tube _i ∈ Im_ , we compute affinity distributions with temperature _τ_ : 



We distill using KL divergence: 



By matching affinity structure rather than only individual tokens, the model learns more coherent spatiotemporal organization among masked regions. 

#### **3.4.3 Spatiotemporal Feature Diversity Regularization (SFDR)** 

To reduce collapse under visually homogeneous segments, we regularize the variance of predicted masked latents, corresponding to the _Diversity Regularization_ block in Fig. 2. Let **Z**<sup>ˆ</sup> _m_ be predicted masked latents aggregated over the batch, and let _σd_ be the standard deviation of dimension _d_ . We use 



This term encourages sufficient dispersion across feature dimensions and stabilizes representation learning in repetitive surgical scenes. 

8 

#### **3.4.4 Overall objective** 

Combining the three terms, the final training objective is 

_L_ = _L_ motion + _λ_ st _L_ st + _λ_ var _L_ var _,_ 

with _λ_ st = 0 _._ 1 and _λ_ var = 0 _._ 3. As summarized in Fig. 2, these objectives jointly improve masked latent prediction by emphasizing dynamic content, preserving masked-token relations, and maintaining feature diversity. 

## **4 Experiments** 

In this section, we rigorously evaluate SurgMotion against major state-of-the-art (SOTA) foundation models. 

### **4.1 Experimental Setup** 

**Comparison Models.** We benchmark SurgMotion against 13 foundation models spanning general vision and surgical-specific domains. For general vision, we include DINOv3 [19] and VideoMAE [7, 8] at multiple scales (Large and Giant/Huge). For surgical foundation models, we select the most recent and performant methods: image-based approaches EndoViT [4] and GastroNet [20]; video-based methods EndoFM [5], EndoSSL [21], EndoMamba [22], and GSViT [6]; scaled self-supervised models SurgeNetXL [23] and SelfSupSurg [9]; and vision-language model SurgVLP [24]. This selection covers diverse pretraining paradigms, including masked reconstruction, contrastive learning, predictive architectures, and multimodal alignment. 

**Evaluation Tasks.** We evaluate SurgMotion across the full spectrum of surgical video understanding. For **temporal understanding** , we assess surgical workflow recognition on 8 datasets spanning laparoscopic, neurosurgical, ophthalmic, and open surgery domains, reporting Accuracy, F1-score, and Jaccard index. For **fine-grained action understanding** , we evaluate action triplet recognition on CholecT50 [26] (mean Average Precision for instruments, verbs, targets, and their combinations), action recognition on SurgicalActions160 [33], AVOS [30], and PolypDiag [34] (Accuracy, F1), and skill assessment on JIGSAWS [31] (MAE, Spearman correlation). For **dense visual perception** , we benchmark polyp segmentation on five colonoscopy datasets [42, 35, 43, 44, 45] (Dice, MAE) and depth estimation on C3VD [37] (RMSE, AbsRel, SqRel, _δ <_ 1 _._ 1). 

**Pretraining Details.** For pretraining, each video frame is resized to have a shorter side of 256 pixels and randomly cropped to 224 _×_ 224 during training. All videos are processed at 1 fps. The pretraining follows a two-stage procedure: **Stage 1** uses _T_ = 16 frames for 200 epochs with a learning rate of 5 _×_ 10<sup>_−_4</sup> , while **Stage 2** continues from the Stage 1 checkpoint using _T_ = 64 frames for 80 epochs. Balanced sampling across specialties and datasets is ensured by assigning each sample _x_ from dataset _d_ under specialty _s_ a weight _w_ ( _x_ ) ∝ 1 _/_ ( _Nd |Ds|_ ), equalizing sampling mass across specialties and datasets. Tube-level masking is applied with a random mask ratio _ρ ∼ U_ (0 _._ 75 _,_ 0 _._ 95). Optimization is performed using AdamW with a weight decay of 0.04 and a cosine learning-rate schedule. We maintain an EMA momentum of _α_ = 0 _._ 99925 and use a distillation temperature of _τ_ = 0 _._ 1. Training is conducted on 4 NVIDIA H800 GPUs, with per-GPU batch sizes of 64 and 16 for Stage 1 and Stage 2, resulting in global batch sizes of 256 and 64, respectively. 

**Finetuning Details.** To evaluate the performance of the foundation models in our study, we perform attentive probing following the procedure layed out in [14]. This involves adding an attentive probing head to the foundation backbone model, which is finetuned on the respective downstream task. The foundation model weights itself stay frozen. The attention head consists of 4 attention blocks with 16 cross-attention heads. Hyperparameter search is implemented by training 16 attentive heads with different learning parameters simultaneously and reporting the best result. We repeat this procedure for all reported foundation models. 

9 

Table 2: **Surgical workflow recognition on laparoscopic procedures.** Evaluation on cholecystectomy (Cholec80, M2CAI-2016) and hysterectomy (AutoLaparo). We report Accuracy (Acc), F1-score (F1), and Jaccard index. Best results are marked in **bold** . 

|**Model**||**Cholec 8**<br>|**0**<br>||**AutoLap**<br>|**aro**<br>|**M**<br>|**2CAI-2**<br>|**016**<br>|
|---|---|---|---|---|---|---|---|---|---|
||Acc|F1|Jaccard|Acc|F1|Jaccard|Acc|F1|Jaccard|
|Dino v3-L|87.20|81.19|69.60|82.00|74.55|66.10|85.55|74.56|65.39|
|Dino v3-H|86.25|78.64|68.84|76.12|65.64|55.03|80.01|68.92|58.65|
|VideoMAE-L|85.13|77.64|67.00|74.26|64.04|52.97|77.24|65.72|54.55|
|VideoMAE-G|79.48|67.51|54.86|63.40|52.97|41.66|60.73|46.75|35.71|
|EndoViT|56.41|36.78|26.92|51.26|42.95|31.76|47.32|27.68|19.61|
|GastroNet|88.90|81.22|72.02|83.73|74.25|65.07|85.53|73.09|63.90|
|GSViT|39.10|8.06|5.71|25.18|6.15|3.88|36.02|8.52|6.35|
|Endo-Mamba|68.81|54.69|42.58|80.96|71.74|62.11|60.00|42.95|32.52|
|Endo SSL|58.70|38.73|28.07|48.27|39.90|27.85|54.22|33.62|24.75|
|EndoFM|57.25|36.13|26.42|56.67|35.77|47.84|53.47|34.46|25.32|
|Selfsupsurg|48.87|19.98|14.17|43.89|26.76|18.72|45.23|21.03|15.00|
|SurgeNetXL|82.37|69.17|57.95|68.21|53.05|43.12|69.87|55.23|44.04|
|SurgVLP|84.46|74.12|63.09|74.36|65.57|54.15|75.85|61.81|50.49|
|**SurgMotion**|**91.05**|**84.17**|**77.95**|**86.37**|**78.73**|**69.86**|**89.45**|**82.91**|**75.42**|



### **4.2 Surgical Workflow Recognition** 

We evaluate temporal understanding through surgical workflow recognition, the task of identifying the current surgical phase from video. This capability is fundamental for applications such as real-time decision support, operating room scheduling, and automated documentation. We assess SurgMotion on 8 datasets spanning laparoscopic, open, endonasal, neurosurgical, and ophthalmic procedures. 

**Laparoscopic Surgery.** We first evaluate on established laparoscopic benchmarks: Cholec80 [10] and M2CAI2016 [25] for cholecystectomy, and AutoLaparo [27] for hysterectomy. As shown in Table 2, SurgMotion consistently outperforms all baselines across all three datasets. On Cholec80, SurgMotion achieves 91.05% accuracy and 77.95% Jaccard, surpassing DINOv3-L by +8.35% in Jaccard index. This substantial improvement in temporal overlap indicates that SurgMotion captures phase boundaries more precisely than frame-wise discriminative models. On AutoLaparo, where several baselines struggle with complex multi-stage workflows (e.g., GSViT achieves only 25.18% accuracy), SurgMotion maintains robust performance at 86.37% accuracy. On M2CAI-2016, SurgMotion achieves 89.45% accuracy and 75.42% Jaccard, outperforming the next best method by over 3% in F1-score. 

**Open, Endonasal, and Neurosurgery.** We evaluate on three distinct surgical settings beyond laparoscopy: EgoSurgery [12] for egocentric open surgery, PitVis [11] for endonasal pituitary surgery, and Atlas-Neurosurgical for cranial neurosurgery. As shown in Table 3, SurgMotion demonstrates substantial improvements over all baselines. On EgoSurgery, which features egocentric viewpoints and variable lighting conditions, SurgMotion achieves 50.72% F1-score, outperforming DINOv3-L by +18.3% and the best surgical foundation model (EndoFM) by +14.6%. On PitVis, SurgMotion attains 75.53% F1-score, surpassing GastroNet by +10.3%. On Atlas-Neurosurgical, SurgMotion achieves 65.26% F1-score, outperforming all baselines including DINOv3-L (+2.1%) and the surgical-specific SurgVLP (+14.1%). 

**Ophthalmic and Hepatic Surgery.** We further evaluate on OphNet [29] for ophthalmic procedures and PMLR50 [28] for laparoscopic liver resection. As shown in Table 4, SurgMotion achieves strong performance 

10 

Table 3: **Surgical workflow recognition on open, endonasal, and neurosurgical procedures.** Evaluation on egocentric open surgery (EgoSurgery), endonasal pituitary surgery (PitVis), and cranial neurosurgery (AtlasNeurosurgical). Best results are marked in **bold** . 

|**Model**||**EgoSurge**<br>|**ry**<br>||**PitVis**<br>||**Atla**<br>|**s-Neurosu**<br>|**rgical**<br>|
|---|---|---|---|---|---|---|---|---|---|
||Acc|F1-score|Jaccard|Acc|F1-score|Jaccard|Acc|F1-score|Jaccard|
|DinoV3-L|75.37|32.43|26.03|78.24|55.75|46.24|79.31|63.19|56.96|
|DinoV3-H|70.08|36.34|29.03|76.70|50.18|41.09|76.54|59.76|53.70|
|VideoMAE-L|64.52|26.84|19.82|80.62|56.89|47.75|76.55|59.13|52.13|
|VideoMAE-G|62.25|20.40|16.13|72.52|44.34|35.59|77.06|53.89|47.08|
|EndoViT|32.32|9.29|6.31|53.09|28.32|20.77|73.71|48.51|42.36|
|GastroNet|48.12|16.08|12.87|82.95|65.22|55.85|77.79|58.71|51.94|
|GSViT|35.74|7.33|5.17|26.68|3.42|2.29|61.36|28.01|24.15|
|Endo Mamba|53.30|13.90|10.32|61.54|35.27|27.84|70.37|44.57|37.87|
|EndoSSL|49.78|12.04|8.86|56.72|27.00|19.60|72.14|48.20|41.86|
|EndoFM|54.85|36.13|26.42|63.31|31.86|23.46|77.87|46.59|41.09|
|Selfsupsurg|36.03|7.38|5.22|47.87|18.63|13.43|65.19|37.26|32.18|
|SurgeNetXL|35.82|7.34|5.18|50.99|18.60|13.83|69.36|44.67|38.35|
|SurgVLP|57.44|23.26|17.46|70.99|29.89|32.06|72.56|51.16|44.52|
|**SurgMotion**|**75.57**|**50.72**|**42.95**|**86.52**|**75.53**|**65.51**|**83.48**|**65.26**|**59.35**|



Table 4: **Surgical workflow recognition on ophthalmic and hepatic procedures.** Evaluation on ophthalmic surgery (OphNet) and laparoscopic liver resection (PMLR50). Best results are marked in **bold** . 

|**Model**|Acc|**OphNet**<br>F1-score|Jaccard|Acc|**PMLR 50**<br>F1-score|Jaccard|
|---|---|---|---|---|---|---|
|DinoV3-L|62.80|39.99|31.99|82.98|63.85|53.76|
|DinoV3-H|63.31|38.93|31.68|73.18|37.59|29.74|
|VideoMAE-L|58.97|34.35|27.00|84.50|68.68|58.25|
|VideoMAE-G|51.53|27.98|21.38|84.07|65.61|54.94|
|EndoViT|28.14|9.57|6.91|75.41|41.92|34.51|
|GastroNet|61.41|37.95|30.56|78.28|56.94|47.23|
|GSViT|17.81|2.52|1.69|65.37|16.13|14.40|
|Endo Mamba|28.94|10.37|7.53|75.50|46.34|35.91|
|EndoSSL|28.33|9.73|6.86|78.16|46.43|39.88|
|EndoFM|27.88|8.06|5.66|65.44|35.35|25.03|
|Selfsupsurg|23.15|6.36|4.48|70.55|29.81|24.73|
|SurgeNetXL|24.97|7.69|5.79|65.26|16.13|13.39|
|SurgVLP|48.13|24.61|18.76|76.75|51.23|42.38|
|**SurgMotion**|**73.04**|**51.22**|**43.83**|**91.91**|**85.65**|**78.56**|



in both domains. On OphNet, which features a very diverse set of opthalmic procedures with over 33 distinc phases, SurgMotion attains 73.04% accuracy and 51.22% F1-score, exceeding DINOv3-L by over 10% in both metrics. On PMLR50, SurgMotion achieves 91.91% accuracy and 85.65% F1-score, outperforming VideoMAE-L by +7.4% in accuracy and +17.0% in F1-score. 

11 

Table 5: **Action recognition.** Evaluation on laparoscopic actions (SurgicalActions160), open surgery (AVOS), and colonoscopic diagnosis (PolypDiag). We report Accuracy (Acc) and F1-score (F1). Best results are marked in **bold** . 

|**Model**|**Surgical**<br>|**Actions160**<br>|**A**<br>|**VOS**<br>|**Pol**<br>|**ypDiag**<br>|
|---|---|---|---|---|---|---|
||Acc|F1-score|Acc|F1-score|Acc|F1-score|
|Dino V3-L|64.58|63.78|64.27|26.02|95.00|93.54|
|Dino V3-H|75.00|72.35|77.37|45.12|97.50|96.67|
|VideoMAE-L|70.83|70.55|59.60|23.01|96.25|94.91|
|VideoMAE-G|41.67|40.67|48.34|8.15|97.50|96.55|
|EndoViT|16.67|10.25|48.34|8.15|96.25|94.91|
|GastroNet|41.67|41.61|54.54|17.84|98.75|98.30|
|GSViT|18.75|11.06|48.34|8.15|75.00|42.86|
|Endo Mamba|58.45|36.23|31.55|22.86|23.54|20.05|
|EndoSSL|12.50|6.77|48.34|8.15|92.50|90.31|
|EndoFM|8.33|3.80|48.34|8.15|97.50|96.44|
|Selfsupsurg|10.42|7.93|48.34|8.15|90.00|85.06|
|SurgNetXL|50.00|49.08|50.44|13.50|97.50|96.67|
|SurgVLP|39.58|36.64|49.66|11.63|93.75|91.80|
|**SurgMotion**|**75.63**|**72.62**|**70.52**|**66.32**|**98.81**|**98.34**|



### **4.3 Fine-Grained Action Understanding** 

Beyond phase-level workflow recognition, surgical video understanding requires fine-grained reasoning about actions, instruments, and their interactions. We evaluate SurgMotion on three complementary tasks: action recognition, action triplet recognition, and skill assessment. 

**Action Recognition.** We evaluate short-term action recognition on SurgicalActions160 [33] for laparoscopic actions, AVOS [30] for open surgery, and PolypDiag [34] for colonoscopic polyp diagnosis. As shown in Table 5, SurgMotion achieves competitive or superior performance across all three benchmarks. On SurgAction160, SurgMotion achieves 75.63% accuracy, matching DINOv3-H while substantially outperforming surgicalspecific models such as SurgeNetXL (50.00%) and SurgVLP (39.58%). On AVOS, SurgMotion attains 66.32% F1-score, dramatically improving over all baselines including DINOv3-H (45.12% F1) by +21.2%. On PolypDiag, SurgMotion achieves 98.81% accuracy, slightly exceeding the strong GastroNet baseline (98.75%). 

**Action Triplet Recognition.** We evaluate the model’s ability to disentangle complex surgical interactions on CholecT50 [26], which requires simultaneous recognition of Instruments (I), Verbs (V), and Targets (T). As shown in Table 6, SurgMotion achieves state-of-the-art performance across all metrics. For individual components, SurgMotion attains 91.55% AP-I, 57.72% AP-V, and 48.18% AP-T, outperforming DINOv3-L by +4.4% on instruments, +3.8% on verbs, and +4.7% on targets. For the challenging triplet association task (AP-IVT), SurgMotion achieves 39.54%, surpassing DINOv3-L by +2.7%. The consistent improvement in verb recognition confirms that SurgMotion’s video-native pretraining captures motion dynamics critical for distinguishing fine-grained surgical actions. 

**Skill Assessment.** We evaluate surgical skill quantification on JIGSAWS [31], a benchmark of robotic surgery tasks with expert skill annotations. The annotations show a skill rating from 1 to 5 in 5 categories, we predict the global skill score. As shown in Table 7, SurgMotion achieves the MAE of 2.649 and the Spearman correlation of 0.770, substantially outperforming all baselines. Compared to GastroNet (MAE 3.596, Spearman 0.500), SurgMotion reduces prediction error by 26% while improving rank correlation by 27 percentage points. 

12 

Table 6: **Action triplet recognition on CholecT50.** We report Average Precision (AP) for individual components—Instruments (I), Verbs (V), Targets (T)—and their associations (IV, IT, IVT). Best results are in **bold** . 

|**Model**|**Com**|**ponent dete**|**ction**|**Tr**|**iplet associa**|**tion**|
|---|---|---|---|---|---|---|
||AP-I|AP-V|AP-T|AP-IV|AP-IT|AP-IVT|
|Dino v3-L|87.16|53.90|43.44|37.31|40.81|36.85|
|Dino v3-H|90.18|55.83|42.70|36.96|34.09|30.06|
|VideoMAE-G|77.06|46.69|40.42|30.60|29.08|24.41|
|VideoMAE|81.58|47.93|36.78|31.11|30.40|25.72|
|EndoViT|52.41|32.01|27.45|20.65|19.42|18.05|
|GastroNet|73.69|48.14|39.44|31.22|30.05|27.37|
|GSViT|18.65|12.17|8.40|6.36|3.56|3.20|
|Endo Mamba|58.45|36.23|31.55|22.86|23.54|20.05|
|EndoSSL|50.47|34.79|27.93|22.67|20.20|20.17|
|EndoFM|48.57|28.52|27.51|18.44|19.71|18.98|
|Selfsupsurg|45.50|29.16|22.65|18.25|14.41|13.50|
|SurgNetXL|74.64|45.58|40.08|29.92|32.51|27.36|
|SurgVLP|75.56|44.75|33.29|29.30|28.26|24.11|
|**SurgMotion**|**91.55**|**57.72**|**48.18**|**40.39**|**43.47**|**39.54**|



Table 7: **Skill assessment on JIGSAWS.** We report Mean Absolute Error (MAE, lower is better) and Spearman correlation (higher is better). Best results are marked in **bold** . 

|**Model**|**MAE**|**Spearman**|
|---|---|---|
|DinoV3-L|3.9895|0.4159|
|DinoV3-H|3.7070|0.4167|
|VideoMAE-L|3.9189|0.4174|
|VideoMAE-G|3.9720|0.3362|
|EndoViT|4.1000|0.4431|
|GastroNet|3.5962|0.5002|
|GSViT|4.7528|NaN|
|EndoSSL|4.2182|0.3986|
|EndoFM|4.0895|0.4465|
|Selfsupsurg|4.0229|NaN|
|SurgNetXL|4.7663|0.0509|
|SurgVLP|3.9817|0.4071|
|**SurgMotion**|**2.649**|**0.770**|



This strong performance suggests that SurgMotion’s learned representations capture not only what actions are performed but also the quality and fluency of their execution. 

### **4.4 Dense Visual Perception** 

Beyond high-level temporal and action understanding, surgical AI systems require dense pixel-level predictions for tasks such as tissue segmentation and 3D reconstruction. We evaluate SurgMotion on polyp segmentation and depth estimation to assess whether our video-level pretraining transfers to dense prediction tasks. 

13 

Table 8: **Polyp segmentation.** Evaluation on in-distribution (Kvasir, CVC-ClinicDB) and out-of-distribution (CVC-300, CVC-ColonDB, ETIS-LaribPolypDB) benchmarks. We report Dice coefficient (higher is better) and MAE (lower is better). Best results are marked in **bold** , second-best are marked by underlined. 

|||**In-Do**|**main**||||**Out-of-Di**|**stributio**|**n**||
|---|---|---|---|---|---|---|---|---|---|---|
||**Clin**|**icDB**|**Kv**|**asir**|**CVC**|**-300**|**Colo**|**nDB**|**E**|**TIS**|
|**Model**|**Dice**_↑_|**MAE**_↓_|**Dice**_↑_|**MAE**_↓_|**Dice**_↑_|**MAE**_↓_|**Dice**_↑_|**MAE**_↓_|**Dice**_↑_|**MAE**_↓_|
|EndoFM|0.8353|0.0222|0.8421|0.0438|0.8352|0.0126|0.6338|0.0485|0.4952|0.0527|
|EndoMamba|0.7182|0.0398|0.6831|0.1017|0.6075|0.0451|0.4684|0.0809|0.2661|0.1175|
|EndoViT|0.8271|0.0236|0.8533|0.0442|0.8303|0.0137|0.6383|0.0515|0.5345|0.0390|
|EndoSSL|0.8349|0.0244|0.8823|0.0379|0.8622|0.0088|0.7270|0.0388|0.6659|0.0240|
|GSViT|0.6218|0.0610|0.6238|0.1331|0.3633|0.1066|0.4027|0.1071|0.2692|0.1504|
|VideoMAE-G|0.8323|0.0249|0.8218|0.0579|0.7883|0.0226|0.6391|0.0539|0.5242|0.0489|
|SelfSupSurg|0.8623|0.0194|0.8452|0.0442|0.8355|0.0116|0.6619|0.0472|0.6057|0.0309|
|SurgNetXL|0.8499|0.0196|0.8317|0.0569|0.7730|0.0238|0.6567|0.0550|0.5534|0.0652|
|SurgVLP|0.8253|0.0260|0.8433|0.0530|0.8508|0.0131|0.6699|0.0493|0.6265|0.0384|
|GastroNet|0.8611|0.0171|0.8897|0.0359|0.8250|0.0134|0.7182|0.0369|0.7108|**0.0160**|
|DINOv3-L|0.8692|0.0164|0.8872|**0.0318**|0.8811|0.0071|0.7646|0.0342|0.7293|0.0197|
|DINOv3-H|0.8676|0.0204|**0.9019**|0.0336|0.8912|**0.0066**|0.7795|0.0345|0.7692|0.0175|
|SurgMotion|**0.8694**|**0.0145**|0.8919|0.0352|**0.9112**|0.0071|**0.7985**|**0.0237**|**0.7798**|0.0163|



**Polyp Segmentation.** We evaluate polyp segmentation on five colonoscopy benchmarks following established protocols [46, 47, 48]: Kvasir [42] and CVC-ClinicDB [35] as in-distribution (ID) datasets, and CVC300 [43], CVC-ColonDB [44], and ETIS-LaribPolypDB [45] as out-of-distribution (OOD) datasets. As shown in Table 8, SurgMotion achieves state-of-the-art performance, particularly on OOD datasets. On the ID benchmarks, SurgMotion attains Dice coefficients of 0.8694 (CVC-ClinicDB) and 0.8919 (Kvasir), matching the strongest baselines. On the more challenging OOD benchmarks, SurgMotion demonstrates superior generalization: 0.9112 Dice on CVC-300, 0.7985 on CVC-ColonDB, and 0.7798 on ETIS-LaribPolypDB, with the lowest MAE across all OOD datasets (0.0071, 0.0237, and 0.0163, respectively). These results indicate that SurgMotion’s pretraining captures generalizable visual features for polyp morphology rather than datasetspecific artifacts. 

**Depth Estimation.** We evaluate monocular depth estimation on the C3VD colonoscopy benchmark [37], which provides ground-truth depth maps from 3D reconstructions. As shown in Table 9, SurgMotion achieves an RMSE of 1.88, outperforming all baselines, including DINOv3-L (1.92), GastroNet (1.91), and the videobased InternVideo2-1b (2.00). SurgMotion also achieves the best AbsRel (0.051) and _δ <_ 1 _._ 1 accuracy (0.86). This strong performance is notable given that SurgMotion was not explicitly trained for geometric prediction, suggesting that our latent prediction objective implicitly captures 3D spatial structure through motion cues. 

### **4.5 Qualitative Analysis** 

We provide qualitative visualizations to complement our quantitative evaluation and offer insight into model behavior. 

**Workflow Recognition.** Figure 3 visualizes temporal phase predictions across 12 representative cases from 8 datasets. Each row shows the predicted phase sequence as a colored bar, with the ground truth at the top. SurgMotion produces smoother predictions that closely track the ground truth phase boundaries, whereas baselines exhibit frequent fragmentation and phase confusion. For instance, on Atlas Case 113, SurgMotion achieves 87.5% accuracy while EndoMamba drops to 42.7%, suffering from severe over-segmentation. On 

14 

Table 9: **Depth estimation on C3VD.** We report RMSE, AbsRel, SqRel (lower is better), and _δ <_ 1 _._ 1 accuracy (higher is better). Best results are marked in **bold** . 

|**Model**|**RMSE**_↓_|**AbsRel**_↓_|**SqRel**_↓_|_δ <_1_._1_↑_|
|---|---|---|---|---|
|VideoMAE-G|2.25|0.062|0.17|0.81|
|EndoViT|2.05|0.057|0.15|0.84|
|GSViT|3.65|0.094|0.45|0.63|
|EndoFM|2.16|0.059|0.16|0.82|
|DINOV3-L|1.92|0.055|0.13|0.84|
|DINOv3-H|1.94|0.057|0.13|0.83|
|SelfSupSurg|3.12|0.094|0.36|0.64|
|SurgVLP|2.94|0.085|0.32|0.68|
|SurgeNetXL|2.88|0.078|0.29|0.72|
|EndoSSL|2.00|0.058|0.15|0.82|
|GastroNet|1.91|0.055|0.13|0.84|
|InternVideo-Next-L|2.22|0.064|0.17|0.79|
|InternVideo2-1b|2.00|0.059|0.14|0.82|
|**SurgMotion**|**1.88**|**0.051**|**0.13**|**0.86**|



challenging cases such as EgoSurgery Case 7, which features dramatic viewpoint changes, SurgMotion (80.9%) substantially outperforms EndoFM (46.7%) and SurgVLP (49.6%). These visualizations confirm that SurgMotion’s temporal representations capture coherent phase structure rather than noisy frame-level predictions. 

**Polyp Segmentation.** Figure 4(a) shows qualitative segmentation results on both in-distribution and out-ofdistribution colonoscopy datasets. SurgMotion consistently produces tighter boundaries that align closely with ground truth polyp contours. On OOD datasets (CVC-300, CVC-ColonDB, ETIS), where domain shift poses significant challenges, baselines such as VideoMAE-G and EndoFM produce incomplete or fragmented masks, while SurgMotion maintains robust segmentation. This visual evidence corroborates the quantitative OOD improvements reported in Table 8. 

**Depth Estimation.** Figure 4(b) visualizes depth predictions across different colon segments (descending, sigmoid, cecum, transverse). SurgMotion produces spatially coherent depth maps that accurately capture the tubular geometry of the colon lumen, with smooth gradients from near (warm colors) to far (cool colors). In contrast, baselines such as SurgVLP and EndoFM exhibit noisy predictions with artifacts near tissue boundaries. The depth maps from SurgMotion most closely match the ground truth across all anatomical segments, demonstrating that our video-native pretraining captures meaningful 3D geometric structure. 

**Triplet Recognition.** Figure 5 presents qualitative results of surgical action triplet recognition on the CholecT50 dataset. SurgMotion accurately identifies the instrument–verb–target triplets across diverse surgical contexts, such as grasping, retracting, and dissecting actions performed on structures like the cystic artery, cystic duct, and gallbladder. The model demonstrates strong capability in disentangling overlapping actions, effectively distinguishing interactions involving multiple instruments and anatomical targets within a single frame. In contrast to prior models prone to confusion under such complex scenes, SurgMotion maintains consistent triplet predictions that align with the ground truth annotations. These results indicate that our video-native pretraining enables robust spatiotemporal understanding of fine-grained surgical dynamics and inter-object relationships. 

15 

## **5 Conclusions** 

In this study, we present SurgMotion, a video-native foundation model for universal surgical video understanding. By shifting from pixel-level reconstruction to latent motion prediction through the V-JEPA framework, SurgMotion learns robust spatiotemporal representations that capture the semantic structure of surgical procedures while disregarding low-level visual noise. Our key technical contributions, motion-guided latent masked prediction, spatiotemporal affinity self-distillation, and spatiotemporal feature diversity regularization, specifically address the challenges of surgical videos, including texture-sparse regions, visual artifacts, and the struggle to maintain temporal coherence. To enable large-scale pretraining, we have curated SurgMotion-15M, the largest multi-modal surgical video dataset to date, spanning 3,658 hours across 13 anatomical regions. 

Extensive experiments have demonstrated that SurgMotion achieves state-of-the-art performance across a comprehensive variety of surgical video understanding tasks. On workflow recognition, SurgMotion outperforms prior methods on all 8 benchmarks, with particularly strong gains on challenging domains such as EgoSurgery (+14.6% F1 over the best surgical foundation model) and PitVis (+10.3% F1). On fine-grained action understanding, SurgMotion sets new state-of-the-art results on triplet recognition, action recognition, and skill assessment. On dense prediction tasks, SurgMotion achieves superior polyp segmentation, especially under domain shift, and best-in-class depth estimation, despite not being explicitly trained for geometric reasoning. 

While SurgMotion-15M is the largest surgical video dataset to date, certain procedures and institutions remain underrepresented. Our results demonstrate that strong cross-procedural generalization is achievable when diverse, large-scale data is available, underscoring the importance of continuing multi-institutional data curation efforts. 

SurgMotion represents a step toward surgical AI that generalizes across procedures, institutions, and imaging modalities. We envision applications in intraoperative decision support, automated documentation, and objective skill assessment, and hope that SurgMotion will serve as a foundation for further research in surgical video understanding. 

## **Acknowledgements** 

This work was supported in part by the InnoHK Program of the Hong Kong SAR Government and the National Natural Science Foundation of China (Grant No. #62306313). We also acknowledge the Guangdong Medical Doctor Association Postgraduate Medical Education Innovative Research Project (Project No. #BJ202602). Felix Holm was supported by a fellowship within the IFI programme of the German Academic Exchange Service (DAAD), funded by the Federal Ministry of Education and Research (BMBF). 

16 



<!-- Start of picture text -->
Atlas: Case 113 Atlas: Case 233<br>Ground truth Ground truth<br>Ours 87.5% 92.3% Ours  87.5% 92.3%<br>DINOv3-H 75.9% 74.8% DINOv3-H 75.9% 74.8%<br>EndoMamba 42.7% 74.3% EndoMamba 42.7% 74.3%<br>EndoFM 59.3% 59.5% EndoFM 59.3% 59.5%<br>GastroNet 78.2% 76.1% GastroNet 78.2% 76.1%<br>VideoMAE-G 80.1% 81.1% VideoMAE-G 80.1% 81.1%<br>AutoLaparo: Case 15 AutoLaparo: Case 16<br>Ground truth Ground truth<br>Ours 91.4% Ours 91.2%<br>DINOv3-H 75.1% DINOv3-H 83.2%<br>EndoFM 41.3% EndoFM 57.6%<br>EndoMamba 86.1% EndoMamba 87.0%<br>GastroNet 88.6% GastroNet 87.8%<br>VideoMAE-G 48.7% VideoMAE-G 73.1%<br>Cholec80: Case 49 Cholec80: Case 78<br>Ground truth Ground truth<br>Ours 96.6% Ours 88.8%<br>DINOv3-H 89.9% DINOv3-H 84.6%<br>EndoFM 39.0% EndoFM 51.0%<br>EndoMamba 62.1% EndoMamba 64.4%<br>GastroNet 91.7% GastroNet 72.4%<br>VideoMAE-G 75.5% VideoMAE-G 82.1%<br>PitVis: Case 6 PitVis: Case 12<br>Ground truth Ground truth<br>Ours 84.5% Ours 83.4%<br>DINOv3-H 77.4% DINOv3-H 70.9%<br>EndoFM 64.5% EndoFM 50.8%<br>EndoMamba 65.0% EndoMamba 45.7%<br>GastroNet 81.9% GastroNet 78.5%<br>VideoMAE-G 67.6% VideoMAE-G 69.1%<br>OphNet: Case 747 EgoSurgery: Case 7<br>Ground truth Ground truth<br>Ours 86.1% Ours  80.9%<br>DINOv3-H 81.4% DINOv3-H 66.9%<br>EndoMamba 61.8% EndoMamba 66.4%<br>EndoSSL 65.5% EndoFM 46.7%<br>SurgVLP 77.9% SurgVLP 49.6%<br>VideoMAE-G 76.7% VideoMAE-G 70.0%<br>PmLR50: Case 45 M2CAI: Case 13<br>Ground truth Ground truth<br>Ours 97.2% Ours 91.0%<br>DINOv3-H 74.2% DINOv3-H 83.5%<br>EndoMamba 78.5% EndoFM 62.8%<br>GastroNet 61.2% EndoMamba 67.5%<br>SurgVLP 73.8% GastroNet 81.7%<br>VideoMAE-G 80.3% VideoMAE-G 60.3%<br><!-- End of picture text -->

Figure 3: **Multi-dataset qualitative showcase of surgical phase recognition** . Frame-wise phase predictions on representative cases from different surgical datasets are compared with the ground truth and competing methods. Our method yields predictions that better match the ground truth, with clearer phase boundaries and stronger temporal consistency across diverse scenarios. 

17 



<!-- Start of picture text -->
Image SurgVLP EndoFM VideoMAE-G GastroNet DINOv3-H SurgMotion GT<br>(a) Polyp Segmentation. ID: Kvasir, CVC-ClinicDB. OOD: CVC-300, CVC-ColonDB, ETIS-LaribPolypDB.<br>Predictions are overlaid in magenta; GT is overlaid in blue.<br>Image SurgVLP EndoFM VideoMAE-G GastroNet DINOv3-H SurgMotion GT<br>(b) Colonoscopic Depth Estimation (C3VD).  Depth heatmaps across colon segments (descending, sigmoid, cecum,<br>transverse).<br>(ID)<br>Kvasir<br>(ID)<br>CVC-ClinicDB<br>(OOD)<br>CVC-300<br>(OOD)<br>CVC-ColonDB<br>ETIS (OOD)<br>Colon<br>Descending<br>Colon<br>Sigmoid<br>Cecum<br>Colon<br>Transcending<br><!-- End of picture text -->

Figure 4: **Qualitative Results on Polyp Segmentation and Colonoscopic Depth Estimation.** Columns show the input frame, predictions from SurgVLP, EndoFM, VideoMAE-G, GastroNet, DINOv3-H, and our SurgMotion, as well as the ground truth (GT). SurgMotion yields tighter polyp boundaries under domain shift and more spatially coherent depth maps across anatomical segments. 

18 



<!-- Start of picture text -->
Ours Ours<br>Ours Ours<br>Ours Ours<br>Ours Ours<br>Ours Ours<br><!-- End of picture text -->

Figure 5: **Illustration of surgical action triplet recognition on CholecT50 dataset.** These frames from a CholecT50 laparoscopic cholecystectomy video exemplify the model’s capability to disentangle overlapping surgical actions. 

19 

## **References** 

- [1] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby _et al._ , “Dinov2: Learning robust visual features without supervision,” _arXiv preprint arXiv:2304.07193_ , 2023. 

- [2] K. He, X. Chen, S. Xie, Y. Li, P. Doll´ar, and R. Girshick, “Masked autoencoders are scalable vision learners,” in _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , 2022, pp. 16 000–16 009. 

- [3] C. Feichtenhofer, Y. Li, K. He _et al._ , “Masked autoencoders as spatiotemporal learners,” _Advances in neural information processing systems_ , vol. 35, pp. 35 946–35 958, 2022. 

- [4] D. Bati´c, F. Holm, E. Ozsoy,<sup>¨</sup> T. Czempiel, and N. Navab, “Endovit: pretraining vision transformers on a large collection of endoscopic images,” _International Journal of Computer Assisted Radiology and Surgery_ , vol. 19, no. 6, pp. 1085–1091, 2024. 

- [5] Z. Wang, C. Liu, S. Zhang, and Q. Dou, “Foundation model for endoscopy video analysis via largescale self-supervised pre-train,” in _International Conference on Medical Image Computing and ComputerAssisted Intervention_ . Springer, 2023, pp. 101–111. 

- [6] S. Schmidgall, J. W. Kim, J. Jopling, and A. Krieger, “General surgery vision transformer: A video pretrained foundation model for general surgery,” _arXiv preprint arXiv:2403.05949_ , 2024. 

- [7] Z. Tong, Y. Song, J. Wang, and L. Wang, “Videomae: Masked autoencoders are data-efficient learners for self-supervised video pre-training,” in _Advances in Neural Information Processing Systems (NeurIPS)_ , vol. 35, 2022, pp. 10 078–10 093. 

- [8] L. Wang, B. Huang, Z. Zhao, Z. Tong, Y. He, Y. Wang, Y. Wang, and Y. Qiao, “Videomae v2: Scaling video masked autoencoders with dual masking,” in _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ . IEEE, 2023, pp. 14 549–14 560. 

- [9] S. Ramesh, V. Srivastav, D. Alapatt, T. Yu, A. Murali, L. Sestini, C. I. Nwoye, I. Hamoud, S. Sharma, A. Fleurentin _et al._ , “Dissecting self-supervised learning methods for surgical computer vision,” _Medical Image Analysis_ , vol. 88, p. 102844, 2023. 

- [10] A. P. Twinanda, S. Shehata, D. Mutter, J. Marescaux, M. De Mathelin, and N. Padoy, “Endonet: a deep architecture for recognition tasks on laparoscopic videos,” _IEEE transactions on medical imaging_ , vol. 36, no. 1, pp. 86–97, 2016. 

- [11] A. Das, D. Z. Khan, D. Psychogyios, Y. Zhang, J. G. Hanrahan, F. Vasconcelos, Y. Pang, Z. Chen, J. Wu, X. Zou _et al._ , “Pitvis-2023 challenge: Workflow recognition in videos of endoscopic pituitary surgery,” _Medical Image Analysis_ , p. 103716, 2025. 

- [12] R. Fujii, M. Hatano, H. Saito, and H. Kajita, “Egosurgery-phase: a dataset of surgical phase recognition from egocentric open surgery videos,” in _International Conference on Medical Image Computing and Computer-Assisted Intervention_ . Springer, 2024, pp. 187–196. 

- [13] A. Bardes, Q. Garrido, J. Ponce, X. Chen, M. Rabbat, Y. LeCun, M. Assran, and N. Ballas, “Revisiting feature prediction for learning visual representations from video,” _arXiv preprint arXiv:2404.08471_ , 2024. 

- [14] M. Assran, A. Bardes, D. Fan, Q. Garrido, R. Howes, M. Muckley, A. Rizvi, C. Roberts, K. Sinha, A. Zholus _et al._ , “V-jepa 2: Self-supervised video models enable understanding, prediction and planning,” _arXiv preprint arXiv:2506.09985_ , 2025. 

- [15] J.-B. Grill, F. Strub, F. Altch´e, C. Tallec, P. Richemond, E. Buchatskaya, C. Doersch, B. Avila Pires, Z. Guo, M. Gheshlaghi Azar _et al._ , “Bootstrap your own latent: A new approach to self-supervised learning,” in _Advances in Neural Information Processing Systems (NeurIPS)_ , vol. 33, 2020, pp. 21 271–21 284. 

20 

- [16] Y. Wang, K. Li, X. Li, J. Yu, Y. He, G. Chen, B. Pei, R. Zheng, J. Xu, Z. Wang, Y. Shi, T. Jiang, S. Li, H. Zhang, Y. Huang, Y. Qiao, Y. Wang, and L. Wang, “Internvideo2: Scaling video foundation models for multimodal video understanding,” in _European Conference on Computer Vision (ECCV)_ , 2024. 

- [17] C. Wang, K. Li, Y. He, Y. Wang, Z. Yan, J. Yu, Y. Wang, and L. Wang, “Internvideo-next: Towards general video foundation models without video-text supervision,” _arXiv preprint arXiv:2512.01342_ , 2025. 

- [18] M. Caron, H. Touvron, I. Misra, H. J´egou, J. Mairal, P. Bojanowski, and A. Joulin, “Emerging properties in self-supervised vision transformers,” in _Proceedings of the IEEE/CVF international conference on computer vision_ , 2021, pp. 9650–9660. 

- [19] M. Seitzer _et al._ , “Dinov3,” _arXiv preprint arXiv:2508.10104_ , 2025. 

- [20] M. R. Jong, T. G. Boers, K. N. Fockens, J. B. Jukema, C. H. Kusters, T. J. Jaspers, R. v. E. van Heslinga, F. C. Slooter, M. R. Struyvenberg, R. Bisschops _et al._ , “Gastronet-5m: A multicenter dataset for developing foundation models in gastrointestinal endoscopy,” _Gastroenterology_ , 2025. 

- [21] R. Hirsch, M. Caron, R. Cohen, A. Livne, R. Shapiro, T. Golany, R. Goldenberg, D. Freedman, and E. Rivlin, “Self-supervised learning for endoscopic video analysis,” in _International Conference on Medical Image Computing and Computer-Assisted Intervention_ . Springer, 2023, pp. 569–578. 

- [22] Q. Tian, H. Liao, X. Huang, B. Yang, D. Lei, S. Ourselin, and H. Liu, “Endomamba: an efficient foundation model for endoscopic videos via hierarchical pre-training,” in _International Conference on Medical Image Computing and Computer-Assisted Intervention_ . Springer, 2025, pp. 224–234. 

- [23] T. J. Jaspers, R. L. de Jong, Y. Li, C. H. Kusters, F. H. Bakker, R. C. van Jaarsveld, G. M. Kuiper, R. van Hillegersberg, J. P. Ruurda, W. M. Brinkman _et al._ , “Scaling up self-supervised learning for improved surgical foundation models,” _arXiv preprint arXiv:2501.09436_ , 2025. 

- [24] K. Yuan, V. Srivastav, T. Yu, J. L. Lavanchy, J. Marescaux, P. Mascagni, N. Navab, and N. Padoy, “Learning multi-modal representations by watching hundreds of surgical video lectures,” _Medical Image Analysis_ , p. 103644, 2025. 

- [25] R. Stauder, D. Ostler, M. Kranzfelder, S. Koller, H. Feußner, and N. Navab, “The tum lapchole dataset for the m2cai 2016 workflow challenge,” _arXiv preprint arXiv:1610.09278_ , 2017. 

- [26] C. I. Nwoye, T. Yu, C. Gonzalez, B. Seeliger, P. Mascagni, D. Mutter, J. Marescaux, and N. Padoy, “Rendezvous: Attention mechanisms for the recognition of surgical action triplets in endoscopic videos,” _Medical Image Analysis_ , vol. 78, p. 102433, 2022. 

- [27] Z. Wang, B. Lu, Y. Long, F. Zhong, T.-H. Cheung, Q. Dou, and Y. Liu, “Autolaparo: A new dataset of integrated multi-tasks for image-guided surgical automation in laparoscopic hysterectomy,” in _International Conference on Medical Image Computing and Computer-Assisted Intervention_ . Springer, 2022, pp. 486–496. 

- [28] D. Guo, W. Si, Z. Li, J. Pei, and P.-A. Heng, “Surgical workflow recognition and blocking effectiveness detection in laparoscopic liver resection with pringle maneuver,” in _Proceedings of the AAAI Conference on Artificial Intelligence_ , vol. 39, no. 3, 2025, pp. 3220–3228. 

- [29] M. Hu, P. Xia, L. Wang, S. Yan, F. Tang, Z. Xu, Y. Luo, K. Song, J. Leitner, X. Cheng _et al._ , “Ophnet: A large-scale video benchmark for ophthalmic surgical workflow understanding,” in _European Conference on Computer Vision (ECCV)_ . Springer, 2024. 

- [30] E. D. Goodman, K. K. Patel, Y. Zhang, W. Locke, C. J. Kennedy, R. Mehrotra, S. Ren, M. Guan, O. Zohar, M. Downing _et al._ , “Analyzing surgical technique in diverse open surgical videos with multitask machine learning,” _JAMA Surgery_ , 2024. 

21 

- [31] N. Ahmidi, L. Tao, S. Sefati, Y. Gao, C. Lea, B. B. Haro, L. Zappella, S. Khudanpur, R. Vidal, and G. D. Hager, “A dataset and benchmarks for segmentation and recognition of gestures in robotic surgery,” _IEEE Transactions on Biomedical Engineering_ , vol. 64, no. 9, pp. 2025–2041, 2017. 

- [32] H. Hoffmann, I. Funke, P. Peters, D. K. Venkatesh, J. Egger, D. Rivoir, R. R¨ohrig, F. H¨olzle, S. Bodenstedt, M.-C. Willemer, S. Speidel, and B. Puladi, “Aixsuture: vision-based assessment of open suturing skills,” _International Journal of Computer Assisted Radiology and Surgery_ , vol. 19, no. 6, pp. 1045–1052, 2024. 

- [33] K. Schoeffmann, H. Husslein, S. Kletz, S. Petscharnig, B. M¨unzer, and C. Beecks, “Video retrieval in laparoscopic video recordings with dynamic content descriptors,” _Multimedia Tools and Applications_ , vol. 77, no. 13, pp. 16 813–16 832, 2018. 

- [34] Y. Tian, G. Pang, F. Liu, Y. Liu, C. Wang, Y. Chen, J. Verjans, and G. Carneiro, “Contrastive transformerbased multiple instance learning for weakly supervised polyp frame detection,” in _International Conference on Medical Image Computing and Computer-Assisted Intervention_ . Springer, 2022, pp. 88–98. 

- [35] J. Bernal, F. J. S´anchez, G. Fern´andez-Esparrach, D. Gil, C. Rodr´ıguez, and F. Vilari˜no, “Wm-dova maps for accurate polyp highlighting in colonoscopy: Validation vs. saliency maps from physicians,” _Computerized Medical Imaging and Graphics_ , vol. 43, pp. 99–111, 2015. 

- [36] A. Rau, P. E. Edwards, O. F. Ahmad, P. Riordan, M. Janatka, L. B. Lovat, and D. Stoyanov, “Implicit domain adaptation with conditional generative adversarial networks for depth prediction in endoscopy,” _International Journal of Computer Assisted Radiology and Surgery_ , vol. 14, pp. 1167–1176, 2019. 

- [37] T. L. Bobrow, M. Golhar, R. Vijayan, V. S. Akshintala, J. R. Garcia, and N. J. Durr, “Colonoscopy 3d video dataset with paired depth from 2d-3d registration,” _Medical Image Analysis_ , p. 102956, 2023. 

- [38] H. Al Hajj, M. Lamard, P.-H. Conze, B. Cochener, and G. Quellec, “Cataracts: Challenge on automatic tool annotation for cataract surgery,” _Medical Image Analysis_ , vol. 52, pp. 24–41, 2019. [Online]. Available: https://dx.doi.org/10.21227/ac97-8m18 

- [39] J. L. Lavanchy, S. Ramesh, D. Dall’Alba, C. Gonzalez, P. Fiorini, B. P. M¨uller-Stich, P. C. Nett, J. Marescaux, D. Mutter, and N. Padoy, “Challenges in multi-centric generalization: phase and step recognition in roux-en-y gastric bypass surgery,” _International Journal of Computer Assisted Radiology and Surgery_ , vol. 19, pp. 2249–2258, 2024. 

- [40] G. Wang, H. Xiao, R. Zhang, H. Gao, L. Bai, X. Yang, Z. Li, H. Li, and H. Ren, “Copesd: A multilevel surgical motion dataset for training large vision-language models to co-pilot endoscopic submucosal dissection,” in _Proceedings of the 33rd ACM International Conference on Multimedia_ , 2024. 

- [41] N. Valderrama, O. Zisimopoulos, and S. Giannarou, “Towards holistic surgical scene understanding,” in _International Conference on Medical Image Computing and Computer-Assisted Intervention_ . Springer, 2023, pp. 442–452. 

- [42] D. Jha, P. H. Smedsrud, M. A. Riegler, P. Halvorsen, T. de Lange, D. Johansen, and H. D. Johansen, “Kvasir-seg: A segmented polyp dataset,” in _MultiMedia Modeling: 26th International Conference, MMM 2020, Daejeon, South Korea, January 5–8, 2020, Proceedings, Part II 26_ , 2020, pp. 451–462. 

- [43] D. V´azquez, J. Bernal, F. J. S´anchez, G. Fern´andez-Esparrach, A. M. L´opez, A. Romero, M. Drozdzal, and A. Courville, “A benchmark for endoluminal scene segmentation of colonoscopy images,” _Journal of healthcare engineering_ , vol. 2017, no. 1, p. 4037190, 2017. 

- [44] J. Bernal, J. S´anchez, and F. Vilarino, “Towards automatic polyp detection with a polyp appearance model,” _Pattern Recognition_ , vol. 45, no. 9, pp. 3166–3182, 2012. 

22 

- [45] J. Silva, A. Histace, O. Romain, X. Dray, and B. Granado, “Toward embedded detection of polyps in wce images for early diagnosis of colorectal cancer,” _International journal of computer assisted radiology and surgery_ , vol. 9, no. 2, pp. 283–293, 2014. 

- [46] D.-P. Fan, G.-P. Ji, T. Zhou, G. Chen, H. Fu, J. Shen, and L. Shao, “Pranet: Parallel reverse attention network for polyp segmentation,” in _International conference on medical image computing and computerassisted intervention_ . Springer, 2020, pp. 263–273. 

- [47] T. Kim, H. Lee, and D. Kim, “Uacanet: Uncertainty augmented context attention for polyp segmentation,” in _Proceedings of the 29th ACM international conference on multimedia_ , 2021, pp. 2167–2175. 

- [48] B.-C. Hu, G.-P. Ji, D. Shao, and D.-P. Fan, “Pranet-v2: Dual-supervised reverse attention for medical image segmentation,” _arXiv preprint arXiv:2504.10986_ , 2025. 

23 

