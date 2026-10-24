## Multimodal Graph Representation Learning for Robust Surgical Workflow Recognition with Adversarial Feature Disentanglement⋆

Long Baia,b,1, Boyi Maa,c,1, Ruohan Wangd, Guankun Wanga, Beilei Cuia, Zhongliang Jiangb, Mobarakol Islame, Zhe Mine,f, Jiewen Laia, Nassir Navabb and Hongliang Rena,∗

aDepartment ofElectronic Engineering, The Chinese University ofHong Kong, Hong Kong, China

bChair for Computer Aided Medical Procedures, Technical University ofMunich, Munich, Germany

cDepartment ofBiomedical Engineering, University ofToronto, Toronto, Canada

dCenter for Computational and Molecular Biology, Brown University, Providence, United States

eUCL Hawkes Institute, University College London, London, UK

fSchool ofControl Science and Engineering, Shandong University, Jinan, China

## ART I CLE I NFO

Keywords: surgical data science workflow recognition multimodal fusion adversarial learning robustness

## ABSTRACT

Surgical workflow recognition is vital for automating tasks, supporting decision-making, and training novice surgeons, ultimately improving patient safety and standardizing procedures. However, data corruption can lead to performance degradation due to issues like occlusion from bleeding or smoke in surgical scenes and problems with data storage and transmission. Therefore, a robust workflow recognition model is urgently needed. In this case, we explore a robust graph-based multimodal approach to integrating vision and kinematic data to enhance accuracy and reliability. Vision data captures dynamic surgical scenes, while kinematic data provides precise movement information, overcoming limitations of visual recognition under adverse conditions. We propose a multimodal Graph Representation network with Adversarial feature Disentanglement (GRAD) for robust surgical workflow recognition in challenging scenarios with domain shifts or corrupted data. Specifically, we introduce a Multimodal Disentanglement Graph Network (MDGNet) that captures fine-grained visual information while explicitly modeling the complex relationships between vision and kinematic embeddings through graph-based message modeling. To align feature spaces across modalities, we propose a Vision-Kinematic Adversarial (VKA) framework that leverages adversarial training to reduce modality gaps and improve feature consistency. Furthermore, we design a Contextual Calibrated Decoder, incorporating temporal and contextual priors to enhance robustness against domain shifts and corrupted data. Extensive comparative and ablation experiments demonstrate the effectiveness of our model and proposed modules. Specifically, we achieved an accuracy of 86.87% and 92.38% on two public datasets, respectively. Moreover, our robustness experiments show that our method effectively handles data corruption during storage and transmission, exhibiting excellent stability and robustness. Our approach aims to advance automated surgical workflow recognition, addressing the complexities and dynamism inherent in surgical procedures.

offering excellent precision, stability, and control that sig- nificantly exceeds the capabilities of the human hand [15, 26, 89]. Robotic systems, such as the da Vinci Surgical System, have been widely adopted across various surgical disciplines [72, 86], demonstrating significant benefits like minimally invasive procedures, reduced patient recovery times, and improved surgical outcomes [12, 87, 90]. These sophisticated robots act as extensions of the surgeon’s exper- tise, enabling complex operations to be performed through small incisions with enhanced visualization and dexter- ity [22, 48]. As the adoption of robotic-assisted surgery continues to grow, the demand for intelligent systems that can recognize and understand surgical workflows becomes increasingly significant [6, 13, 63]. [URL 🔗](#page-0)

Artificial Intelligence (AI)-assisted workflow recogni- tion in robotic surgery serves multiple essential purposes [5, 36, 38, 71]. It is the key to automating routine tasks, pro- viding decision-making support, and enabling context-aware assistance during operations [7, 88]. Furthermore, under- standing surgical workflows is crucial for training junior surgeons, as it enables the objective assessment of technical skills and procedural knowledge [59]. By analyzing the [URL 🔗](#page-0)

## 1. Introduction

The advent of robot-assisted minimally invasive surgery (RMIS) has marked a transformative era in modern medicine,

∗Corresponding author: H. Ren.

ORCID(s): 0000-0002-9762-6821 (L. Bai); 0009-0001-6913-6171 (B.

Ma); 0009-0005-2146-2292 (R. Wang); 0000-0003-2440-4950 (G. Wang); 0009-0009-7900-8032 (B. Cui); 0000-0001-7461-2200 (Z. Jiang); 0000-0002-7162-2822 (M. Islam); 0000-0002-8903-1561 (Z. Min); 0000-0002-2676-7387 (J. Lai); 0000-0002-6032-5611 (N. Navab); 0000-0002-6488-1551 (H. Ren)

1Co-first authors: L. Bai and B. Ma.


*Figure 1: Comparison between the vision-based workflow recognition framework and the multimodal workflow recogni- tion framework.*

sequence and structure of surgical tasks, workflow recogni- tion systems can provide real-time feedback, facilitate error correction, and contribute to the standardization of surgical procedures, ultimately leading to improved patient safety and better training methodologies [10]. [URL 🔗](#page-0)

Vision-based surgical workflow recognition has sig- nificantly advanced [81]. Vision data, derived from high- resolution cameras, captures the dynamic surgical scene, allowing for the visual identification of instruments, tis- sue interactions, and operative gestures [39, 105]. These methods show promise in accurately identifying surgical phases and tool usage. However, they face challenges in complex scenarios, such as occlusions or variable lighting. Similarly, kinematic-based recognition systems focus on the mechanical aspects of surgery, analyzing movement patterns and instrument trajectories to deduce ongoing tasks [20]. While these systems are less affected by visual interference, they often lack the context that visual cues provide, which is crucial for a comprehensive understanding of the surgical workflow. Kinematic systems might also misinterpret com- plex instrument interactions. In this case, despite advance- ments in single-modality systems, their isolated application is often inadequate for capturing the full spectrum ofsurgical scenes. This necessitates exploring multimodal workflow recognition approaches that take advantage of both visual and kinematic data for enhanced performance and reliability. [URL 🔗](#page-0)

Currently, the fusion ofvision and kinematic data presents a compelling approach that promises enhanced accuracy and reliability [59]. Visual data effectively provides the context and semantic information needed for workflow analysis. Meanwhile, kinematic data, which includes the precise movements and positions of robotic arms and instrument tips, is immune to visual obstructions and offers additional information not always discernible through vision alone. [URL 🔗](#page-0)

These two data modalities complement each other, with vision providing context and kinematic data offering unam- biguous motion patterns [33, 59]. Therefore, as presented in Figure 1, a multimodal system shall enable a more complete and detailed interpretation of surgical activities, leading to a robust workflow recognition system that can adapt to the complex and variable nature of surgical environments. [URL 🔗](#page-0)

However, current vision-kinematics methods primarily concentrate on fusing different modalities without ade- quately exploring the rich features within each modality. This limitation can impede the model’s ability to compre- hend complex surgical scenarios, as it may not capture the unique higher-level semantic information inherent to each data type. Optimizing each modality separately can more effectively uncover and leverage the intrinsic characteristics of each data type, thereby enhancing overall performance. Moreover, in surgical and endoscopic environments, data quality may degrade due to camera movement, occlusions, or noise and quality loss during data transmission. Therefore, robust automatic recognition of surgical workflows is crucial to deal with noise and disturbances typical in surgical settings. Current approaches might not sufficiently address the need for robustness, potentially leading to unstable performance in real-world operating environments. Hence, our method needs to be robust enough to handle explicit data corruption issues, ensuring clinical-level accuracy. These identified weaknesses highlight the necessity for improvements in multimodal data processing and robustness in surgical workflow recognition systems. We will focus on strategies that enhance robustness through adversarial discrimination and graph calibration loss. Additionally, we will further strengthen the feature representation of visual nodes through disentanglement to better handle the complexity and dynamism of surgical procedures. Such strategies would contribute to advancing the accuracy and stability of automated surgical workflow recognition.

To tackle the above challenges, we propose a multi- modal Graph Representation network with Adversarial fea- tureDisentanglement (GRAD) to perform the surgical work- flow recognition with vision and kinematics data. We first developed a Multimodal Disentanglement Graph Network (MDGNet) to explore and model the intra-modal and inter- modal feature connections between vision and kinematic modalities. Initially, we explore the unique features of each modality to enhance internal feature learning. For visual data, we delve deeply into disentangling features across three domains: spatial, wavelet, and Fourier, and propose the disentanglement graph learning strategy. This multi-domain approach allows us to disentangle complex patterns and representations that may not be apparent in the raw data. To capture the dynamic aspects of visual data, we employ Tem- poral Convolutional Networks (TCNs), specifically designed to extract time-related characteristics from sequential data. For kinematics data regarding object motion, we use both TCNs and Long Short-Term Memory (LSTM) networks to extract temporal movement features. By fully decoupling intra-modal information, we can effectively distinguish and


utilize the independence and complementarity of each data modality. Subsequently, the attention-based graph network models the kinematic information and visual data as nodes to learn relational representations, while the attention mech- anism strengthens interdependencies in complex data.

Moreover, to impose more effective constraints on the multimodal feature space, and to align the distributions of different modalities, we propose a Vision-Kinematic Adver- sarial (VKA) learning strategy. By optimizing a discrimina- tive joint feature space, our strategy brings the multimodal feature distributions closer together, thereby reducing inter- modal discrepancies. Finally, we propose our contextual calibrated decoder to maintain high robustness against data corruption. Graph networks may have low confidence due to the shallow layers. We integrate graph embeddings with the original multimodal feature embeddings and use a regular- ized calibration optimization function to address prediction confidence issues, significantly improving the adaptiveness and stability of our model when dealing with corrupted data (e.g., data with noise, blur, occlusion, or digital corruptions). The main contributions of this work are as follows:

- We present the GRAD framework, a robust graph representation learning framework for surgical work- flow recognition using visual and kinematic data. Our architecture leverages the MDGNet to establish in- teractions between different modalities based on in- depth information exploration within each modality, providing an effective solution for surgical workflow analysis.

- We further incorporate Vision-Kinematic Adversarial (VKA) learning, which forces the model to align multimodal feature distribution during the pre-fusion stage. Additionally, our contextual calibrated decoder further strengthens the output confidence and en- hances robustness.

- We validate our proposed GRAD framework on two publicly available robotic surgery datasets. Our ap- proach achieves superior performance compared to both single-modality and multimodal methods. Our ablation studies further demonstrate the effectiveness of the proposed components. Moreover, in robustness tests, our method shows more stable performance and slower performance degradation, proving its real- world clinical efficacy.

## 2. Related Work

## 2.1. Multimodal Graph Learning

Over the past several years, graph representation learn- ing has gained significant attention in many different do- mains [65, 11, 93, 61, 111, 110]. The relationships and de- pendencies between different modalities can be better repre- sented graphically, therefore, graph representation learning has been verified as an effective way to implement multi- feature fusion. In recent studies, an increasing number of [URL 🔗](#page-0)

multimodal fusion tasks demonstrated great performance with graph representation learning in the medical domain. Zeng et al. [108] utilized a multilayer graph convolutional network (GCN) to fuse multimodal information, effectively capturing long-range contextual dependencies in the text while filtering out visual noise unrelated to the textual con- tent. Zhu et al. [112] introduced a feature fusion mecha- nism that grouped pixels with similar characteristics into nodes and transformed spatial domain input features into the graph domain. Hou et al. [30] developed a hybrid GCN that combines the strengths of traditional GCNs and hyper- graph convolutional networks (HCNs). This model lever- ages node message passing to enable both intra-modal and inter-modal interactions within multimodal graphs. Yue et al. [106] improved fusion effectiveness by employing a GCN to integrate diverse features from multiple data modalities, utilizing graph topology to propagate messages and repre- sent node features along with their interrelations through the graph structure. Qu et al. [74] considered subjects’ rela- tionships within and between modalities by learning a graph embedding from multimodal data. Li et al. [49] suggested a 3D Haar semi-tight graph learning method to integrate data from various modalities, e.g., vision, text, audio, and statistical information. Meng et al. [62] utilized a masked graph network to integrate various modalities for emotion recognition and prediction. [URL 🔗](#page-0)

Currently, initial works use graph learning to fuse and model the multimodal perception information in surgical scenarios. However, existing work mainly employs simple multimodal modeling using graph-based methods without deeply exploring the performance of different modalities in surgery or researching more robust graph solutions. There- fore, we aim to develop effective and robust graph represen- tation learning methods to further enhance the real-world application potential of surgical workflow recognition.

## 2.2. Surgical Workflow Recognition

Automatic surgical workflow recognition is vital in mod- ern intelligent operating rooms as it provides context aware- ness for computer-assisted systems [3, 53, 57, 58, 103, 102, 104]. Early methods for phase recognition relied on hand-crafted visual features and temporal models like Hid- den Markov Models (HMMs) [18]. The emergence of deep learning techniques significantly improved phase recogni- tion performance. Convolutional Neural Networks (CNNs) became the standard for extracting visual features from individual frames, while LSTMs and TCNs were employed to model temporal dependencies in video sequences [37]. Several studies have focused on effectively combining spa- tial and temporal information. SV-RCNet [36] seamlessly integrates a ResNet and an LSTM network, enabling joint optimization of visual and temporal features during training. Recognizing that surgical phases often exhibit long-range temporal dependencies, researchers have explored methods for incorporating longer temporal contexts. TMRNet [38] introduces a temporal memory relation network that utilizes a long-range memory bank and a temporal variation layer [URL 🔗](#page-0)


to encode long-range and multi-scale temporal information. TeCNO [16] employs a multi-stage TCN with dilated convo- lutions to achieve a larger receptive field for capturing longer temporal dependencies. Trans-SVNet [39] leverages a hy- brid embedding aggregation transformer to fuse spatial and temporal embeddings, allowing for the retrieval of critical information from longer temporal sequences. [URL 🔗](#page-0)

Integrating visual data alone for surgical workflow recog- nition can be unreliable due to variations in surgical pro- cedures, lighting, camera angles, and noise from blood or smoke [59]. Kinematics data, however, is more robust, capturing the position and orientation of instruments and providing valuable guidance for visual analysis. Recent methods, such as multimodal graph learning [55, 59] and [URL 🔗](#page-0)

feature fusion techniques [1, [URL 🔗](#page-0)

73], model and fuse relational [URL 🔗](#page-0)

features from both image and kinematics data. This approach allows models to learn useful representations of instrument tips unaffected by changes in the surgical scene. Developing robust models for surgical workflow recognition is challeng- ing due to variations in surgical data from differing tech- niques, imaging equipment, and domain shifts. To address these challenges, multi-modal fusion can enhance model robustness, facilitating successful deployment in clinical settings. Developing robust models that can handle data variations and corrupted inputs is critical for successfully translating surgical workflow recognition technology into real-world operating rooms.

## 2.3. Network Calibration

Calibration in deep learning adjusts predicted proba- bilities to match true outcome likelihoods [85]. Deep neu- ral networks often give overconfident predictions, affecting critical decisions. Methods to address this include post- hoc adjustments, regularization, and uncertainty estimation. Common post-hoc methods are Platt scaling [69], Bayesian binning [64], and isotonic regression [107]. Recent work expands these, such as bias removal [32] and kernel density estimation [70]. Regularization techniques include L2 [25], entropy [68], and focal loss [52]. New research explores implicit and explicit regularization [50], adaptive binning [8] and class-distribution-aware calibration [34]. Uncertainty estimation uses Bayesian networks [9, 21], ensembles [43], [URL 🔗](#page-0)

and Gumbel-softmax [35, [URL 🔗](#page-0)

further refined uncertainty estimation techniques. Kim et al. [41] replaces traditional softmax confidence scores with energy scores. Proper calibration allows machine learning classifiers to deliver trustworthy and interpretable predic- tions, effectively aligning the gap between the model’s pre- dictions and real probabilities. [URL 🔗](#page-0)

Graph Neural Networks (GNNs) learn from graph- structured data, unlike traditional deep learning methods that handle independent data. Traditional calibration methods are ineffective for GNNs [80], which often show under- confidence [31, 91]. Tailored calibration for GNNs is cru- cial for trustworthiness. Recent techniques include trans- forming logits to calibrated confidence for nodes [94], Graph Attention Temperature Scaling [31], topology-aware [URL 🔗](#page-0)

67]. [URL 🔗](#page-0)

Recent advancements have

calibration with cross-entropy and ECE [92], Adaptive Spectral Temperature Scaling for local structures [96], and analyzing decisive and homophilic edges [99]. Numerous studies highlight the susceptibility of deep learning to com- mon data distortions such as blurring and Gaussian noise [19]. Recent research indicates that calibration methods, ensuring that predicted probabilities accurately reflect true outcomes, not only boost models’ accuracy but also enhance the robustness [66]. This alignment of predictions with reality is vital for maintaining performance and security in dynamic environments. Thus, the well-calibrated models provide trustworthy uncertainty estimates, making them less vulnerable to real-world noise attacks [32, 56]. [URL 🔗](#page-0)

## 3. Methodology

## 3.1. Preliminaries

Long et al. [59] dynamically integrate visual and kine- matics information in the latent feature space of the rela- tional graph convolutional network (RGCN). It first extracts embeddings from video sequences using temporal CNNs and from kinematics data of robotic arms using TCNs and LSTMs. These multimodal embeddings are then fed into a relational graph learning module, which identifies three re- lation types between nodes corresponding to video, left arm kinematics, and right arm kinematics. The module performs interactive message passing between nodes of different rela- tion types through hierarchical graph convolutional layers. This allows the complementary information to be fused from visual and kinematics modalities through learning mul- tiple relation-specific weights. Finally, the updated node representations are used for gesture classification through a prediction head. [URL 🔗](#page-0)

## 3.1.1. Graph Network for Vision & Kinematics

## 3.1.2. Graph Attention Network

Using the masked self-attentional layers, graph attention networks (GATs) [83] allow assigning varying importance to different nodes without costly matrix operations or prior knowledge of the graph structure. Compared to other fusion methods, by learning the attention weights to be assigned, we can allocate appropriate focus on different modalities, thereby facilitating the integration and complementation of [URL 🔗](#page-0)

{

multimodal. The input is denoted as: 𝐡 =

ℎ⃗𝑖 ∈ ℝ𝐹, where 𝑁 denotes the number of nodes and 𝐹

represents the number of features associated with each node.

To obtain higher-level features, a shared linear trans- formation parametrized by a weight matrix, 𝐖 ∈ 𝐹′×𝐹, is first applied to every node, after which self-attention is performed to compute attention coefficients, allowing each node to interact with every other node. The importance of node 𝑗’s feature to node 𝑖 shall be formulated as:

where 𝑎𝑡𝑡𝑛 ∶ ℝ𝐹′ × ℝ𝐹′ → ℝ denotes a shared attentional mechanism. The 𝑒𝑖𝑗 is only computed for nodes 𝑗 ∈ N𝑖,

ℎ⃗1, ⃗ℎ2,⋯, ⃗ℎ𝑁 } ,


*Figure 2: The overall architecture of our GRAD framework, consisting of the Multimodal Disentanglement Graph Network (Visual Representation Disentanglement module, the Kinematic Temporal Representation Extraction module, the Multimodal Graph Learning module), the Vision-Kinematic Adversarial Training, and the Calibrated Prediction Decoder. Our GRAD framework extracts visual and kinematic features through intra-modal feature mining. It uses adversarial learning to approximate the representation distributions of both. The graph network facilitates the interaction between the two modalities, and network calibration ensures robust prediction results.*

where 𝐍𝑖 is some neighborhood of node 𝑖 in the graph, thus, enhancing the model’s capacity to capture accurate relationships between various modalities.

## 3.1.3. Adversarial Learning

Adversarial learning usually comprises two main com- ponents: a generator and a discriminator. It was initially developed to tackle the generative modeling problem [23]. The generator aims to learn the underlying probability dis- tributions and generate synthetic data based on the training samples. Conversely, the discriminator’s objective is to dif- ferentiate between generated examples and real data. Dur- ing training, the generator tries to minimize the difference between synthetic and real data, while the discriminator seeks to maximize the accuracy of distinguishing fake sam- ples from real ones. This competitive dynamic helps the generator capture the true data distributions in the training examples. Given these characteristics, adversarial learning [URL 🔗](#page-0)

is well-suited for modality distribution translation, offering a promising approach for robust model training.

## 3.2. Proposed Framework: GRAD

Our GRAD framework, as presented in Figure 2, com- prises a visual representation learning branch with feature disentanglement, a kinematic feature extractor to extract sequential kinematics information, a graph attention network for multimodal fusion, the multimodal discriminator for adversarial learning, and the calibrated prediction decoder as the output module. [URL 🔗](#page-0)

## 3.2.1. Multimodal Disentanglement Graph Network

We propose the Multimodal Disentanglement Graph Network (MDGNet) to encode and model the multimodal embeddings from vision and kinematics data. First, we transform the kinematic signals into feature embeddings. The kinematic information is collected by the robotic plat- form, wherein each manipulator records seven degrees of


freedom (DoFs): three for positional coordinates, three for orientation, and one for the gripper’s opening angle. These values collectively describe the translation and rotation of a robotic manipulator in three-dimensional space. The task space can be presented as a 7-dimensional vector: T = [𝑥𝑙, 𝑦𝑙, 𝑧𝑙, 𝛼𝑙, 𝛽𝑙, 𝛾𝑙, 𝜃𝑙], where 𝑥𝑟, 𝑦𝑟, 𝑧𝑟 are the Cartesian coordinates that specify the position of the left end effector, 𝛼𝑙, 𝛽𝑙, 𝛾𝑙 represent the left arm’s orientation in term of roll, pitch, and yaw angles, 𝜃𝑙 denotes the opening angle of the left gripper. The same concept applies to the right robotic arm. Eventually, the kinematic information for two robotic arms at each frame is represented by 14 numerical values. The MISAW dataset further provides the output grip voltage for reference, so the kinematic information of the MISAW dataset is in the format of 16 numerical values. Extracting effective information from a single frame of kinematic data is challenging due to its sparse and complex nature, necessitating the modeling of temporal information over sequences. To preserve the fine-grained details and extract more temporally correlated information, we designed a temporal feature extractor that combines LSTM [24] with TCN [44] in parallel. By stacking a hierarchy of temporal convolutions, pooling, and upsampling operations, the TCN addresses the issue of independence of kinematic informa- tion over time, mitigating blurry and incoherent recognition. Using the multi-scale hierarchical structure, the TCN shall capture detailed relations between timestamps within the kinematics data. The LSTM maps the previous kinematics vector and previous hidden state to a new hidden state, capturing the long-range dependencies within the kinematic data. We then use the average of the LSTM and TCN outputs as the kinematic representations for subsequent multimodal fusion. The extraction process of the kinematic features 𝑥𝑘 𝑡 , 𝑡 ∈ {1, 2,…, 𝑇} is described as: [URL 🔗](#page-0)

= LSTM(𝑥𝑡) + TCN(𝑥𝑡) × 0.5 [ ]

Consequently, we have achieved the modeling of kinematic information on a large temporal scale. We combine the left and right kinematic features for training, as our experiments demonstrate that this approach brings better performance compared to training them separately.

Second, we disentangle the visual signal into spatial and frequency domains and extract feature representations. Understanding and interpreting visual data from surgical videos poses a significant challenge due to the rich and detailed information they contain. This includes various movements of instruments, subtle surgical procedures, and similar backgrounds throughout the videos. Distinguishing between different surgical phases within the same video can be highly complex. Current multimodal graph networks primarily consider the spatial domain while processing vi- sual signals; however, the frequency domain can further preserve low-level statistical data [97] and modality-specific information [101]. Therefore, to extract complex patterns in visual information that may not be discernible in raw visual data, we aim to provide explicit supervision of visual feature [URL 🔗](#page-0)

information by jointly mining the spatial and frequency do- mains. We further exploit the multi-level structural informa- tion and fundamental texture information of images through wavelet transform [4] and Fourier amplitude spectra [109], enabling the model to retain modality-invariant content and enhancing its robustness against corrupted inputs. [URL 🔗](#page-0)

We employwavelet and Fourier transforms for frequency domain analysis and feature extraction. Wavelet transforms capture image features at different scales, aiding the model in understanding the multi-level structure from coarse to fine details, and better recognizing local features such as edges, corners, and textures. They also exhibit high robustness to noise and variations under non-ideal conditions. The for- mula for extracting wavelet features is provided below:

where the 𝑥(ℎ, 𝑤) represents the image in the spatial domain. 𝐴𝑗 (𝑚, 𝑛), 𝐷𝐿𝑍 𝑗 (𝑚, 𝑛), 𝐷𝑍𝐿 𝑗 (𝑚, 𝑛) and 𝐷𝑍𝑍(𝑚, 𝑛) 𝑗 denotes the approximation coefficients, horizontal detail coefficients, vertical detail coefficients and diagonal detail coefficients, respectively. Here, 𝜙𝑗,𝑚, 𝜓𝑍 𝑗,𝑚 and 𝜓𝑉 𝑗,𝑚 are the scaling func- tion (father wavelet), horizontal and vertical wavelet func- tions (mother wavelet).

Fourier amplitude spectra reveal the intensity of various frequency components in an image, assisting the model in understanding the fundamental textures and structural patterns while emphasizing global information for a better overall understanding of the image content. The formula for extracting Fourier amplitude spectra is provided below:

where the 𝑥(ℎ, 𝑤) represents the spatial domain image and the 𝐹(𝑝, 𝑞) denotes each point in the Fourier domain. The discrete Fourier amplitude spectra are defined by the square root of the sum of the real and imaginary parts of 𝐹(𝑝, 𝑞).

Specifically, for visual data in the spatial, wavelet, and Fourier domains, we establish the same feature extraction strategy. Given the spatial domain as an example: 𝐼𝑡, 𝑡 ∈ {1, 2,⋯, 𝑇} denote the image frames from the video se- quence. We first utilize the custom-trained ResNet-18 [27] [URL 🔗](#page-0)


as the feature extraction backbone, which transforms 224 × 224 × 3 RGB images into a spatial feature representation. Then, TCN is implemented to extract video features 𝑥𝑖 𝑡, 𝑡 ∈ {1, 2,⋯, 𝑇} with long-range temporal patterns from a series

of frame-wise features. The process can be formulated as:

𝑥𝑖 𝑡 = TCN { 𝐷𝑟𝑜𝑝𝑜𝑢𝑡 [ ReLU CNN(𝐼𝑡) [ ]]}

where a dropout rate of 0.5 is utilized to avoid overfitting. The same visual feature extraction strategy is also applied in the wavelet and Fourier domains. The obtained feature representation can be represented as 𝑥𝑤 𝑡 , 𝑡 ∈ {1, 2,⋯, 𝑇} and 𝑥𝑓 𝑡 , 𝑡 ∈ {1, 2,⋯, 𝑇}. The spatial, wavelet, and Fourier domains shall offer complementary and comprehensive im- age representations [78, 79]. The spatial domain provides a direct representation of the original image. The wavelet domain enables multiresolution analysis, effectively captur- ing details and overall structures at different scales, mak- ing it particularly suitable for handling data with varying spatiotemporal characteristics. The Fourier domain, by an- alyzing the frequency components of an image, helps in understanding its global structure and patterns. By lever- aging the complementary information from these domains and integrating global and local features, the model’s ability to resist various types of noise and disturbances can be enhanced, further strengthening its robustness. [URL 🔗](#page-0)

Finally, we establish our backbone with GAT [84] to model the relational graph of visual and kinematic repre- sentations. Unlike previous methods that use stacked GCN layers [42, 76] to update features based on fixed adjacent matrix, which fail to grasp the inherent relationship be- tween different modalities effectively, attention-based graph networks can learn the intrinsic interplay among diverse modalities and assign different importance levels to nodes corresponding to various modalities [82]. This allows for better cross-modal information exchange as the model can discern certain inherent relationships dynamically during the learning process. The effectiveness of GAT is also veri- fied in Section 4.3.4. We consider the three visual domains (spatial, wavelet, Fourier) and the combined kinematic em- bedding as four sets of graph node features, denoted as 𝐧 = { 𝑥⃗𝑖 𝑡, ⃗𝑥𝑤 𝑡 , ⃗𝑥𝑓 𝑡 , ⃗𝑡𝑥𝑘 } . These nodes are then refined through an aggregation process that incorporates messages from other nodes in the neighborhood. Taking spatial embeddings as an example, the formulation of the updated nodes (⃗𝑥𝑖 𝑡)′ is as follows: [URL 🔗](#page-0)

where 𝐾 represents the number of independent attention mechanisms, and 𝐖⋅ denotes the weight matrix correspond- ing to the input linear transformation, 𝛼𝑖𝑤 denotes the atten- tion score between nodes 𝑖 & 𝑤, other attention scores can be inferred in the same way. To enable comparison of the attention coefficients across different nodes, the nodes in a

neighborhood of 𝑖 are normalized using the softmax func- tion. Taking 𝛼𝑖𝑤 as an example, the function is calculated:

𝐖⃗𝑛𝑖||𝐖⃗𝑛𝑤

)

where ⃗𝐚 is a weight vector and ⋅𝑇 represents its transposition. In this formula, 𝐖 ∈ ℝ𝐹′×𝐹 is a weight matrix where 𝐹 is the number of features in each node. By leveraging the attention mechanism in graph networks, the identification of critical associations can be enhanced while disregarding irrelevant information, thereby facilitating the capture of both intra-modal contextual information and inter-modal complementary information.

## 3.2.2. Vision-Kinematic Adversarial Training

Given that the visual and kinematic representations be- long to different modalities, a significant disparity exists be- tween their statistical properties. Previous study shows that matching those distributions in separate embedding spaces will significantly enhance the model’s sensitivity to irrel- evant perturbations [60, 77]. Therefore, mapping different representations into a common embedding space can reduce the gap between different modalities [54, 95]. In this case, we propose Vision-Kinematic Adversarial (VKA) learning to align them within a shared embedding space, which can facilitate the correlation of heterogeneous data from kinematic and visual modalities. Consequently, when facing random noise in real applications, the model can leverage information from alternative sources for compensation. [URL 🔗](#page-0)

The multimodal adversarial learning implements a dis- criminator 𝐷 to distinguish the target modality as true but other source modalities as false, while the generator aims to fool the discriminator into classifying the target modal- ity as true. The generator and discriminator engage in an adversarial, min-max game to learn a modality-invariant embedding space that allows diverse and complex modalities to be seamlessly matched. The loss function for this model

can be broken down into two parts: fake adversarial loss

and the true adversarial loss L𝑡𝑟𝑢. Through adding constraints to varying distributions, adversarial learning can learn com- mon representations to effectively correlate heterogeneous information from diverse modalities.

We have four inputs from two different modalities: 𝑥𝑖 𝑡, 𝑥𝑤 𝑡 , and 𝑥𝑓 𝑡 from the visual modality, and 𝑥𝑘 𝑡 from the kine-

matic modality. We consider

𝑡

the others are source modalities. The rationale of treating kinematic as the target modality has been empirically ver- ified in Section 4.3.6. Visual information from the discrete wavelet transform and the Fourier transform provides im- portant high-frequency and spectral details. Using these as additional source modalities can enhance adversarial learn- ing to reduce the semantic gap between kinematic and visual representations. This helps constrain the differences between these modalities. While the generator turns distributions [URL 🔗](#page-0)

L𝑓𝑎𝑙

𝑥𝑘

as the target modality, while


Multimodal Graph Representation Learning for Robust Surgical Workflow Recognition with Adversarial Feature Disentanglement

from different modalities into a shared embedding space, a discriminator distinguishes the representations originating from the kinematic or visual modalities. The discriminator is formulated as:

where 𝑗 ∈ {𝑖, 𝑤, 𝑓, 𝑘} represents different modality, and 𝑙 (⋅) denotes the linear function. In our case, the discriminator 𝐷 aims to differentiate the 𝑥𝑘 𝑡 from kinematic modality as false but 𝑥𝑖 𝑡, 𝑥𝑤 𝑡 , and 𝑥𝑓 𝑡 from visual modality as true. The loss function of adversarial learning is defined as:

When the discriminator cannot accurately distinguish kine- matic representations from visual ones, it indicates that the learned distributions are mapped into a modality-invariant embedding space. By aligning the distributions of the kine- matic modality with those of the visual modality, we can maintain enhanced and robust modality representations. Through adversarial learning, our GRAD framework bet- ter implements cross-modal information complementarity. Thus, when visual signals encounter corruption, the model’s sensitivity to random noise is reduced, improving the robust- ness of the surgical workflow recognition process.

## 3.2.3. Contextual Calibrated Decoder

We aim to enhance the robustness of our multimodal graph network against corrupted data by proposing our con- textual calibrated decoder, which consists of heterogeneous embedding fusion and calibrated optimization fusion. Typi- cally, the depth ofdeep neural networks helps improve repre- sentation capabilities but can lead to overfitting [25]. How- ever, GNNs usually have a relatively low capacity because stacking more GNN layers can cause gradient vanishing issues. Previous work [91] has verified that although the ac- curacy of GNNs decreases with more layers, the confidence score first rises and then falls. This suggests that the low confidence of GNNs might be due to their shallow layers. In this case, we explore the heterogeneous embedding fusion setup to integrate the vision-kinematic feature embeddings and the graph output embeddings, and feed the integrated embeddings to the prediction encoder. The vision-kinematic embeddings can be expressed as: [URL 🔗](#page-0)

𝐸𝑣−𝑘 = 𝑙 ( 𝑥𝑖 𝑡 ∥ 𝑥𝑤 𝑡 ∥ 𝑥𝑓 𝑡 ∥ 𝑥𝑘 𝑡 )

The graph output embeddings can be expressed as:

𝐸𝑔 = 𝑙 [ 𝐺𝐴𝑇 ( 𝑥𝑖 𝑡, 𝑥𝑤 𝑡 , 𝑥𝑓 𝑡 , 𝑥𝑘 𝑡 )]

where 𝑙 (⋅) represents the linear function. Then, the predic- tion encoder input is formulated as:

in which the

𝛼 and 𝛽 are the combination ratios of the

vision-kinematic embeddings and the graph embeddings. The two coefficients are empirically verified through the experiments in Section 4.3.5. By combining and interact- ing vision-kinematic features with GNNs embeddings, we can achieve better feature interaction and complementarity, leading to improved prediction results. In the following experiment section, we perform grid-based ablation studies on the parameters 𝛼 and 𝛽. Our prediction encoder comprises a fully connected layer, and the classification nodes corre- spond to the number of classes. [URL 🔗](#page-0)

As for the optimization function, the typical cross- entropy loss will be applied to the decoder to optimize the network. However, we aim to achieve calibration by ampli- fying the output logits during the training of graph networks. For a model , we set {𝑓, 𝑦} are the pair of the input feature and the ground truth label, and ̂𝑝𝑦 =  (𝑓, 𝑦) is the output probability that  predicts a label 𝑦 for an input feature 𝑓. The predicted confidence is ̂𝑝 = max ̂𝑝𝑦, and the label that  predicts is ̂𝑦 = argmax ̂𝑝𝑦. The GNN is considered perfectly calibrated when the predicted probability, ̂𝑝, aligns exactly with 𝑝, the true probability of making a correct prediction for node 𝑖.

The conventional cross-entropy loss can be expressed as:

where 𝑝 is the target distribution and ̂𝑝 is predicted distribu- tion. The relationship between the cross-entropy loss and the KL divergence can be represented as:

= − ∑𝐾 𝑖=1 𝑝𝑖 log 𝑝𝑖 is a constant, representing

In which(𝑃)

the entropy for a given real distribution. We aim to amplify

the logits to increase the model’s confidence during the computation of max. Therefore, we add a minimal-entropy regularization term to the implicit minimization performed by cross-entropy loss. We intend to incorporate a regulariza- tion term into the cross-entropy loss such that it minimizes when the confidence equals the true probability. Assume the regularization term is 𝑃( ̂𝑃), meaning the target probabil- ity distribution multiplied by the entropy of the predicted distribution. If the predicted distribution aligns with the target distribution, the term will lead to lower entropy and a smaller value for the given expression. By converging this regularization term, the model can achieve better calibration, resulting in stronger robustness. With the new regularization


*Figure 3: Visualization of GRAD compared to MRG-Net (Vis+Kin), Trans-SVNet (Vis), and RL-TCN (Kin) model on MISAW dataset. The different color bands represent each category, which are needle holding, suture making, suture handling, 1◦ knot, 2◦ knot, and 3◦ knot (with order).*

term, the new loss function can be formulated as:

𝜆 is set to 0.02, which is proven to be the most effective parameter by our experiments. Therefore, our final loss function can be expressed as:

𝛾 and 𝛿 are the loss ratios, which are also empirically confirmed through the experiments in Section 4.3.7. [URL 🔗](#page-0)

## 4. Experiments

## 4.1. Datasets

Extensive experiments are carried out using two publicly available datasets for surgical workflow recognition: the MIcro Surgical Anastomose Workflow (MISAW) dataset and the CUHK-MRG dataset, which is a multimodal dataset created on the daVinci ResearchKit platform at The Chinese University of Hong Kong (CUHK), as described in the paper MRG-Net [59]. These two datasets originate from two different types of robot-assisted surgeries, and include visual data, kinematic data, and workflow recognition annotations. Prior works have validated their proposed methods on these datasets [59, 98]. [URL 🔗](#page-0)

MISAW Dataset [33]: The MISAW dataset consists of 27 recordings of micro-surgical anastomosis procedures conducted on synthetic blood vessels, featuring the work of three surgeons and three engineering students. The kine- matic data and video footage were recorded concurrently at [URL 🔗](#page-0)

30 Hz via a high-resolution stereo microscope, which has a resolution of 960 by 540 pixels, along with a master-slave robotic system. Each individual frame is tagged with six

categories of surgical gestures, as illustrated in Figure

3. The [URL 🔗](#page-0)

dataset was divided into a training set with 17 cases and a testing set with 10 cases.

CUHK-MRG Dataset [59]: This dataset consists of 24 [URL 🔗](#page-0)

sequences and the five different gesture steps, which can be seen from Figure 4. The kinematic data encompasses the position and orientation of the end-effector and the gripper’s opening angle. At the same time, video recordings are made at a frequency of 30 Hz. The experimental setting adopts 3-fold cross-validation (8 sequences for training and 4 for testing) following [59]. [URL 🔗](#page-0)

## 4.2. Implementation Details

The end-to-end framework comprises separate visual and kinematic feature extractors, multimodal graph repre- sentation learning, adversarial discriminators, and the work- flow prediction decoder. Initially, the video frames are re- sized to a resolution of 320×256 pixels, with a random crop to 224 by 224 pixels applied to minimize training load and mitigate the risk of overfitting. The video sequence infor- mation is then trained through a pre-trained ResNet18 [27]. After that, the spatial-CNN features are generated from the backbone [44] to make the training procedure more efficient. The graph attention layers feature hidden and output states that are 64 dimensions in size, with a dropout rate of 0.5 implemented. The discriminators are composed of three fully connected layers with dimensions {64, 64}, {64, 16}, {16, 1}. The workflow prediction decoder is also a fully connected layer, and its output dimension corresponds to the number of categories in the dataset. The network is opti- mized with the Adam optimizer, a learning rate of 1×10−4, and a batch size of 64. [URL 🔗](#page-0)


Multimodal Graph Representation Learning for Robust Surgical Workflow Recognition with Adversarial Feature Disentanglement

*Table 1 Comparison of our proposed method against the state-of-the-art single-modality and multi-modality solutions on the MISAW dataset. ∗ indicates that the model shall shuffle the data during training, making it infeasible to calculate the edit score. † means results are extracted from [33]. [URL 🔗](#page-0)*

| Methods | Vis Kin | Acc | Edit OP OR OF1 CP | CR CF1 |
| --- | --- | --- | --- | --- |
| LSTM [29] | % " |   |   | 42.65 42.14 42.39 44.57 43.59 36.78 42.78 38.87 |
| GRU [14] | % " |   |   | 61.63 59.62 72.01 68.87 70.40 38.47 41.79 40.06 |
| TCN [47] | % " |   |   | 60.54 44.53 60.54 60.52 60.52 61.23 57.13 52.51 |
| RL-TCN [53] | % " |   |   | 67.72 64.36 68.97 72.42 70.65 61.97 69.73 65.63 |
| GRU [14] | " % |   |   | 71.12 70.32 68.05 73.87 70.84 46.20 48.50 47.32 |
| TCN [47] | " % |   |   | 68.16 77.07 68.16 69.02 68.59 56.64 60.61 59.38 |
| RL-TCN [53] | " % |   |   | 68.63 79.02 66.30 67.70 66.99 59.15 63.19 61.10 |
| TeCNO [16] | " % |   |   | 76.64 77.26 73.73 77.68 75.65 61.50 64.00 62.73 |
| TMRNet [38] | " % |   |   | 74.44 79.10 74.02 74.71 74.36 63.12 60.45 61.76 |
| Trans-SVNet [39] | " % |   |   | 77.34 78.60 74.28 76.91 75.58 61.55 62.15 61.85 |
| AI-Endo [10] | " % |   |   | 70.35 76.09 74.28 75.34 74.81 67.73 70.87 69.27 |
| Surgformer [100] | " % | 56.25 | ∗ | 40.96 40.48 40.73 36.51 47.01 41.10 |
| Pitfall-BN [75] | " % | 71.07 | ∗ | 71.13 71.06 71.09 71.49 63.89 67.48 |
| LC-SC-CRF [46] | " " |   |   | 78.62 87.74 78.76 78.25 78.51 77.48 78.86 78.14 |
| MRG-Net† [59] | " " | 83.75 | 86.74 - - - - | 81.34 80.97 |
| GRAD (Ours) | " " | 86.87 97.50 86.83 88.23 87.52 | 83.07 | 91.66 87.16 |

*Table 2 Comparison of our proposed method against the state-of-the-art single-modality and multi-modality solutions on the CUHK-MRG*

*dataset. score.*

*∗*

*indicates that the model shall shuffle the data during the training process, making it unfeasible to calculate the edit*

| Methods | Vis Kin | Acc | Edit OP OR OF1 CP | CR CF1 |
| --- | --- | --- | --- | --- |
| LSTM [29] | % " | 81.59 | 100.00 | 86.41 85.44 85.90 86.85 88.28 87.54 |
| GRU [14] | % " | 86.27 | 100.00 | 90.11 89.45 89.77 86.72 88.21 87.45 |
| TCN [47] | % " | 91.29 | 100.00 | 92.46 91.41 91.93 89.10 89.23 88.82 |
| RL-TCN [53] | % " | 89.35 | 97.62 | 91.44 91.93 91.67 88.03 89.29 88.64 |
| GRU [14] | " % | 86.80 | 100.00 | 89.79 88.79 89.27 86.49 87.73 87.08 |
| TCN [47] | " % | 90.40 | 97.22 | 92.58 91.85 92.21 88.85 88.68 88.73 |
| RL-TCN [53] | " % | 89.56 | 100.00 | 91.96 92.24 92.10 88.31 89.12 88.70 |
| TeCNO [16] | " % | 91.00 | 91.31 | 90.55 91.37 90.96 86.13 85.82 85.96 |
| TMRNet [38] | " % | 90.74 | 89.29 | 87.50 87.56 87.53 84.83 75.72 79.97 |
| Trans-SVNet [39] | " % | 89.69 | 90.02 | 89.47 89.24 89.35 84.72 84.68 84.70 |
| AI-Endo [10] | " % | 89.96 | 88.18 | 89.19 86.09 87.61 75.38 74.21 74.79 |
| Surgformer [100] | " % | 90.25 | ∗ | 92.18 89.80 90.51 84.83 87.72 86.25 |
| Pitfall-BN [75] | " % | 78.25 | ∗ | 77.71 76.88 77.29 78.66 77.68 78.11 |
| LC-SC-CRF [46] | " " | 89.97 | 92.91 | 89.85 90.03 89.94 88.89 89.50 89.12 |
| MRG-Net [59] | " " | 89.86 | 100.00 | 92.80 92.24 92.53 88.88 89.41 89.12 |
| GRAD (Ours) | " " |   |   | 92.38 100.00 92.83 93.20 93.01 91.40 92.60 92.00 |

Table 3 Ablation study of the proposed modules on the MISAW Dataset. ‘CAL’ denotes the calibration module, ‘VRD’ denotes visual

representation disentanglement, and ‘VKA’ denotes vision-kinematic adversarial training. We (i) exclude the calibration module, (ii) remove the visual feature disentanglement, and (iii) remove the vision-kinematic adversarial training.

| CAL | VRD | VKA | Acc | Edit OP |   |   | OR OF1 | CP | CR | CF1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| % | % | % | 83.75 | 90.29 | 84.75 | 84.56 | 84.66 | 81.00 | 84.75 | 82.83 |
| ! | % | % | 82.12 | 94.58 | 84.85 | 86.64 | 85.74 | 83.83 | 89.39 | 86.52 |
| % | ! | % | 84.02 | 97.29 | 85.72 | 87.72 | 86.71 | 84.53 | 87.60 | 86.04 |
| % | % | ! | 83.01 | 91.39 | 83.93 | 86.77 | 85.33 | 82.59 | 85.75 | 84.14 |
| ! | ! | % | 82.71 | 96.25 | 85.83 | 87.90 | 86.85 | 84.66 | 89.92 | 87.21 |
| % | ! | ! | 84.42 | 97.50 | 86.20 | 87.89 | 87.04 | 82.67 | 86.83 | 84.67 |
| ! | % | ! | 84.20 | 97.50 | 85.56 | 87.90 | 86.71 | 81.77 | 90.80 | 86.05 |
| ! | ! | ! | 86.87 | 97.50 | 86.83 | 88.23 | 87.52 | 83.07 | 91.66 | 87.16 |


*Figure 4: Figure of CUHK-MRG Dataset illustrates five different gesture steps, which are idle (no action performed), reach for peg (with left hand), lift peg (with left hand), exchange (transfer the peg to right hand), place peg (with right hand).*

*Table 4 Ablation study of the joint modeling of the left and right kinematic data on the MISAW dataset. The first row indicates that the method shall learn the kinematic features of the left and right hands separately and model them as individual graph nodes. The second row indicates that the method shall learn the kinematic features of both hands jointly and model them as a combined graph node.*

| Joint Modeling of | Acc | Edit | OP | OR | OF1 | CP | CR | CF1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Left-Right Kinematic |   |   |   |   |   |   |   |   |
| % | 83.43 | 97.50 | 86.50 | 85.79 | 86.14 | 75.30 | 80.55 | 77.85 |
| ! | 86.87 | 97.50 | 86.83 | 88.23 | 87.52 | 83.07 | 91.66 | 87.16 |

The employed evaluation metrics on those two datasets include (i) Accuracy (%) in the frame-wise level, referring to the calculation of the proportion of frames that are correctly identified; (ii) Edit Score [45] (ranging from 0 to 100, with higher values indicating better performance), which is intended to evaluate effectiveness in video segmentation, specifically focusing on maintaining temporal smoothness. For the ground truth segments 𝑆 = 𝑆1, 𝑆2,…, 𝑆𝑥 and the predicted segments 𝑆′ = 𝑆′ 1, 𝑆′ 2,…, 𝑆′ 𝑦, the edit number 𝑁𝑒𝑑𝑖𝑡 is normalized by the maximum value of 𝑥 and 𝑦. Then, the Edit Score is computed as Edit Score = 100×(1−𝑁𝑒𝑑𝑖𝑡); (iii) Overall Precision (OP), is calculated as the proportion of true positives (correctly identified positive instances) to the total of true positives and false positives (incorrectly identi- fied positive instances); (iv) Overall Recall (OR), represents the proportion of true positives compared to the total number of true positives and false negatives (the positive instances that were misclassified as negatives); (v) Overall F1 Scores (OF1), the harmonic mean of precision and recall; (vi) average per-Class Precision (CP), is the average of precision values for each class, vii) average per-Class Recall (CR), is the average of recall values for each class; (viii) average per- Class F1 Scores (CF1), is the average of F1 scores for each class. Our method is compared against the state-of-the-art kinematics-based, vision-based, and multimodal solutions. [URL 🔗](#page-0)

## 4.3. Experimental Results

We compare our proposed GRAD framework against various state-of-the-art surgical workflow recognition meth- ods, including (i) kinematic-based methods: LSTM [29], GRU [14], TCN [47], RL-TCN [53]; (ii) vision-based meth- ods: GRU [14], TCN [47], RL-TCN [53], TeCNO [16], TM- RNet [38], Trans-SVNet [39], AI-Endo [10], Surgformer [100], Pitfall-BN [75]; (iii) multimodal methods: LC-SC-CRF [46], MRG-Net [59]. [URL 🔗](#page-0)

For the MSIAW dataset, based on the experimental results in Table 1, the model GRU using visual data presents better results on most indicators compared to the same model using motion data where the accuracy is increased by 9.49%. The same phenomenon can be observed in TCN and RL- TCN models. Moreover, the highest score on accuracy with single-modality is 77.34% achieved by the Trans-SVNet using visual data. This accuracy surpasses the highest attain- able accuracy using motion data by 9.62%. It demonstrates that the utilization of visual data achieves superior results compared to the model trained on motion data. This signifies that video encompasses more information than the kinematic data on this dataset and can facilitate the model’s training process and enhance accuracy. By comparing the perfor- mance of models using multimodal data and unimodal data, it can be observed that using multimodal data can increase the performance on most metrics by at least 2% compared to unimodal data. Moreover, as can be seen, the LC-SC-CRF [URL 🔗](#page-0)

## 4.3.1. Comparison with the State-of-the-Art


*Table 5 Ablation study for the effectiveness of the wavelet and Fourier domain in the graph disentanglement on the MISAW dataset.*

| Wavelet | Fourier | Acc | Edit OP |   |   | OR OF1 | CP | CR | CF1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| % | % | 84.20 | 97.50 | 85.56 | 87.90 | 86.71 | 81.77 | 90.80 | 86.05 |
| % | ! | 84.12 | 97.50 | 83.84 | 85.32 | 84.57 | 80.01 | 87.73 | 83.69 |
| ! | % | 84.36 | 95.00 | 86.54 | 87.10 | 86.82 | 82.96 | 87.85 | 85.33 |
| ! | ! | 86.87 | 97.50 | 86.83 | 88.23 | 87.52 | 83.07 | 91.66 | 87.16 |

*Table 6 Ablation study for the effectiveness of different graph networks on the MISAW dataset. ‘GCN’ represents the graph convolutional networks. ‘RGCN’ denotes the relational graph convolutional network. ‘GAT’ is the graph attention network.*

| Graph Networks | Acc | Edit OP |   |   | OR OF1 | CP | CR | CF1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GCN [42] | 84.79 | 94.29 | 84.56 | 86.06 | 85.30 | 85.02 | 89.86 | 85.67 |
| RGCN [76] | 83.42 | 94.58 | 86.91 | 87.72 | 87.31 | 84.47 | 88.18 | 84.63 |
| GAT [83] | 86.87 | 97.50 | 86.83 | 88.23 | 87.52 | 83.07 | 91.66 | 87.16 |

model can improve the edit score by 8.72% which signifies that with multimodal data, the model can be trained more thoroughly, leading to a much better result. Therefore, it can be inferred that multimodal data has the ability to integrate information from two unimodal data sources and offer more comprehensive knowledge for phrase recognition.

Furthermore, on the MISAWdataset, our model achieves 86.87% of accuracy and outperforms the second-highest score by 3.12%. Our model also presents 97.50% on the edit score, suggesting our superiority in learning the sequential order. Generally, our model yields 86.83% on the overall pre- cision score, 88.23% on the overall recall score, and 87.52% on the overall F1 score, which increases the second-highest score by at least 8% and presents the highest score on average per-class recall and F1 scores. However, the performance at the average per-class precision metric is lower than MRG- Net, showcasing the limitation of our proposed method. Even though our model learns the ordering of actions well, it gets confused in discerning the beginning and end of certain phrases. Thus, it leads to lower precision scores on certain categories and decreases the average per-class scores. Nevertheless, our method achieves the best performance on most metrics. Overall, from the analysis of the comparison results, it can be observed that our model showcases superior performance against the existing state-of-the-art methods. Besides, Figure 3 presents the visualization and qualitative comparison of the MISAW dataset. [URL 🔗](#page-0)

As presented in Table 2 and Figure 4, on the CUHK- MRG dataset, our model exhibits the best performance in all metrics except for the edit score, and increases the av- erage pre-class precision and recall metrics by around 2% compared to the second-highest scores. When comparing with single-modal methods, we find that, although using only single-modal data might achieve good performance in terms of accuracy (e.g., TCN, TeCNO, TMRNet), it cannot perform well across all metrics. This is because a single modality does contain discriminative information for work- flow recognition. However, when both kinematic and vision modalities are used, the model gains more complementary [URL 🔗](#page-0)

information. Only by using multiple modalities together can

we significantly enhance overall recognition performance. When using two modalities, Table 2 presents that all metrics ofseveral multimodal models are at high levels. Compared to single-modality methods, the fusion of multimodal data can explore the information in the dataset more deeply and help the model make accurate recognition. Our GRAD frame- work achieves the best overall accuracy of 92.83%, which is 6.42% higher than the LSTM model using kinematics data. [URL 🔗](#page-0)

Besides, we note that some of the edit scores reach the performance of 100%, and the reason may be that the CUHK-MRG dataset is a small dataset with 12 sequences being collected, and every site follows the same sequential operation. The edit score assesses the model’s ability to ac- curately predict the sequence of action segments, regardless of minor temporal variations. Therefore, when the model is small and trained for an adequate number of epochs, it becomes straightforward for the model to memorize this order. There is a possibility that the model is not aware of the sequential order of the two phrases, but rather sub- consciously gives the specific sequence. Consequently, a model’s edit score can achieve 100% on a small dataset.

## 4.3.2. Ablation Study on the Proposed Modules

Table 3 presents the ablation study of the proposed modules. We (i) exclude the calibration module, (ii) remove the visual feature disentanglement, and (iii) remove the vision-kinematic adversarial training. Firstly, we can see that for the baseline in the first row, introducing any new technique results in a performance boost, improving most of the metrics. Besides, when the calibration module is introduced in our model, the performance on average per- class precision, recall, and F1 scores always show significant growth, indicating that the model benefits from the confi- dence boost provided by calibration, making the network more reliable and enhancing the model’s overall perfor- mance. Additionally, adding visual feature disentanglement to different models consistently improves performance, with [URL 🔗](#page-0)


*Table 7 Ablation study of the combination ratio of the vision-kinematic embeddings and graph embeddings on the MISAW dataset, e.g., the 𝛼 and 𝛽 in the Equation (12). [URL 🔗](#page-0)*

| V-K (𝛼) | Graph (𝛽) | Acc | Edit OP |   |   | OR OF1 | CP | CR | CF1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0 | 1.0 | 85.16 | 97.14 | 83.81 | 84.35 | 84.08 | 79.40 | 85.52 | 82.35 |
| 0.1 | 0.9 | 85.53 | 97.50 | 82.08 | 84.19 | 83.12 | 78.51 | 87.80 | 82.90 |
| 0.2 | 0.8 | 85.38 | 97.50 | 83.23 | 84.84 | 84.03 | 79.71 | 89.22 | 84.20 |
| 0.3 | 0.7 | 86.87 | 97.50 | 86.83 | 88.23 | 87.52 | 83.07 | 91.66 | 87.16 |
| 0.4 | 0.6 | 84.38 | 95.00 | 81.38 | 83.87 | 82.60 | 76.77 | 84.86 | 80.61 |
| 0.5 | 0.5 | 83.87 | 95.00 | 83.46 | 84.68 | 84.07 | 78.44 | 85.73 | 81.93 |
| 0.6 | 0.4 | 83.89 | 97.50 | 83.33 | 84.68 | 84.00 | 80.20 | 89.88 | 84.76 |
| 0.7 | 0.3 | 86.92 | 97.50 | 86.53 | 88.06 | 87.29 | 82.76 | 91.51 | 86.92 |
| 0.8 | 0.2 | 85.79 | 97.50 | 81.86 | 83.71 | 82.78 | 77.34 | 84.69 | 80.85 |
| 0.9 | 0.1 | 85.42 | 97.50 | 83.31 | 85.32 | 84.30 | 79.43 | 88.04 | 83.51 |
| 1.0 | 0.0 | 83.81 | 94.29 | 85.49 | 88.39 | 86.92 | 81.36 | 89.98 | 85.46 |

*Table 8 Ablation study of the source and target modality in adversarial training on the MISAW dataset. ‘K’ denotes kinematic modality. ‘V’, ‘W’, and ‘F’ denote the 𝑥𝑖 𝑡, 𝑥𝑤 𝑡 , and 𝑥𝑓 𝑡 from the visual modality, respectively.*

| Source Modality | Target Modality | Acc | Edit OP |   |   | OR OF1 | CP | CR | CF1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| K | V | 82.97 | 96.25 | 85.83 | 86.23 | 86.03 | 83.48 | 89.15 | 83.78 |
| K | V+W+F | 84.07 | 94.14 | 84.08 | 85.43 | 84.75 | 82.48 | 87.92 | 82.72 |
| V | K | 82.68 | 86.17 | 86.03 | 88.39 | 87.18 | 85.00 | 89.46 | 85.93 |
| V+W+F | K | 86.87 | 97.50 | 86.83 | 88.23 | 87.52 | 83.07 | 91.66 | 87.16 |

stable advancements across all metrics. Therefore, the re- sults showcase that using feature disentanglement effectively captures extensive information from the video, leading to a better understanding of the workflow process and improved accuracy. Furthermore, when vision-kinematic adversarial training is added with baseline and visual feature disentan- glement, most metrics show a slight increase. This suggests that by projecting vision and kinematic features into joint embedding space, adversarial training can narrow down the modality gap between two inputs and eventually boost per- formance. Overall, Table 3 demonstrates the effectiveness of our proposed submodules, achieving the best performance only when combined. Additionally, Table 4 verifies whether we should model the kinematics of the left and right hands together or separately. The results showcase that when we model the left and right kinematic information together and construct it as a joint graph node, the model shall achieve the best performance. [URL 🔗](#page-0)

## 4.3.3. Ablation Study on Visual Disentanglement

To understand how visual feature disentanglement im- proves the model, we conduct an ablation analysis on dif- ferent feature disentanglement domains. The experimental results are shown in Table 5. The results indicate that in- troducing the wavelet transform alone significantly boosts performance, while using the Fourier amplitude spectrum alone does not improve it.Wavelet transformbetter identifies multi-scale local features in visual signals and better under- stands edge textures, enhancing the model’s ability to rec- ognize details. In contrast, the Fourier amplitude spectrum emphasizes global structural information in visual signals. Using Fourier alone may lead to some performance loss, [URL 🔗](#page-0)

but combining the local multi-scale understanding from the wavelet domain with the global structure understanding from Fourier provides the model with more diverse information. This complementary use results in notable performance improvement. In summary, the spatial, wavelet, and Fourier domainswe use are essential for the model’s effectiveness, as they contribute to information extraction, feature utilization, and an overall understanding of our model’s workflow.

## 4.3.4. Ablation Study on the Graph Network

To justify the superiority of using GAT in our graph learning part, we perform ablation experiments over differ- ent graph networks on the MISAW dataset, with GCN and RGCN as comparison networks. In GCN, the input adjacent matrix is given by relating kinematic, spatial, wavelet, and Fourier modalities together as one initial setting for graph representation learning. RGCN, on the other hand, can de- fine more adjacent matrices which will broaden the limited correlation within GCN. In this case, we can construct four different relationships based on our features. For example, the kinematic features are influenced by spatial, wavelet, and Fourier features. The setting is extended to the other three features in the same manner. As for GAT, the utilization of the attention mechanism enables the network to assign vary- ing weights to all related nodes according to their respective levels of influence. Table 6 shows that by assigning varying levels of importance to nodes dynamically, GAT can learn the most valuable feature representations. However, when a fixed relationship is assigned to the graph network, certain inherent weight relationships may remain unseen, leading to suboptimal performance of GCN and RGCN. [URL 🔗](#page-0)


*Table 9 Ablation study of the loss function ratio on the MISAW dataset, e.g., the 𝛾 and 𝛿 in the Equation (16). [URL 🔗](#page-0)*

| 𝐶𝐶𝐸 (𝛾) | 𝐴𝐿 (𝛿) | Acc | Edit OP |   |   | OR OF1 | CP | CR | CF1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 | 85.41 | 97.50 | 83.02 | 85.16 | 84.08 | 79.61 | 89.30 | 84.17 |
| 0.95 | 0.05 | 83.62 | 94.17 | 85.96 | 87.90 | 86.92 | 82.13 | 91.00 | 86.34 |
| 0.9 | 0.1 | 86.87 | 97.50 | 86.83 | 88.23 | 87.52 | 83.07 | 91.66 | 87.16 |
| 0.85 | 0.15 | 84.68 | 97.50 | 83.07 | 83.87 | 83.47 | 79.31 | 86.83 | 82.90 |
| 0.8 | 0.2 | 83.93 | 97.50 | 83.30 | 84.50 | 83.90 | 81.17 | 86.33 | 83.67 |

*Table 10 Ablation study of the 𝜆 in the Equation (15) of calibration loss function on the MISAW dataset. [URL 🔗](#page-0)*

| 𝜆 | Acc | Edit | OP | OR | OF1 | CP | CR | CF1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.01 | 84.12 | 97.14 | 80.79 | 82.74 | 81.75 | 81.42 | 87.76 | 84.47 |
| 0.02 | 86.87 | 97.50 | 86.83 | 88.23 | 87.52 | 83.07 | 91.66 | 87.16 |
| 0.03 | 84.97 | 95.56 | 83.25 | 84.19 | 83.72 | 83.37 | 89.82 | 86.47 |
| 0.04 | 84.93 | 97.14 | 83.65 | 85.81 | 84.71 | 83.55 | 89.32 | 86.34 |
| 0.05 | 86.10 | 97.50 | 83.31 | 85.32 | 84.44 | 83.31 | 89.02 | 86.07 |

## 4.3.5. Ablation Study on the Ratio ofVision-Kinematic Feature & Graph Embeddings

For the weighted fusion of vision-kinematic feature em- beddings and graph output embeddings in Equation (12), we further investigate how the parameters 𝛼 and 𝛽 in Equa- tion (12) achieve a perfect balance. Specifically, we inte- grate the feature embeddings from visual and kinematic data before the graph network with the embedding output by the graph network based on Equation (12). The final input to the workflow prediction decoder is the weighted sum of these two different outputs. Our objective is to seek the best performance of our model by finding the optimal balance between vision-kinematic feature output and graph output through ablation experiments. Through a thorough comparative analysis of various ratios in Table 7, we find that the ratio of 0.7 for 𝛼 and 0.3 for 𝛽 results in the highest accuracy, but it slightly underperforms in all other evaluation metrics compared to the ratio of 0.3 for 𝛼 and 0.7 for 𝛽. Dif- ferent ratio configurations exhibit respective advantages on different indicators, demonstrating that both the feature and graph output embeddings contribute specifically to model performance. Finally, 𝛼 = 0.3 and 𝛽 = 0.7 achieve the highest performance across multiple metrics; therefore, we use this configuration as the final parameters for our model. [URL 🔗](#page-0)

## 4.3.6. Ablation Study on the Source and Target Modalities in Adversarial Training

To investigate how treating different inputs as source and target modalities affects the model’s performance, we con- duct related experiments in Table 8. By designating various modalities as source and target, adversarial learning enables the projection of distinct modalities into a shared embedding space, aligning target distributions with source distributions within a modality-invariant space. The results in Table 8 indicate that when kinematic features are used as the source modality and visual representations as the target modality, the model exhibits inferior performance across most met- rics compared to other configurations. This is likely due to [URL 🔗](#page-0)

the inherent difficulty of mapping visual distributions onto kinematic distributions, as visual representations contain more complex information. Additionally, the wavelet and Fourier representations, which share similarities with spa- tial representations, allow the model to process them more effectively in combination. Overall, the results demonstrate that treating visual representations from three domains as the source modality and kinematic data as the target modality enables the model to achieve satisfactory performance.

## 4.3.7. Ablation Study of the Loss Function

Experiments in Table 9 analyze the ratio between ad- versarial loss 𝐴𝐿 and calibrated classification loss 𝐶𝐶𝐸 to find the ideal balance. As shown in Table 9, the model performs poorly when the adversarial loss proportion is too high. This suggests that overemphasizing the reduction of the modality gap may cause the model to neglect the workflow recognition task, leading to poor performance. However, when the adversarial loss ratio is too low or absent, the model also performs poorly on most metrics, indicating the need for adversarial loss to help approximate and integrate both modalities. Ultimately, by analyzing the results, we find the best performance with a ratio of 0.9 for 𝐶𝐶𝐸 and 0.1 for adversarial loss 𝐴𝐿. This demonstrates the importance of reducing the gap between different modal- ities and highlights the contribution adversarial loss makes to our model. Additionally, we explored the setting of the regularization coefficient 𝜆 in Equation (15). Table 10 shows that we achieve the best results when 𝜆 = 0.02. [URL 🔗](#page-0)

## 4.3.8. Ablation Study on Different Fusion Strategies

In Table 11, we compare our GRAD method with other multimodal fusion techniques to demonstrate its effective- ness in integrating visual and kinematic data for robotics. We evaluate different fusion methods while keeping other modules consistent. The methods include Matrix Addi- tion, Concatenation, Co-Attention (V2K: vision as attention [URL 🔗](#page-0)


*Table 11 Ablation study on different fusion strategies compared with our graph learning strategy. All experiments employ the same backbone, feature disentanglement, adversarial learning, and loss function. "Attn": Attention.*

| Methods | Acc | Edit OP |   |   | OR OF1 | CP | CR | CF1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Add | 84.60 | 94.55 | 81.09 | 83.71 | 82.38 | 75.88 | 83.23 | 79.39 |
| Concat | 83.73 | 90.83 | 81.07 | 82.90 | 81.98 | 76.63 | 83.60 | 79.96 |
| Cross Attn (V2K) | 70.61 | 81.81 | 83.62 | 87.26 | 85.40 | 80.64 | 82.23 | 81.43 |
| Cross Attn (K2V) | 71.99 | 81.57 | 88.68 | 84.68 | 86.63 | 85.37 | 88.05 | 86.69 |
| Self Attn | 76.44 | 86.33 | 85.93 | 82.74 | 84.31 | 82.24 | 86.23 | 84.19 |
| Gated Fusion | 84.24 | 95.00 | 77.54 | 81.29 | 79.37 | 66.33 | 72.91 | 69.46 |
| BAN | 74.66 | 89.76 | 86.54 | 84.03 | 85.27 | 83.53 | 88.76 | 86.07 |
| TDA | 74.78 | 88.17 | 88.43 | 86.29 | 87.35 | 84.46 | 88.80 | 86.57 |
| AFF | 85.82 | 95.00 | 86.42 | 87.26 | 86.84 | 74.56 | 82.53 | 78.34 |
| iAFF | 83.05 | 96.25 | 85.65 | 87.58 | 86.60 | 80.98 | 87.46 | 84.10 |
| GRAD (Ours) | 86.87 | 97.50 | 86.83 | 88.23 | 87.52 | 83.07 | 91.66 | 87.16 |

80.00

75.00

70.00

65.00

60.00

*Figure 5: An example of corrupted data visualization for robustness assessment, featuring four different types ranging from noise to digital damage.*

8

8

7

Accuracy

7

6

6

*Figure 6: Robustness experiments on the MISAW dataset under 18 different corruptions.*


*Table 12 Robustness comparison experiments of our GRAD with MRG-Net across 18 different types of pollution and 5 levels of severity on the MISAW dataset.*

| Severity | Level 1 |   | Level 2 |   | Level 3 |   | Level 4 |   | Level 5 |   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Model |   |   |   |   |   |   |   |   |   | MRG [59] GRAD MRG [59] GRAD MRG [59] GRAD MRG [59] GRAD MRG [59] GRAD |
| Gauss | 81.90 | 83.03 | 79.35 | 82.58 | 77.87 | 80.65 | 75.15 | 80.19 | 70.93 | 74.96 |
| Shot Noise | 80.73 | 80.24 | 76.80 | 77.44 | 71.45 | 77.19 | 68.17 | 75.76 | 68.86 | 76.15 |
| Impulse | 76.72 | 81.02 | 67.87 | 77.56 | 66.28 | 74.86 | 67.28 | 74.74 | 64.07 | 75.00 |
| Speckle | 79.20 | 83.15 | 75.85 | 81.57 | 73.15 | 81.02 | 70.76 | 79.71 | 67.20 | 75.91 |
| Defocus | 77.19 | 80.71 | 76.27 | 79.09 | 72.35 | 77.77 | 72.17 | 76.56 | 73.06 | 76.42 |
| Motion | 79.34 | 82.35 | 76.70 | 79.95 | 75.66 | 77.41 | 75.36 | 75.53 | 72.93 | 76.06 |
| Blur Glass | 80.52 | 83.25 | 77.68 | 82.55 | 75.04 | 79.49 | 72.48 | 77.26 | 69.87 | 75.42 |
| Gauss | 81.90 | 83.03 | 79.35 | 82.58 | 77.87 | 80.65 | 75.15 | 80.19 | 70.93 | 74.96 |
| Zoom | 74.58 | 81.71 | 71.36 | 80.72 | 64.11 | 79.62 | 62.50 | 78.62 | 59.39 | 76.99 |
| Brightness | 81.76 | 85.30 | 80.97 | 83.17 | 78.73 | 80.54 | 74.84 | 78.13 | 69.70 | 79.44 |
| Occlusion Spatter | 77.95 | 82.31 | 75.29 | 79.01 | 74.71 | 79.14 | 71.18 | 76.88 | 63.05 | 70.83 |
| Smoke | 81.84 | 82.33 | 74.49 | 78.29 | 66.27 | 76.14 | 62.13 | 73.32 | 58.52 | 71.77 |
| Contrast | 80.73 | 84.79 | 78.80 | 83.61 | 76.84 | 81.42 | 72.18 | 77.04 | 67.18 | 76.76 |
| Elastic | 80.68 | 83.63 | 79.75 | 82.82 | 77.79 | 82.02 | 74.44 | 80.53 | 71.68 | 78.56 |
| Gamma Digital | 82.46 | 84.19 | 76.69 | 82.37 | 75.37 | 81.66 | 73.99 | 76.87 | 69.72 | 71.65 |
| Jpeg | 78.84 | 81.38 | 78.03 | 79.28 | 75.09 | 75.58 | 68.68 | 73.23 | 63.15 | 81.90 |
| Pixelate | 81.12 | 83.59 | 80.97 | 83.11 | 79.15 | 82.64 | 75.02 | 81.18 | 73.44 | 79.22 |
| Saturate | 81.04 | 83.42 | 77.29 | 83.04 | 75.85 | 81.21 | 70.86 | 73.27 | 64.74 | 71.30 |
| Mean | 79.95 | 82.77 | 76.67 | 81.00 | 73.81 | 79.40 | 70.60 | 88.15 | 63.12 | 75.92 |

*Figure 7: Robustness experiments on the MISAW dataset at each severity level.*

queries; K2V: kinematics as attention queries) [51], Self- Attention [82], Gated Fusion [6], BAN [40], TDA [2], AFF &iAFF [17]. Specifically, we replace our graph-based fusion with these different methods, while retaining the visual fea- ture disentanglement and calibration components to ensure a fair comparison. The results show that attention-based methods like AFF and iAFF achieve decent performance, but simpler combination methods or gated networks yield better outcomes. This indicates that direct attention-based fusion might not be optimal, and further exploration of the interaction between kinematics and vision is needed. Our graph representation approach achieves the best results. [URL 🔗](#page-0)

## 4.4. Robustness

To thoroughly evaluate the robustness of our proposed GRAD framework, we conduct extensive experiments under

diverse image corruptions using the MISAW dataset, per- forming a comparative analysis with MRG-Net [59]. For a comprehensive experiment, we assess visual data exposed to 18 distinct corruption types, categorized into four main groups: Noise, Blur, Occlusion, and Digital corruptions, [URL 🔗](#page-0)

from a 2D image corruption benchmark [28]. Figure [URL 🔗](#page-0)

[5](#page-0)

provides a visualization of each corruption type. Besides, each corruption type is applied across five severity levels, enabling a thorough observation of model performance un- der escalating corruption intensities.

Figure 7 shows the average results across 18 corruption types at each corruption severity intensity. We can find that the performance of both methods gradually decreases as the severity level of corruption increases. However, our model consistently outperforms the baseline MRG-Net across all severity levels, and exhibits significantly less performance decline. This outcome underscores the superior stability of our model in the presence of perturbations from visual corruption. Through our proposed GRAD framework, the model can better extract detailed information from corrupted images, thereby demonstrating greater robustness. Further- more, we evaluate the average results across different sever- ity levels, which are provided in Figure 6. When the severity intensity reaches the highest level, our model achieves an accuracy exceeding 75% across most categories of image corruption and surpasses 80% in certain categories. In con- trast, MRG-Net performs lower than 75% in all corruption types. This demonstrates our model’s remarkable perfor- mance in handling various forms of image degradation. Detailed results for each type of corruption are shown in Table 12, where the comparative analysis reveals substantial performance for our GRAD model across various corruption scenarios. Our model consistently attains higher accuracy [URL 🔗](#page-0)


than MRG-Net, a testament to the efficacy of incorporat- ing multimodal adversarial training, visual disentanglement, and calibration loss. These methods markedly bolster the robustness of our model, enhancing its resilience to image corruption challenges and leading to more reliable real- world application ability.

## 4.5. Discussion and Limitations

Overall, our proposed GRAD framework consistently achieves the best performance compared to existing state- of-the-art methods, even on corrupted data. This demon- strates that our framework excels in performance and is highly robust to handle data corruption. With our feature disentanglement and effective graph representation strategy, our model can efficiently model representations from dif- ferent modalities and find the optimal multimodal balance. Additionally, with our graph calibration and multimodal adversarial training, our framework can robustly work on low-quality data in different scenarios, demonstrating great potential for real-world applications.

While our proposed model demonstrates improved per- formance on both datasets, limitations still exist. (i) Firstly, the short duration of specific phrases in the dataset leads to difficulties in accurate recognition during training, causing a bias toward more frequent categories and resulting in potential misclassifications. Future work should address this data imbalance to enhance the model’s ability to learn short- duration phrases. (ii) Secondly, training with multimodal data requires significant time and resources, including pre- cise annotation methods for aligning visual signals, kine- matic data, and labels, as well as additional computational costs for training and inference. Evaluating the trade-off between performance gains and resource consumption is es- sential, alongside exploring more efficient strategies at both the data and algorithm levels. (iii) Lastly, all experiments in this study are conducted on benchmark datasets rather than real surgical scenes, due to the lack of a comprehensive multimodal dataset for surgical applications. This limits the evaluation of our proposed framework under real-world clin- ical conditions, where unique challenges such as lighting, occlusions, and sensor variability exist. Future efforts should focus on creating high-quality multimodal datasets tailored to surgical scenarios to validate and improve the model’s robustness and applicability.

## 5. Conclusion

In this paper, we present a robust workflow recognition model, GRAD, offering a comprehensive and robust solu- tion for surgical workflow recognition by effectively inte- grating multiple data modalities. Our framework integrates visual feature disentanglement across different domains with graph-based representation learning, effectively capturing complementary information between visual and kinematic modalities. Furthermore, we propose the Vision-Kinematic Adversarial strategy to align multimodal feature distribu- tions and the contextual calibrated prediction decoder to

enhance robustness and confidence in modality represen- tations, even under data corruption or domain shifts. The proposed GRAD framework outperforms existing single- modality and multi-modality methods in surgical workflow recognition tasks on two public datasets. Extensive ablation experiments further demonstrated the effectiveness of the individual components. The disentanglement of spatial and frequency features in visual data and the temporal modeling of kinematic data significantly enhanced workflow recog- nition accuracy. Moreover, the GRAD model exhibits high robustness against data corruption, showing slower perfor- mance degradation in robustness tests. Overall, our GRAD framework presents strong reliability and adaptability by capturing intricate patterns and relationships inherent in visual and kinematic data, proving its potential for real-world clinical applications.

## References

- [1] Ahmidi, N., Tao, L., Sefati, S., Gao, Y., Lea, C., Haro, B.B., Zap- pella, L., Khudanpur, S., Vidal, R., Hager, G.D., 2017. A dataset and benchmarks for segmentation and recognition of gestures in robotic surgery. IEEE Transactions on Biomedical Engineering 64, 2025– 2041.

- [2] Anderson, P., He, X., Buehler, C., Teney, D., Johnson, M., Gould, S., Zhang, L., 2018. Bottom-up and top-down attention for image captioning and visual question answering, in: Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 6077–6086.

- [3] Ayobi, N., Rodríguez, S., Pérez, A., Hernández, I., Aparicio, N., Dessevres, E., Peña, S., Santander, J., Caicedo, J.I., Fernández, N., et al., 2024. Pixel-wise recognition for holistic surgical scene understanding. arXiv preprint arXiv:2401.11174 .

- [4] Bai, L., Chen, T., Wu, Y., Wang, A., Islam, M., Ren, H., 2023a. Llcaps: Learning to illuminate low-light capsule endoscopy with curved wavelet attention and reverse diffusion, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer. pp. 34–44.

- [5] Bai, L., Islam, M., Ren, H., 2023b. Cat-vil: co-attention gated vision-language embedding for visual question localized-answering in robotic surgery, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer. pp. 397– 407.

- [6] Bai, L., Islam, M., Seenivasan, L., Ren, H., 2023c. Surgical- vqla: Transformer with gated vision-language embedding for visual question localized-answering in robotic surgery, in: 2023 IEEE International Conference on Robotics and Automation (ICRA), pp. 6859–6865. doi:10.1109/ICRA48891.2023.10160403.

- [7] Bai, L., Wang, G., Wang, J., Yang, X., Gao, H., Liang, X., Wang, A., Islam, M., Ren, H., 2024. Ossar: Towards open-set surgi- cal activity recognition in robot-assisted surgery. arXiv preprint arXiv:2402.06985 .

- [8] Berta, E., Bach, F., Jordan, M., 2024. Classifier calibration with roc-regularized isotonic regression, in: International Conference on Artificial Intelligence and Statistics, PMLR. pp. 1972–1980.

- [9] Blundell, C., Cornebise, J., Kavukcuoglu, K., Wierstra, D., 2015. Weight uncertainty in neural network, in: International conference on machine learning, PMLR. pp. 1613–1622.

- [10] Cao, J., Yip, H.C., Chen, Y., Scheppach, M., Luo, X., Yang, H., Cheng, M.K., Long, Y., Jin, Y., Chiu, P.W.Y., et al., 2023. Intelligent surgical workflow recognition for endoscopic submucosal dissection with real-time animal study. Nature Communications 14, 6676.

- [11] Chen, P., Zhao, L., Piao, Y., Ding, H., Cui, X., 2023. Multimodal visual-textual object graph attention network for propaganda detec- tion in memes. Multimedia Tools and Applications , 1–16.


- [12] Chen, T., Lyu, Q., Bai, L., Guo, E., Gao, H., Yang, X., Ren, H., Zhou, L., 2024a. Lightdiff: Surgical endoscopic image low-light en- hancement with t-diffusion, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer. pp. 369–379.

- [13] Chen, T., Yang, S., Wang, J., Bai, L., Ren, H., Zhou, L., 2024b. Surgsora: Decoupled rgbd-flow diffusion model for controllable surgical video generation. arXiv preprint arXiv:2412.14018 .

- [14] Cho, K., Van Merriënboer, B., Bahdanau, D., Bengio, Y., 2014. On the properties of neural machine translation: Encoder-decoder approaches. arXiv preprint arXiv:1409.1259 .

- [15] Cui, B., Islam, M., Bai, L., Ren, H., 2024. Surgical-dino: Adapter learning of foundation model for depth estimation in endoscopic surgery. arXiv preprint arXiv:2401.06013 .

- [16] Czempiel, T., Paschali, M., Keicher, M., Simson, W., Feussner, H., Kim, S.T., Navab, N., 2020. Tecno: Surgical phase recognition with multi-stage temporal convolutional networks, in: Medical Image Computing and Computer Assisted Intervention–MICCAI 2020: 23rd International Conference, Lima, Peru, October 4–8, 2020, Proceedings, Part III 23, Springer. pp. 343–352.

- [17] Dai, Y., Gieseke, F., Oehmcke, S., Wu, Y., Barnard, K., 2021. Attentional feature fusion, in: Proceedings of the IEEE/CVF winter conference on applications of computer vision, pp. 3560–3569.

- [18] Dergachyova, O., Bouget, D., Huaulmé, A., Morandi, X., Jannin, P., 2016. Automatic data-driven real-time segmentation and recogni- tion of surgical workflow. International journal of computer assisted radiology and surgery 11, 1081–1089.

- [19] Dodge, S., Karam, L., 2016. Understanding how image quality af- fects deep neural networks, in: 2016 eighth international conference on quality of multimedia experience (QoMEX), IEEE. pp. 1–6.

- [20] Forestier, G., Petitjean, F., Senin, P., Despinoy, F., Huaulmé, A., Fawaz, H.I., Weber, J., Idoumghar, L., Muller, P.A., Jannin, P., 2018. Surgical motion analysis using discriminative interpretable patterns. Artificial intelligence in medicine 91, 3–11.

- [21] Fortunato, M., Blundell, C., Vinyals, O., 2017. Bayesian recurrent neural networks. arXiv preprint arXiv:1704.02798 .

- [22] Gao, H., Yang, X., Xiao, X., Zhu, X., Zhang, T., Hou, C., Liu, H., Meng, M.Q.H., Sun, L., Zuo, X., et al., 2023. Transendoscopic flexible parallel continuum robotic mechanism for bimanual endo- scopic submucosal dissection. The International Journal ofRobotics Research , 02783649231209338.

- [23] Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., Bengio, Y., 2020. Generative adversarial networks. Communications of the ACM 63, 139–144.

- [24] Graves, A., Graves, A., 2012. Long short-term memory. Supervised sequence labelling with recurrent neural networks , 37–45.

- [25] Guo, C., Pleiss, G., Sun, Y., Weinberger, K.Q., 2017. On calibration of modern neural networks, in: International conference on machine learning, PMLR. pp. 1321–1330.

- [26] Han, J., Davids, J., Ashrafian, H., Darzi, A., Elson, D.S., Sodergren, M., 2022. A systematic review of robotic surgery: From supervised paradigms to fully autonomous robotic approaches. The Interna- tional Journal of Medical Robotics and Computer Assisted Surgery 18, e2358.

- [27] He, K., Zhang, X., Ren, S., Sun, J., 2016. Deep residual learning for image recognition, in: Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770–778.

- [28] Hendrycks, D., Dietterich, T., 2019. Benchmarking neural network robustness to common corruptions and perturbations. arXiv preprint arXiv:1903.12261 .

- [29] Hochreiter, S., Schmidhuber, J., 1997. Long short-term memory. Neural computation 9, 1735–80. doi:10.1162/neco.1997.9.8.1735.

- [30] Hou, W., Lin, C., Yu, L., Qin, J., Yu, R., Wang, L., 2023. Hybrid graph convolutional network with online masked autoencoder for robust multimodal cancer survival prediction. IEEE Transactions on Medical Imaging 42, 2462–2473. doi:10.1109/TMI.2023.3253760.

- [31] Hsu, H.H.H., Shen, Y., Tomani, C., Cremers, D., 2022. What makes graph neural networks miscalibrated? Advances in Neural

- Information Processing Systems 35, 13775–13786.

- [32] Huang, Z., Qiu, Z., Wang, Z., Ponti, E.M., Titov, I., 2024. Post- hoc reward calibration: A case study on length bias. arXiv preprint arXiv:2409.17407 .

- [33] Huaulmé, A., Sarikaya, D., Le Mut, K., Despinoy, F., Long, Y., Dou, Q., Chng, C.B., Lin, W., Kondo, S., Bravo-Sánchez, L., et al., 2021. Micro-surgical anastomose workflow recognition challenge report. Computer Methods and Programs in Biomedicine 212, 106452.

- [34] Islam, M., Seenivasan, L., Ren, H., Glocker, B., 2021. Class- distribution-aware calibration for long-tailed visual recognition. arXiv preprint arXiv:2109.05263 .

- [35] Jang, E., Gu, S., Poole, B., 2016. Categorical reparameterization with gumbel-softmax. arXiv preprint arXiv:1611.01144 .

- [36] Jin, Y., Dou, Q., Chen, H.,Yu, L., Qin, J., Fu, C.W., Heng, P.A., 2017. Sv-rcnet: workflow recognition from surgical videos using recurrent convolutional network. IEEE transactions on medical imaging 37, 1114–1126.

- [37] Jin, Y., Li, H., Dou, Q., Chen, H., Qin, J., Fu, C.W., Heng, P.A., 2020. Multi-task recurrent convolutional network with correlation loss for surgical video analysis. Medical image analysis 59, 101572.

- [38] Jin, Y., Long, Y., Chen, C., Zhao, Z., Dou, Q., Heng, P.A., 2021. Temporal memory relation network for workflow recognition from surgical video. IEEE Transactions on Medical Imaging 40, 1911– 1923.

- [39] Jin, Y., Long, Y., Gao, X., Stoyanov, D., Dou, Q., Heng, P.A., 2022. Trans-svnet: hybrid embedding aggregation transformer for surgical workflow analysis. International Journal of Computer Assisted Radiology and Surgery 17, 2193–2202.

- [40] Kim, J.H., Jun, J., Zhang, B.T., 2018. Bilinear attention networks. Advances in neural information processing systems 31.

- [41] Kim, M., Kwon, J., 2025. Uncertainty calibration with energy based instance-wise scaling in the wild dataset, in: European Conference on Computer Vision, Springer. pp. 232–248.

- [42] Kipf, T.N., Welling, M., 2016. Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907 .

- [43] Lakshminarayanan, B., Pritzel, A., Blundell, C., 2017. Simple and scalable predictive uncertainty estimation using deep ensembles. Advances in neural information processing systems 30.

- [44] Lea, C., Flynn, M.D., Vidal, R., Reiter, A., Hager, G.D., 2017. Tem- poral convolutional networks for action segmentation and detection, in: proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 156–165.

- [45] Lea, C., Reiter, A., Vidal, R., Hager, G.D., 2016a. Segmental spatiotemporal cnns for fine-grained action segmentation, in: Com- puter Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part III 14, Springer. pp. 36–52.

- [46] Lea, C., Vidal, R., Hager, G.D., 2016b. Learning convolutional action primitives for fine-grained action recognition, in: 2016 IEEE international conference on robotics and automation (ICRA), IEEE. pp. 1642–1649.

- [47] Lea, C., Vidal, R., Reiter, A., Hager, G.D., 2016c. Temporal convolutional networks: A unified approach to action segmentation, in: Computer Vision–ECCV 2016 Workshops: Amsterdam, The Netherlands, October 8-10 and 15-16, 2016, Proceedings, Part III 14, Springer. pp. 47–54.

- [48] Li, L., Li, X., Ouyang, B., Mo, H., Ren, H., Yang, S., 2023. Three-dimensional collision avoidance method for robot-assisted minimally invasive surgery. Cyborg and Bionic Systems 4, 0042.

- [49] Li, M., Zhuang, X., Bai, L., Ding, W., 2024a. Multimodal graph learning based on 3d haar semi-tight framelet for student engage- ment prediction. Information Fusion 105, 102224.

- [50] Li, Z., Chen, S., Yang, J., Luo, L., 2024b. Efficiency calibration of implicit regularization in deep networks via self-paced curriculum- driven singular value selection, in: Larson, K. (Ed.), Proceedings of the Thirty-Third International Joint Conference on Artificial In- telligence, IJCAI-24, International Joint Conferences on Artificial Intelligence Organization. pp. 4515–4523.


- [51] Lin, H., Cheng, X., Wu, X., Shen, D., 2022. Cat: Cross attention in vision transformer, in: 2022 IEEE international conference on multimedia and expo (ICME), IEEE. pp. 1–6.

- [52] Lin, T., 2017. Focal loss for dense object detection. arXiv preprint arXiv:1708.02002 .

- [53] Liu, D., Jiang, T., 2018. Deep reinforcement learning for surgical gesture segmentation and classification, in: Medical Image Com- puting and Computer Assisted Intervention–MICCAI 2018: 21st International Conference, Granada, Spain, September 16-20, 2018, Proceedings, Part IV 11, Springer. pp. 247–255.

- [54] Liu, J., Fan, X., Huang, Z.,Wu, G., Liu, R., Zhong,W., Luo, Z., 2022. Target-aware dual adversarial learning and a multi-scenario multi- modality benchmark to fuse infrared and visible for object detection, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 5802–5811.

- [55] Liu, J., Long, Y., Chen, K., Leung, C.H., Wang, Z., Dou, Q., 2023a. Visual-kinematics graph learning for procedure-agnostic instrument tip segmentation in robotic surgeries, in: 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), IEEE. pp. 4633–4639.

- [56] Liu, L., Pan, Y., Li, X., Chen, G., 2024. Uncertainty estimation and quantification for llms: A simple supervised approach. arXiv preprint arXiv:2404.15993 .

- [57] Liu, Y., Boels, M., Garcia-Peraza-Herrera, L.C., Vercauteren, T., Dasgupta, P., Granados, A., Ourselin, S., 2025. Lovit: Long video transformer for surgical phase recognition. Medical Image Analysis 99, 103366.

- [58] Liu, Y., Huo, J., Peng, J., Sparks, R., Dasgupta, P., Granados, A., Ourselin, S., 2023b. Skit: a fast key information video trans- former for online surgical phase recognition, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 21074–21084.

- [59] Long, Y., Wu, J.Y., Lu, B., Jin, Y., Unberath, M., Liu, Y.H., Heng, P.A., Dou, Q., 2021. Relational graph learning on visual and kinematics embeddings for accurate gesture recognition in robotic surgery, in: 2021 IEEE International Conference on Robotics and Automation (ICRA), IEEE. pp. 13346–13353.

- [60] Mai, S., Hu, H., Xing, S., 2020. Modality to modality translation: An adversarial representation learning and graph fusion network for multimodal fusion, in: proceedings of the AAAI conference on artificial intelligence, pp. 164–172.

- [61] Mai, S., Xing, S., He, J., Zeng, Y., Hu, H., 2023. Multimodal graph for unaligned multimodal sequence analysis via graph convolution and graph pooling. ACM TRANSACTIONS ON MULTIMEDIA COMPUTING COMMUNICATIONS AND APPLICATIONS 19. doi:10.1145/3542927.

- [62] Meng, T., Zhang, F., Shou,Y., Shao, H., Ai,W., Li, K., 2024. Masked graph learning with recurrent alignment for multimodal emotion recognition in conversation. IEEE/ACM Transactions on Audio, Speech, and Language Processing 32, 4298–4312.

- [63] Moglia, A., Georgiou, K., Georgiou, E., Satava, R.M., Cuschieri, A., 2021. A systematic review on artificial intelligence in robot-assisted surgery. International Journal of Surgery 95, 106151.

- [64] Naeini, M.P., Cooper, G., Hauskrecht, M., 2015. Obtaining well calibrated probabilities using bayesian binning, in: Proceedings of the AAAI conference on artificial intelligence.

- [65] Nong, L., Peng, J., Zhang, W., Lin, J., Qiu, H., Wang, J., 2022. Adaptive multi-hypergraph convolutional networks for 3d object classification. IEEE Transactions on Multimedia .

- [66] Park, J., Kim, J., Lee, J.S., 2024. Impact of regularization on cal- ibration and robustness: from the representation space perspective. arXiv preprint arXiv:2410.03999 .

- [67] Pei, J., Wang, C., Szarvas, G., 2022. Transformer uncertainty estimation with hierarchical stochastic attention, in: Proceedings of the AAAI Conference on Artificial Intelligence, pp. 11147–11155.

- [68] Pereyra, G., Tucker, G., Chorowski, J., Kaiser, Ł., Hinton, G., 2017. Regularizing neural networks by penalizing confident output distributions. arXiv preprint arXiv:1701.06548 .

- [69] Platt, J., et al., 1999. Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. Ad- vances in large margin classifiers 10, 61–74.

- [70] Popordanoska, T., Tiulpin, A., Blaschko, M.B., 2024. Beyond classification: Definition and density-based estimation of calibration in object detection, in: Proceedings of the IEEE/CVF Winter Con- ference on Applications of Computer Vision, pp. 585–594.

- [71] Psychogyios, D., Colleoni, E., Van Amsterdam, B., Li, C.Y., Huang, S.Y., Li, Y., Jia, F., Zou, B., Wang, G., Liu, Y., et al., 2023. Sar-rarp50: Segmentation of surgical instrumentation and action recognition on robot-assisted radical prostatectomy challenge. arXiv preprint arXiv:2401.00496 .

- [72] Pugin, F., Bucher, P., Morel, P., 2011. History of robotic surgery: from aesop® and zeus® to da vinci®. Journal of visceral surgery 148, e3–e8.

- [73] Qin, Y., Pedram, S.A., Feyzabadi, S., Allan, M., McLeod, A.J., Burdick, J.W., Azizian, M., 2020. Temporal segmentation ofsurgical sub-tasks through deep learning with multiple data sources, in: 2020 IEEE International Conference on Robotics and Automation (ICRA), IEEE. pp. 371–377.

- [74] Qu, G., Xiao, L., Hu, W., Wang, J., Zhang, K., Calhoun, V.D., Wang, Y.P., 2021. Ensemble manifold regularized multi-modal graph convolutional network for cognitive ability prediction. IEEE Transactions on Biomedical Engineering 68, 3564–3573.

- [75] Rivoir, D., Funke, I., Speidel, S., 2024. On the pitfalls of batch normalization for end-to-end video learning: a study on surgical workflow analysis. Medical Image Analysis 94, 103126.

- [76] Schlichtkrull, M., Kipf, T.N., Bloem, P., Van Den Berg, R., Titov, I., Welling, M., 2018. Modeling relational data with graph con- volutional networks, in: The semantic web: 15th international con- ference, ESWC 2018, Heraklion, Crete, Greece, June 3–7, 2018, proceedings 15, Springer. pp. 593–607.

- [77] Shang, Y., Gao, C., Chen, J., Jin, D., Ma, H., Li, Y., 2023. Enhancing adversarial robustness of multi-modal recommendation via modality balancing, in: Proceedings of the 31st ACM International Confer- ence on Multimedia, pp. 6274–6282.

- [78] Strang, G., 1993. Wavelet transforms versus fourier transforms. Bulletin of the American Mathematical Society 28, 288–305.

- [79] Tan, J., Pei, S., Qin, W., Fu, B., Li, X., Huang, L., 2024. Wavelet- based mamba with fourier adjustment for low-light image enhance- ment, in: Proceedings of the Asian Conference on Computer Vision, pp. 3449–3464.

- [80] Teixeira, L., Jalaian, B., Ribeiro, B., 2019. Are graph neural networks miscalibrated? arXiv preprint arXiv:1905.02296 .

- [81] Twinanda, A.P., Shehata, S., Mutter, D., Marescaux, J., De Mathelin, M., Padoy, N., 2016. Endonet: a deep architecture for recognition tasks on laparoscopic videos. IEEE transactions on medical imaging 36, 86–97.

- [82] Vaswani, A., 2017. Attention is all you need. Advances in Neural Information Processing Systems .

- [83] Veličković, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., Bengio, Y., 2017. Graph attention networks. arXiv preprint arXiv:1710.10903 .

- [84] Velickovic, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., Bengio, Y., et al., 2017. Graph attention networks. stat 1050, 10– 48550.

- [85] Wang, C., 2023. Calibration in deep learning: A survey of the state- of-the-art. arXiv preprint arXiv:2308.01222 .

- [86] Wang, F., Zhang, Y., Jin, D., Jiang, Z., Liu, Y., Knoll, A., Jiang, H., Ying, Y., Zhou, M., 2024a. Magnetic soft microrobot design for cell grasping and transportation. Cyborg and Bionic Systems 5, 0109.

- [87] Wang, G., Bai, L., Nah, W.J., Wang, J., Zhang, Z., Chen, Z., Wu, J., Islam, M., Liu, H., Ren, H., 2024b. Surgical-lvlm: Learning to adapt large vision-language model for grounded visual question answering in robotic surgery. arXiv preprint arXiv:2405.10948 .

- [88] Wang, G., Ren, T.A., Lai, J., Bai, L., Ren, H., 2023. Domain adaptive sim-to-real segmentation of oropharyngeal organs. Medical & Biological Engineering & Computing 61, 2745–2755.


- [89] Wang, G., Xiao, H., Gao, H., Zhang, R., Bai, L., Yang, X., Li, Z., Li, H., Ren, H., 2024c. Copesd: A multi-level surgical motion dataset for training large vision-language models to co-pilot endoscopic submucosal dissection. arXiv preprint arXiv:2410.07540 .

- [90] Wang, H., Yang, G., Zhang, S., Qin, J., Guo, Y., Xu, B., Jin, Y., Zhu, L., 2024d. Video-instrument synergistic network for referring video instrument segmentation in robotic surgery. IEEE Transactions on Medical Imaging .

- [91] Wang, M., Yang, H., Cheng, Q., 2022a. Gcl: Graph calibration loss for trustworthy graph neural network, in: Proceedings of the 30th ACM International Conference on Multimedia, pp. 988–996.

- [92] Wang, M., Yang, H., Huang, J., Cheng, Q., 2024e. Moderate message passing improves calibration: A universal way to mitigate confidence bias in graph neural networks, in: Proceedings of the AAAI Conference on Artificial Intelligence, pp. 21681–21689.

- [93] Wang, Q.,Wei, Y., Yin, J.,Wu, J., Song, X., Nie, L., 2021a. Dualgnn: Dual graph neural network for multimedia recommendation. IEEE Transactions on Multimedia .

- [94] Wang, X., Liu, H., Shi, C., Yang, C., 2021b. Be confident! towards trustworthy graph neural networks via confidence calibration. Ad- vances in Neural Information Processing Systems 34, 23768–23779.

- [95] Wang, Y., Feng, Y., Zhang, L., Zhou, J.T., Liu, Y., Goh, R.S.M., Zhen, L., 2022b. Adversarial multimodal fusion with attention mechanism for skin lesion classification using clinical and dermo- scopic images. Medical Image Analysis 81, 102535.

- [96] Xie, X., Luo, B., Li, Y., Yang, C., Gui, W., 2024. Exploring het- erophily in calibration of graph neural networks. Neurocomputing 604, 128294.

- [97] Xu, Q., Zhang, R., Zhang, Y., Wang, Y., Tian, Q., 2021. A fourier- based framework for domain generalization, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 14383–14392.

- [98] Yamada, Y., Colan, J., Davila, A., Hasegawa, Y., 2024. Multimodal semi-supervised learning for online recognition of multi-granularity surgical workflows. International Journal of Computer Assisted Radiology and Surgery 19, 1075–1083.

- [99] Yang, C., Yang, C., Shi, C., Li, Y., Zhang, Z., Zhou, J., 2024a. Calibrating graph neural networks from a data-centric perspective, in: Proceedings of the ACM onWeb Conference 2024, pp. 745–755.

- [100] Yang, S., Luo, L., Wang, Q., Chen, H., 2024b. Surgformer: Surgical transformer with hierarchical temporal attention for surgical phase recognition, in: International Conference on Medical Image Com- puting and Computer-Assisted Intervention, Springer. pp. 606–616.

- [101] Yang, Y., Soatto, S., 2020. Fda: Fourier domain adaptation for se- mantic segmentation, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 4085–4095.

- [102] Yuan, K., Holden, M., Gao, S., Lee, W., 2022. Anticipation for surgical workflow through instrument interaction and recognized signals. Medical Image Analysis 82, 102611.

- [103] Yuan, K., Holden, M., Gao, S., Lee, W.S., 2021. Surgical work- flow anticipation using instrument interaction, in: Medical Image Computing and Computer Assisted Intervention–MICCAI 2021: 24th International Conference, Strasbourg, France, September 27– October 1, 2021, Proceedings, Part IV 24, Springer. pp. 615–625.

- [104] Yuan, K., Srivastav, V., Navab, N., Padoy, N., 2024. Hecvl: Hierar- chical video-language pretraining for zero-shot surgical phase recog- nition, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer. pp. 306–316.

- [105] Yuan, K., Srivastav, V., Yu, T., Lavanchy, J.L., Mascagni, P., Navab, N., Padoy, N., 2023. Learning multi-modal representations by watching hundreds of surgical video lectures. arXiv preprint arXiv:2307.15220 .

- [106] Yue, Z., Ding, S., Yang, S., Wang, L., Li, Y., 2021. Multimodal in- formation fusion approach for noncontact heart rate estimation using facial videos and graph convolutional network. IEEE Transactions on Instrumentation and Measurement 71, 1–13.

- [107] Zadrozny, B., Elkan, C., 2002. Transforming classifier scores into accurate multiclass probability estimates, in: Proceedings of the eighth ACM SIGKDD international conference on Knowledge discovery and data mining, pp. 694–699.

- [108] Zeng, D., Chen, X., Song, Z., Xue, Y., Cai, Q., 2023. Multimodal interaction and fused graph convolution network for sentiment clas- sification of online reviews. Mathematics 11, 2335.

- [109] Zeng, J., Huang, L., Bai, X., Wang, K., 2024. Dbfft: Adversarial- robust dual-branch frequency domain feature fusion in vision trans- formers. Information Fusion 108, 102387.

- [110] Zhu, H., Xu, D., Huang, Y., Jin, Z., Ding, W., Tong, J., Chong, G., 2024. Graph structure enhanced pre-training language model for knowledge graph completion. IEEE Transactions on Emerging Topics in Computational Intelligence .

- [111] Zhu, L., Wan, B., Li, C., Tian, G., Hou, Y., Yuan, K., 2021. Dyadic relational graph convolutional networks for skeleton-based human interaction recognition. Pattern Recognition 115, 107920.

- [112] Zhu, Z., He, X., Qi, G., Li, Y., Cong, B., Liu, Y., 2023. Brain tumor segmentation based on the fusion of deep semantics and edge information in multimodal mri. Information Fusion 91, 376–387.
