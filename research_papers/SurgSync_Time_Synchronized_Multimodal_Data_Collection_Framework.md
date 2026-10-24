# **SurgSync: Time-Synchronized Multi-Modal Data Collection Framework and Dataset for Surgical Robotics** 

- _∗_ Haoying Zhou1 _,_ 2, _∗_ Chang Liu2, Yimeng Wu2, Junlin Wu2 _,_ 3, Zijian Wu4, Yu Chung Lee4, Sara Martuscelli5, Septimiu E. Salcudean<sup>4</sup> , Gregory S. Fischer<sup>1</sup> and Peter Kazanzides<sup>2</sup><sup>_,_3</sup> 

**_Abstract_ —Most existing robotic surgery systems adopt a human-in-the-loop paradigm, often with the surgeon directly teleoperating the robotic system. Adding intelligence to these robots would enable higher-level control, such as supervised autonomy or even full autonomy. However, artificial intelligence (AI) requires large amounts of training data, which is currently lacking. This work proposes SurgSync, a multi-modal data collection framework with offline and online synchronization to support training and real-time inference, respectively. The framework is implemented on a da Vinci Research Kit (dVRK) and introduces (1) dual-mode (online/offline-matching) synchronized recorders, (2) a modern stereo endoscope to achieve image quality on par with clinical systems, and (3) additional sensors such as a side-view camera and a novel capacitive contact sensor to provide ground truth contact data. The framework also incorporates a post-processing toolbox for tasks such as depth estimation, optical flow, and a practical kinematic reprojection method using Gaussian heatmap. User studies with participants of varying skill levels are performed with ex-vivo tissue to provide clinically realistic data, and a network for surgical skill assessment is employed to demonstrate utilization of the collected data. Through the user study experiments, we obtained a dataset of 214 validated instances across multiple canonical training tasks. All software and data are available at surgsync.github.io.** 

## I. INTRODUCTION 

Robot-assisted surgery (RAS) has revolutionized the field of medical science by providing surgeons with enhanced dexterity, advanced visualization, and precision when performing clinical procedures over the past two decades. The success of the da Vinci® Surgical System (dVSS, Intuitive Surgical Inc. Sunnyvale, CA) is the epitome of this revolution [1]. 

In the research domain, high-quality, well-annotated datasets are foundational to progress in artificial intelligence (AI) applications [2] for RAS [3]. They enable data-driven perception, modeling and control, spanning instrument tracking [4]–[9], tissue interaction understanding [10]–[12], skill assessment [13]–[16], and surgery automation [17]–[21]. 

This work was supported in part by NSF AccelNet awards OISE-1927275 and OISE-1927354. 

> _∗_ These authors contributed equally to this work. 

> 1Department of Robotics Engineering, Worcester Polytechnic Institute, Worcester, MA, USA. Emails: hzhou6, gfischer@wpi.edu 

> 2Laboratory for Computational Sensing and Robotics, Johns Hopkins University, Baltimore, MD, USA. 

> 3Department of Computer Science, Johns Hopkins University, Baltimore, MD, USA. Email: pkaz@jhu.edu 

> 4 Robotics and Control Laboratory, the University of British Columbia, Vancouver, Canada. 

> 5 Department of Electronics, Information and Bioengineering, Politecnico di Milano, Milano, Italy 

Despite this momentum, robotics applications, especially for surgical scenarios, face a shortage of high-quality training and validation data due to highly diverse data distribution and the expensive cost of data collection in the physical world. Even though generating synthetic data for training [22]– [28] can mitigate this problem through sim-to-real transfer, the sim-to-real gap limits the complexity of procedures and overall performance. Therefore, there remains a need for realworld datasets to ensure effective evaluation. 

Many available datasets in the surgical robotics domain suffer from three practical limitations: (i) weak or inconsistent time alignment across sensing modalities, which obscures cause-and-effect and degrades sequence models; (ii) legacy imaging pipelines that limit visual fidelity and downstream vision performance; and (iii) narrow task coverage and post-collection tooling that constrain reproducibility and reuse. These gaps are especially consequential for systems like the da Vinci Research Kit (dVRK, also known as dVRK Classic) [29] and dVRK-Si [30], where fine motor actions, bi-manual coordination, and tissue dynamics evolve on subsecond timescales and must be captured coherently across vision and robot states. 

To address these challenges, we present an open-source data collection framework for surgical robotic systems, such as dVRK Classic and dVRK-Si, with the following contributions: 

- **Time-synchronization design pattern:** two synchronized recorders (online/offline-matching) for smooth and continuous teleoperation recording, acknowledging time synchronization as a first-class design constraint; 

- **Upgraded imaging stack:** integration of a modern chipon-tip endoscope (Cornerstone Robotics (CSR) Ltd., Hong Kong, China) with dVRK-Si to enable highperformance imaging stack; 

- **Tool-tissue contact ground-truth sensing:** a capacitive contact sensor, interfaced via a digital input of the dVRK controller, for seamless acquisition of tool-tissue contact ground truth on ex-vivo tissues; 

- **Post-collection processing toolbox:** a configurable and extensible post-collection toolbox for better reusability; 

- **User-study dataset:** user studies for data collection, including multiple practical training procedures performed on phantoms and ex-vivo tissues, such as peg transfer, tissue manipulation, suturing and dissection. 

In addition, we validate the usability of our dataset via implementing a state-of-the-art skill assessment algorithm 



<!-- Start of picture text -->
Optical Flow Calculation Kinematic Reprojection<br>[ θ1 ,  θ2 ,  θ3 ,  θ4 ,  θ5 ,  θ6 ] Joint States<br>Kinematics<br>Attention-Weighted Reprojection<br>SurgSync<br>L<br>R Stereo Disparity Map<br>Matching<br>Rectified Input<br>Time-Synchronized Multi-modality<br>Depth/Disparity from Stereo & Novel Hardware Integrations<br>Contact Sensor Integration Synchronized Synchronized Post-Collection<br>Recorder Data Processing Toolbox<br>Surgical Contact Digital input  dVRK<br>Instrument Sensor Controller<br>Client PC Stereo Images<br>Contact Detection Topic Rectification<br>SoftwaredVRK Online-MatchingRecorder Side Images Kinematic<br>Reprojection<br>ECM Kinematics<br>Stereo Images Data<br>dVRK Master PC  Optical Flow<br>Update Kinematics Kinematic / Dynamic PSM Kinematics<br> based on Endoscope Data Depth<br>EndoscopeNew Video TransmissionFiber-Optic Cable Cable  Control UnitCamera(CCU) HDMI Output HDMI-to-USBadapter Offline-MatchingRecorder Contact  AnnotationEstimationData<br>Timestamps<br>Side Camere Side Images<br>Camera System<br>Data Collection Framework Dataset<br><!-- End of picture text -->

Fig. 1. Overview of the proposed SurgSync dataset and data collection framework. We collect 214 recordings while performing multiple canonical training tasks on ex-vivo tissues (primarily) and phantoms. We also implement post-collection processing using our toolbox. Multiple input modalities, including visual, kinematics/dynamics, tool-tissue contact, event/phase description, are provided for further algorithm training. 

[15] on the suturing-task subset of our data. 

Our proposed framework primarily communicates via Robot Operating System (ROS) and can be extended to general surgical robots supported within the ROS ecosystem. Furthermore, the cross-platform validation demonstrates that the framework can be used by multiple groups to create a combined, diverse, large-scale dataset. An overview of the system and dataset are shown in Fig. 1. 

## II. RELATED WORK 

The surgical robotics domain has been facing a lack of high-quality, well-annotated large-scale public datasets. Researchers have devoted substantial efforts to generate synthetic data through simulation [22]–[28], [31]–[33], however, a persistent trade-off remains among photorealism, physically faithful tool-tissue interactions and task complexity. Therefore, real-world data remain essential for training and rigorous evaluation. 

In 2014, Gao et al. introduced JIGSAWS [34], [35], the first open-source dataset for surgical gesture recognition. It catalyzed AI research in RAS [36] yet exhibits known 

limitations [37], including suboptimal image quality and visual fidelity due to the legacy image pipeline. Also in 2014, we introduced the da Vinci Research Kit (dVRK) [29], an open-source research platform derived from the dVSS, which enabled collection of additional datasets on phantoms for peg transfer [21], tool retraction/palpation [11], pattern cutting [38] and other related training tasks [39]. However, both JIGSAWS and these dVRK datasets were collected with phantoms, which may not capture the dynamics of ex-vivo or in-vivo tool-tissue interactions. 

Beyond these efforts, the EndoVis challenges have released public datasets over several years [6]–[8], providing benchmarks for endoscopic perception and segmentation tasks. Various multi-modal endoscopic datasets [40] have also been released for Large Language Model (LLM) investigations. All aforementioned datasets share one or more constraints: (i) suboptimal time alignment across different modalities, (ii) narrow task coverage, (iii) absence of instrument/robot kinematic data, (iv) missing camera parameters for further calibration, and (v) lack of multi-view image information. 

Finally, our previous work [9] investigated advanced annotation using fluorescence-based approaches, but logged only sparse, discrete events, which limits temporal resolution and can obscure critical dynamics information. 

that fall within a user-defined time tolerance are admitted. This yields slightly uneven inter-sample intervals (irregular ∆ _t_ ), but retains the natural continuity of smooth teleoperation segments and avoids label/feature drift. In our study, a time tolerance of 10 ms is selected to ensure both tight time alignment and consecutive recorder output. The choice of the time tolerance depends on task requirements and hardware I/O performance. This design can be used for real-time scenarios. 

III. METHODOLOGY 

## _A. Synchronized Recorder_ 

We design and implement two practical synchronized recorders in modern C++, which aim to record temporally aligned data streams across multiple sensing modalities. Specifically, they synchronize stereo (or mono) video streams with kinematic data from the patient-side manipulators (PSMs) and endoscopic camera manipulator (ECM) of the dVRK, including both measured and desired (also known as setpoint) states. In addition, the contact sensor signals are seamlessly integrated within the dVRK data stream and recorded with the same time base. We offer two operation modes for the synchronized recorders so that the user can choose between online or offline time-matching approaches. Both recorders operate on smooth and continuous teleoperation, preserving fine-grained temporal context for sequence models, dynamics and interaction analysis. All communications are based on ROS. 

_2) Offline-Matching Recorder:_ Our offline-matching approach decouples recording from time alignment to maximize the recording system efficiency. Therefore, this recorder produces synchronized datasets in two stages: (i) a lightweight recorder logs camera streams to videos and raw kinematic streams to binary files with minimal processing; (ii) an offline post-processing pipeline reconstructs a fixed-rate frame sequence and, for each frame, gathers the _k_ closest samples using nearest-timestamp lookup for subsequent interpolation (we select _k_ = 1 for simplicity). Compared to the onlinematching recorder (which pairs visual and kinematic data in real time), this two-stage design avoids tolerance-based dropping of data during capture, yielding a higher throughput and uniform intervals between synchronized data packets at the cost of requiring more storage and substantial time for post-collection time-matching and interpolation. 

_1) Online-Matching Recorder:_ The design enforces strict time synchronization using multi-threading: only samples 

This offline-matching recorder is intended for data collection where real-time data acquisition is non-essential, for example, building large training datasets for robot policy learning. By recording every camera frame at the target FPS and interpolating kinematics offline using a configurable rule, the pipeline eliminates tolerance-based drops at capture 

|**Algorithm 1** Online-Matching Recorder|lection where real-time data acquisition is non-essential, for|
|---|---|
|**procedure** ONLINEMATCHRECORDER<br>|example, building large training datasets for robot policy|
|**Initialize Subscribers**<br>Subscribe to video streams based on user preferences<br>**For each** enabled arm (PSMs or ECM):<br>Subscribe to <arm>/measured_js/cp/cv<br>|learning. By recording every camera frame at the target<br>FPS and interpolating kinematics offline using a configurable<br>rule, the pipeline eliminates tolerance-based drops at capture|
|Subscribe to <arm>/setpoint_js/cp<br>**If**arm is PSM: subscribe to jaw streams, contact sensors|**Algorithm 2** Offline-Matching Recorder|
|**Data Structures**|**procedure** DETACHEDRECORDER|
|Buffers: queues for images, kinematics (per arm)|**Initialize Directory**|
|SyncedQueue: queue of synchronized data packets|Create <run_id> with subdirs: kin/, meta/|
|Global state: latest setpoint & jaw values (per arm)|**Initialize Subscribers**|
|**Sync Thread**|Subscribe to video streams based on user preferences|
|**Loop** until cut-off requested from keyboard input:<br>|**For each** enabled arm (PSMs or ECM):<br>|
|**If** SyncedQueue full _→_drop oldest images, continue|Subscribe to <arm>/measured_js/cp/cv|
|**If** image buffers empty _→_wait|Subscribe to <arm>/setpoint_js/cp|
|Retrieve image reference timestamp|**If** PSM: subscribe to jaw streams, contact sensors|
|**For each** arm:|**Video Writer** (fixed-rate)|
|GetClosestFromQueue(kin<br>buffer, ref<br>stamp)<br>Construct SyncedPacket including reference time-|Create VideoStreamRecorder(s) for left/right[/side]<br>Record frames at pre-defined FPS|
|-stamp, images and kinematic information|**If** ahead of schedule:|
|Enqueue packet to SyncedQueue|Wait until next frame’s expected timestamp|
|Pop used image(s) from buffer|Image callbacks push frames to corresponding recorder(s)|
|Notify writer threads|**Kinematic Writer** (binary)|
|**Writer Threads** (default pool size: _N_ = 4)|**For each** arm/topic callback:|
|**Loop:**|Append to FastBinWriter file in kin/|
|Wait on SyncedQueue, pop one packet|**Runtime Control**|
|Create temporary folder <timestamp>|Start AsyncSpinner for callbacks|
|Save the images as PNGs|Launch key listener; press ‘q’ _→_stop recording|
|Save the arm kinematics to JSON files|**Shutdown**|
|(Optional) Save data to HDF5 and convert later|Stop recorders; flush and close all writers|
|**Shutdown and Cleanup (Press ‘q’ on the keyboard)**<br>Obtain the cut-off timestamp _tend_|Write start_times.json and end_times.json<br>**end procedure**|
|Release buffers until the reference timestamp passes _tend_|**procedure** DETACHEDPOST-COLLECTIONMATCHING|
|Remove incomplete folders|Decouple the videos into independent image frames|
|ReformatDataStorage() into final dataset layout<br>**end procedure**|Convert binary kinematic files to readable JSON files<br>**end procedure**|



and produces a uniformly sampled training set. It is the more suitable choice for offline learning workflows, where deterministic post-processing is preferable to on-the-fly data matching. 

our experiments. To evaluate the contact sensor performance, we randomly select one instance each from tissue manipulation, suturing and dissection, as shown in section IV-B, and assess detection accuracy. The resulting accuracies are 99.1% for tissue manipulation, 74.3% for dissection and 45.2% for suturing. The misclassifications are primarily attributed to sensor noise, humidity-induced changes in capacitance and intermittent short-circuit events when instruments contacted the same conductive object (e.g., a suturing needle or a small tissue fragment). Those misclassifications are eventually addressed by manual re-annotation using the GUI in section III-D2. 

## _B. Contact Sensor Integration_ 

Our framework supports the integration of custom sensors. For example, we incorporated a contact sensor implemented with an Arduino UNO Rev3 and the Capacitive Sensing Library [41]. The assembled hardware prototype and schematics are shown in Fig. 2. 



<!-- Start of picture text -->
OUT3<br>OUT1<br>OUT2<br>SEND 3 SEND 2 SEND 1<br>a GND Vin<br>REC1<br>REC3 REC2<br>PSM3 PSM2 PSM1 GROUND POWER12V 12V<br><!-- End of picture text -->

## _C. Modern Endoscope Integration_ 

**OUT3** We integrate a contemporary chip-on-tip endoscope with **OUT1 OUT2** the dVRK-Si system to improve visual fidelity and enable **SEND 2 SEND 1 GROUND POWER12V** frame-accurate alignment with robot kinematics and con- ~~<u><mark>a</mark></u>~~ **GND Vin** tact signals. The full imaging pipeline is shown in Fig. 1 **REC1** (bottom), comprising hardware mounting, signal capture, and **REC2** ROS integration. Our integration yields higher-quality images compared to the default dVRK-Si scope and enables coherent **PSM2 PSM1 GROUND POWER12V 12V** multi-modal dataset construction. 

_1) Hardware Integration:_ The endoscope connects to a clinical-grade Camera Control Unit (CCU) via video and fiber-optic illumination cables. The CCU outputs two 1080p HDMI video streams, which are routed to the host PC using HDMI-to-USB frame capture devices. A compact 3D-printed holder mounts the endoscope onto the dVRK-Si Endoscopic Camera Manipulator (ECM) with coaxial alignment, quick attachment, and strain relief. CAD models and overall system setup are shown in Fig. 4. 

Fig. 2. Hardware setup and schematics. The Arduino and the proto-board are shown on the left, the schematics of the electronics on the bottom right, and the connection to the dVRK controller on the top right. 

For monopolar and bipolar instruments, the connection to the sensor is achieved by wrapping a wire around their connectors; for non-electrosurgical (non-polar) instruments, we opened the tool housing, and attached a wire to the rotation coupling of one joint with proper strain relief and wire insulation (Fig. 3). 

_2) Software Integration:_ On the host PC, frames are acquired through the v4l2src backend using gscam & gstreamer and published to ROS image topics. Each frame is timestamped on arrival using the host monotonic clock. This shared time base enables precise temporal alignment during post-processing. The same pipeline supports stereo 





<!-- Start of picture text -->
ECM<br>VR Headset<br>Custom Adapter<br>PSM1<br>PSM2<br>Custom Holder<br>MTMs SUJ<br>CCU<br>Side Camera<br><!-- End of picture text -->

Fig. 3. Instrument wire connection. Non-polar instrument (left), monopolar (center), and bipolar (right). For non-polar instrument, the tool housing is opened and the wire insulation layer is removed for better visualization. 

The library computes the capacitance at the receiving pin in arbitrary units; the returned values depend on the chosen resistor and on the material and size of the contacting object. The sensor exhibits the best performance with human tissue, while ex-vivo animal tissue such as chicken breast, thinsliced beef or pork also provide similarly reliable results. A contact threshold is defined to binarize the signal into contact and non-contact states, which are then transmitted through the Arduino output pin to the digital input of the dVRK controller (Fig. 2, top right). Each digital input is registered in the dVRK software framework by specifying bit ID, FPGA board, and related parameters in a configuration file. 

In this study, we used 2 MΩ resistors and a threshold of 205 for the best performance on the ex-vivo tissues used in 

Fig. 4. Overall experimental setup. The endoscope is mounted on the dVRKSi ECM using a custom holder and adapter. 

or side-view cameras through multiple synchronized ROS nodes. 

_3) Image Quality Comparison:_ Compared to the legacy dVRK-Si endoscope, our integrated imaging system yields significantly sharper visual frames. Quantitatively, the average Laplacian variance [42] is over 30 _×_ higher in our system as shown in Table I, indicating substantially improved details and edge clarity. These improvements benefit downstream perception tasks such as segmentation, optical flow calculation, and depth estimation. Representative frames from each system are shown in Fig. 5. 

TABLE I LAPLACIAN VARIANCE ON SUTURING SEQUENCES. 



<!-- Start of picture text -->
System Mean STD<br>CSR Endoscope (Ours) 529 . 48 23 . 77<br>dVRK-Si Endoscope 16 . 93 2 . 47<br>Fig. 5. Representative frames from the dVRK-Si endoscope (left) and<br>our modern CSR endoscope (right). Brightness settings: dVRK-Si CCU at<br>100%, Fu ours at 30%. Both frames are captured with similar 3 camera-phantom<br>distances.<br><!-- End of picture text -->

## _D. Post-Collection Processing Toolbox_ 

Beyond data acquisition, we provide an extensible and highly configurable post-collection processing toolbox that standardizes calibration and data processing. It supports stereo rectification, image resizing, projection of robot kinematics into the endoscopic image frames, and generation of derived modalities, including disparity/depth and optical flow. In addition, it also produces comprehensive annotations for contact detection and event/phase labels. 

_1) Kinematic Reprojection:_ We propose a practical approach using a Gaussian heatmap to project tool tip 3D position kinematic information [43] to 2D gray-scale endoscopic images. This relies on a hand-eye calibration [44] to correct for the well-known inaccuracy of the dVRK kinematics [45]. 

Given the hand-eye calibration and stereo camera parameters, the 3D point _p_ = ( _xp, yp, zp_ )<sup>_T_</sup> on the dVRK PSM tool-yaw link (shown in Fig. 6) is projected to the image plane ( _up, vp_ ) using a pinhole camera model. We can then generate a Gaussian heatmap _G_ centered at ( _up_ , _vp_ ): 



where ( _px, py_ ) indexes image pixels and _σx_ , _σy_ control the spread of the heatmap along the image axes. 

Eventually, we compute an element-wise product between the rectified grayscale stereo images ( _Igray_ ) and the heatmap mask _G_ to obtain attention-weighted images ( _Ia_ = _G⊙Igray_ ) that emphasize the region of interest for further interaction analysis. The overall pipeline is shown in Fig. 6. 



<!-- Start of picture text -->
Chip-on-tip ⊙  =<br>PSM1 Endoscope<br>𝐺𝐺  𝐼𝐼𝑔 𝑔 𝑔 𝑔 𝐼𝐼𝑔𝑔<br>𝐺𝐺 𝑃 𝑃 𝑃 𝐼𝐼𝑔𝑔 𝑃 𝑃 𝑃<br>PSM2<br>⊙ =<br>𝐺𝐺 𝑃 𝑃 𝑃 𝐼𝐼𝑔𝑔 𝑃 𝑃 𝑃<br>Grayscale<br>“oa” Gaussian Rectified og Attention-<br>PSM tool-yaw<br>Heatmap Mask  Image weighted Image<br>link<br><!-- End of picture text -->

Fig. 6. Kinematic Reprojection Pipeline. This approach projects the Cartesian positions of the PSM tool-yaw link to attention-weighted images. 

_2) Data Annotation:_ We developed a custom data annotator with Graphical User Interface (GUI) using PyQt as shown in Fig. 7, enabling users to label contact detection and event/phase description during playback of consecutive frames. 



<!-- Start of picture text -->
Stereo  Multi-Frame Labelling<br>Images<br>(Required) Load<br>Real-time<br>Optional  Annotation<br>Images (*) Labels<br>Display<br>Playback<br>Options<br>Save<br>Label Annotations<br>(*) can show side-view image / •• Phase, Step, Event : hierarchical workflow descriptionsGesture: Arm-level task descriptions<br> disparity map …. • Contact Detection (Boolean)<br><!-- End of picture text -->

Fig. 7. Data Annotation GUI 

_3) Depth Estimation:_ As shown in Fig. 1, we perform disparity estimation using the FoundationStereo model [46] and obtain disparity images. Furthermore, the disparity can be converted to depth via _depth_ = _disparityf ×b_<sup>,where</sup><sup>_f_and</sup> _b_ represent the focal length and baseline distance of the stereo endoscope that can be directly obtained from the stereo camera parameters. 

_4) Optical Flow:_ As shown in Fig. 1, we also employ the RAFT model [47] to compute dense optical flow between consecutive frames. We then apply a custom magnitudeaware filter to suppress low-magnitude noise for better image output. 

IV. EXPERIMENT SETUP 

## _A. System Configuration_ 

Fig. 4 shows the primary system configuration at Johns Hopkins University (JHU). To assess cross-platform generality, we deployed a second configuration on the dVRK Classic at the University of British Columbia (UBC). Both systems share the same camera architecture (stereo endoscope + sideview camera), although the UBC system uses the legacy dVRK Classic endoscope rather than the modern unit and omits the contact sensor. In the UBC setup, cameras stream to 

a client workstation, while the I/O-intensive dVRK software framework runs on the master workstation. The hand-eye calibration was not performed at UBC. Both setups use an Intel® RealSense<sup>_T M_</sup> RGBD camera as the side-view camera. 

Both recorder systems run on workstation-class CPUs: Intel® Xeon(R) W-2245 (JHU) and Intel® Core<sup>_T M_</sup> i9-11900 (UBC). All cameras produce raw image streams with a resolution of 1080p and 30 Hz (side-view) or 60 Hz (stereo endoscope) refresh rate. The dVRK framework runs at 1 kHz. At JHU, the offline-matching recorder sustains up to 10 Hz and omitting the side-view camera can increase the rate to 15 Hz. 

TABLE II 

DATASET DISTRIBUTION, USER GROUPS DEFINED IN SECTION IV-B 

|**Training**|**User**|**Number o**|**f Instances**|**Total**|
|---|---|---|---|---|
|**Task**|**Group**|**Online**|**Offline**||
|Suturing|N|13|2||
|and|E|36|12|104|
|Knot Tying|P|2|39||
|PeTransfer|N|7||18|
|g|E|11|-||
|TiMilti|N|9||21|
|ssue anpuaon|E|12|-||
||N|6|-||
|Dissection|E|15|9|71|
||P|1|40||



## _B. User Study_ 

We conduct a user study to collect data for multiple canonical training tasks on phantoms or ex-vivo tissues, which include but are not limited to: 

- Peg transfer (phantoms and gauze in chicken breast) 

- Single interrupted suturing practice (chicken breast) 

- Tissue manipulation (chicken hearts/breast, beef and pork) 

- Dissection following a trace (beef and pork) 

13 human subjects (3 female, 10 male) participated in our user study. 4 subjects have limited knowledge of dVRK operation and are considered novice users (N); 5 are familiar with dVRK operation and are considered experienced users (E); 4 are surgeons and are considered professional users (P). 

For both recorders, we select the ROS image topic timestamp of the left stereo endoscope camera as the reference timestamp to perform time latency analysis for synchronization quality evaluation across different modalities. Table III summarizes descriptive statistics and Fig. 8 shows the raw time latency distribution. The few outliers could be due to: (1) the time latency for the control loop of the dVRK software, (2) extra time to numerically solve inverse kinematics near singularities, and (3) suboptimal CPU performance. 

The stereo camera parameters are included in the dataset. Notably, our post-collection processing tool can also properly handle data subsets involving endoscope movements if the hand-eye calibration is performed. 

## V. RESULTS 

## _A. Cross-Platform Validation_ 

The open-source data collection framework successfully executed on two different setups at two different institutions (JHU and UBC). The UBC setup achieved similar performance using a larger online-matching time tolerance of 100 ms due to the heavy I/O load from the dVRK software framework. 

## _B. Dataset Distribution_ 

We collected 214 instances over multiple training tasks in our user study. The total frame count per instance varies, depending on the task and the user’s skill level. Of the 214 validated instances, 9 were collected on the UBC setup and only used the online-matching recorder without the hand-eye calibration. 102 instances were collected using the offline-matching recorder. 96 offline-matching instances were collected within the Intuitive abdominal dome. A brief dataset distribution with respect to training tasks and human subjects’ skill level is shown in Table II. 

## _C. Synchronized Recorder Evaluation_ 

To evaluate time-synchronization performance, we randomly selected 4 instances for each recorder from the JHU dataset. The online-matching recordings contain 1523 frames in total and the offline-matching recordings contain 4876 frames. For each frame, excluding the contact sensor modalities which are latched topics, the recorded data contains 20 time-synchronized ROS topics. For the offline-matching recorder, we use all five closest samples for the calculation. 

TABLE III 

SYNCHRONIZED RECORDER COMPARISON AND TIME ANALYSIS 

|**Attribute**|**Online-**<br>**matching**|**Offline-**<br>**matching**|
|---|---|---|
|Time latency<br>mean _±_ std (ms)|6.36 _±_ 4.72|1.35 _±_ 0.81|
|Time latency<br>median (ms)|5.58|1.33|
|Recording<br>frequency (Hz)|4.04 _±_ 1.69|10|
|Ready to use|Yes|No<br>(post-collection time matching<br>and interpolation required)|





Fig. 8. Overall raw time latency distribution of all ROS topics 

## _D. Dataset Validation in Skill Assessment_ 

We evaluate our synchronized data on the task of skill assessment to demonstrate the feasibility of the collected data and the post-collection processing toolbox. We follow a data-driven regression approach using the unified multi-path framework for automatic surgical skill assessment [15]. 

After interpolating the second-stage outputs from the offline-matching recorder, the resulting dataset shares identical data structure with the online-matching one. We selected 43 instances from 8 users with varying experience levels. Each instance is split into suturing or knot-tying segments based on manual annotations. 

An experienced user with knowledge of animal surgery graded all instances using the global rating score (GRS) [15] in the range of 6 to 30. The GRS is derived from an objective rubric [48] with six categories: (1) respect for tissue, (2) suture/needle handling, (3) time and motion, (4) flow of operation, (5) quality of final product, and (6) overall performance. Each category is scored on a scale of 1–5. 

The model takes three kinds of synchronized input modalities: (i) kinematic features (14D), including Cartesian positions (6D), measured velocities (6D), and gripper openings (2D); (ii) visual features (2048D) extracted from a ResNet101 encoder applied to RGB images; and (iii) one-hot encoded gesture labels (14D) based on 14 common surgical gestures, following the JIGSAWS gesture taxonomy [34]. All modalities are aligned per-frame using our synchronized recorder to ensure coherent temporal context across streams. 

To ensure diverse coverage of skill levels, we construct four stratified folds for cross-validation [49]. Each fold partitions data into training and test sets with a balance of GRS distribution and user identity. 

Our model follows the multi-path design of Liu et al. [15], with temporal encoders for kinematic (T), visual (V), and gesture (E) streams. Each head outputs the five GRS scores and is trained with equal-weight mean squared error on normalized labels. A temporal contrastive loss is added for regularization. We report Spearman’s rank correlation coefficient (SROCC) averaged over four folds (Table IV). 

TABLE IV 

SKILL ASSESSMENT PERFORMANCE ON OUR SYNCHRONIZED DATASET. SROCC IS REPORTED FOR EACH CROSS-VALIDATION FOLD. 

|**Task**|**Fold 1**|**Fold 2**|**Fold 3**|**Fold 4**|**Mean** _±_ **Std**|
|---|---|---|---|---|---|
|Suturing|0.658|0.828|0.844|0.881|**0.803** _±_ **0.086**|
|Knot Tying|0.870|0.875|0.618|0.699|**0.765** _±_ **0.111**|



## VI. DISCUSSION AND CONCLUSION 

In this manuscript, we propose SurgSync, a multi-modal data collection framework with (1) dual-mode (online/offline) time-synchronized recorder, (2) advanced hardware integration of a modern stereo endoscope and a novel contact sensor on the dVRK-Si, and (3) a post-collection toolbox for rectification, depth estimation and a kinematic reprojection approach using a Gaussian heatmap. We conduct a user study to acquire datasets and demonstrate feasibility by training and 

evaluating a skill-assessment model on the collected data. The performance of the synchronized recorders is highly dependent on the workstation hardware, yet it can still obtain reliable data even with suboptimal hardware. All open-source software and data are available at https://surgsync.github.io/ 

In future work, we will perform additional data collection using the offline-matching approach to build a large-scale training dataset. Moreover, we will include additional data with more types of instruments and training tasks and add experiments with in-vivo or cadaver environments if possible. 

## ACKNOWLEDGMENT 

Thanks to Dale Bergman, Alessandro Gozzi and the Intuitive Foundation for all the hardware support of the dVRKSi. Thanks to Cornerstone Robotics Ltd. for providing the modern chip-on-tip endoscope. 

## REFERENCES 

- [1] C. D’Ettorre, A. Mariani, A. Stilli, F. R. y Baena, P. Valdastri, A. Deguet _et al._ , “Accelerating Surgical Robotics Research: A Review of 10 Years With the da Vinci Research Kit,” _IEEE Robotics & Automation Magazine_ , vol. 28, no. 4, pp. 56–78, 2021. 

- [2] A. O’Neill, A. Rehman, A. Maddukuri, A. Gupta, A. Padalkar, A. Lee _et al._ , “Open X-Embodiment: Robotic Learning Datasets and RT-X Models: Open X-Embodiment Collaboration 0,” in _IEEE Intl. Conf. on Robotics and Automation (ICRA)_ , 2024, pp. 6892–6903. 

- [3] T. Haidegger, S. Speidel, D. Stoyanov, and R. M. Satava, “RobotAssisted Minimally Invasive Surgery—Surgical Robotics in the Data Age,” _Proceedings of the IEEE_ , vol. 110, no. 7, pp. 835–846, 2022. 

- [4] H. Ding, J. Zhang, P. Kazanzides, J. Y. Wu, and M. Unberath, “CaRTS: Causality-driven robot tool segmentation from vision and kinematics data,” in _Medical Image Computing and Computer-Assisted Intervention (MICCAI)_ . Springer, 2022, pp. 387–398. 

- [5] C. D’Ambrosia, F. Richter, Z.-Y. Chiu, N. Shinde, F. Liu, H. I. Christensen _et al._ , “Robust surgical tool tracking with pixel-based probabilities for projected geometric primitives,” in _IEEE Intl. Conf. on Robotics and Automation (ICRA)_ . IEEE, 2024, pp. 15 455–15 462. 

- [6] N. Fernandes, E. Oliveira, and N. F. Rodrigues, “Future perspectives of deep learning in laparoscopic tool detection, classification, and segmentation: A systematic review,” in _IEEE Intl. Conf. on Serious Games and Applications for Health (SeGAH)_ . IEEE, 2023, pp. 1–8. 

- [7] H. Ding, Y. Zhang, T. Lu, R. Liang, H. Shu, L. Seenivasan _et al._ , “SegSTRONG-C: Segmenting surgical tools robustly on nonadversarial generated corruptions–an EndoVis’ 24 challenge,” _arXiv preprint arXiv:2407.11906_ , 2024. 

- [8] H. Xu, A. Weld, C. Xu, A. Roddan, J. Cartucho, M. A. Karaoglu _et al._ , “SurgRIPE challenge: Benchmark of surgical robot instrument pose estimation,” _Medical Image Analysis_ , p. 103674, 2025. 

- [9] Z. Wu, A. Schmidt, R. Moore, H. Zhou, A. Banks, P. Kazanzides _et al._ , “SurgPose: a dataset for articulated robotic surgical tool pose estimation and tracking,” in _IEEE Intl. Conf. on Robotics and Automation (ICRA)_ , 2025. 

- [10] N. Yilmaz, J. Y. Wu, P. Kazanzides, and U. Tumerdem, “Neural network based inverse dynamics identification and external force estimation on the da Vinci Research Kit,” in _IEEE Intl. Conf. on Robotics and Automation (ICRA)_ . IEEE, 2020, pp. 1387–1393. 

- [11] Z. Chua, A. M. Jarc, and A. M. Okamura, “Toward force estimation in robot-assisted surgery using deep learning with vision and robot state,” in _IEEE Intl. Conf. on Robotics and Automation (ICRA)_ , 2021, pp. 12 335–12 341. 

- [12] M. D. I. Reyzabal, M. Chen, W. Huang, S. Ourselin, and H. Liu, “DaFoEs: Mixing datasets towards the generalization of vision-state deep-learning force estimation in minimally invasive robotic surgery,” _IEEE Robotics and Automation Letters_ , vol. 9, no. 3, pp. 2527–2534, 2024. 

- [13] B. Varadarajan, C. Reiley, H. Lin, S. Khudanpur, and G. Hager, “Data-derived models for segmentation with application to surgical assessment and training,” in _Medical Image Computing and ComputerAssisted Intervention (MICCAI)_ . Springer, 2009, pp. 426–434. 

- [14] A. Zia and I. Essa, “Automated surgical skill assessment in RMIS training,” _Intl. Journal of Computer Assisted Radiology and Surgery_ , vol. 13, no. 5, pp. 731–739, 2018. 

- [15] D. Liu, Q. Li, T. Jiang, Y. Wang, R. Miao, F. Shan _et al._ , “Towards unified surgical skill assessment,” in _IEEE/CVF Conf. on Computer Vision and Pattern Recognition_ , 2021, pp. 9522–9531. 

- [16] K. Lam, J. Chen, Z. Wang, F. M. Iqbal, A. Darzi, B. Lo _et al._ , “Machine learning for technical skill assessment in surgery: a systematic review,” _NPJ Digital Medicine_ , vol. 5, no. 1, p. 24, 2022. 

- [17] H. Zhou, Y. Jiang, S. Gao, S. Wang, P. Kazanzides, and G. S. Fischer, “Suturing tasks automation based on skills learned from demonstrations: A simulation study,” in _Intl. Symp. on Medical Robotics (ISMR)_ . IEEE, 2024, pp. 1–8. 

- [18] P. M. Scheikl, N. Schreiber, C. Haas, N. Freymuth, G. Neumann, R. Lioutikov _et al._ , “Movement primitive diffusion: Learning gentle robotic manipulation of deformable objects,” _IEEE Robotics and Automation Letters_ , vol. 9, no. 6, pp. 5338–5345, 2024. 

- [19] J. W. Kim, T. Z. Zhao, S. Schmidgall, A. Deguet, M. Kobilarov, C. Finn _et al._ , “Surgical Robot Transformer (SRT): Imitation learning for surgical tasks,” _arXiv preprint arXiv:2407.12998_ , 2024. 

- [20] J. W. Kim, J.-T. Chen, P. Hansen, L. X. Shi, A. Goldenberg, S. Schmidgall _et al._ , “SRT-H: A hierarchical framework for autonomous surgery via language-conditioned imitation learning,” _Science Robotics_ , vol. 10, no. 104, p. eadt5254, 2025. 

- [21] Y. Long, A. Lin, D. H. C. Kwok, L. Zhang, Z. Yang, K. Shi _et al._ , “Surgical embodied intelligence for generalized task autonomy in laparoscopic robot-assisted surgery,” _Science Robotics_ , vol. 10, no. 104, p. eadt3093, 2025. 

- [22] E. Colleoni, P. Edwards, and D. Stoyanov, “Synthetic and real inputs for tool segmentation in robotic surgery,” in _Medical Image Computing and Computer-Assisted Intervention (MICCAI)_ . Springer, 2020, pp. 700–710. 

- [23] T. Zeng, G. Loza Galindo, J. Hu, P. Valdastri, and D. Jones, “Realistic surgical image dataset generation based on 3D Gaussian Splatting,” in _Medical Image Computing and Computer-Assisted Intervention (MICCAI)_ . Springer, 2024, pp. 510–519. 

- [24] J. A. Barragan, J. Zhang, H. Zhou, A. Munawar, and P. Kazanzides, “Realistic Data Generation for 6D Pose Estimation of Surgical Instruments,” in _IEEE Intl. Conf. on Robotics and Automation (ICRA)_ , 2024, pp. 13 347–13 353. 

- [25] J. Wu, H. Zhou, P. Kazanzides, A. Munawar, and A. Liu, “SurgicAI: A hierarchical platform for fine-grained surgical policy learning and benchmarking,” _Advances in Neural Information Processing Systems_ , vol. 37, pp. 63 771–63 789, 2024. 

- [26] S. Yang, Z. Wu, M. Hong, Q. Li, D. Shen, S. E. Salcudean, and Y. Jin, “Instrument-Splatting: Controllable photorealistic reconstruction of surgical instruments using Gaussian Splatting,” _arXiv preprint arXiv:2503.04082_ , 2025. 

- [27] Z. Wu, A. Schmidt, P. Kazanzides, and S. E. Salcudean, “Augmenting efficient real-time surgical instrument segmentation in video with point tracking and Segment Anything,” _Healthcare Technology Letters_ , vol. 12, no. 1, p. e12111, 2025. 

- [28] M. Moghani, N. Nelson, M. Ghanem, A. Diaz-Pinto, K. Hari, M. Azizian _et al._ , “SuFIA-BC: Generating high quality demonstration data for visuomotor policy learning in surgical subtasks,” _arXiv preprint arXiv:2504.14857_ , 2025. 

   - [34] Y. Gao, S. S. Vedula, C. E. Reiley, N. Ahmidi, B. Varadarajan, H. C. Lin _et al._ , “JHU-ISI Gesture and Skill Assessment Working Set (JIGSAWS): A surgical activity dataset for human motion modeling,” in _MICCAI workshop: M2cai_ , vol. 3, 2014, p. 3. 

   - [35] N. Ahmidi, L. Tao, S. Sefati, Y. Gao, C. Lea, B. B. Haro _et al._ , “A dataset and benchmarks for segmentation and recognition of gestures in robotic surgery,” _IEEE Trans. on Biomedical Engineering_ , vol. 64, no. 9, pp. 2025–2041, 2017. 

   - [36] A. Moglia, K. Georgiou, E. Georgiou, R. M. Satava, and A. Cuschieri, “A systematic review on artificial intelligence in robot-assisted surgery,” _International Journal of Surgery_ , vol. 95, p. 106151, 2021. 

   - [37] A. Hendricks, M. Panoff, K. Xiao, Z. Wang, S. Wang, and C. Bobda, “Exploring the limitations and implications of the JIGSAWS dataset for robot-assisted surgery,” _IEEE Robotics and Automation Letters_ , vol. 9, no. 11, pp. 9534–9541, 2024. 

   - [38] Y. Sharon, T. Nevo, D. Naftalovich, L. Bahar, Y. Refaely, and I. Nisky, “Augmenting robot-assisted pattern cutting with periodic perturbationscan we make dry lab training more realistic?” _IEEE Trans. on Biomedical Engineering_ , 2024. 

   - [39] I. Rivas-Blanco, C. J. P. Del-Pulgar, A. Mariani, G. Tortora, and A. J. Reina, “A surgical dataset from the da Vinci Research Kit for task automation and recognition,” in _Intl. Conf. on Electrical, Computer, Communications and Mechatronics Engineering (ICECCME)_ . IEEE, 2023, pp. 1–6. 

   - [40] S. Liu, B. Zheng, W. Chen, Z. Peng, Z. Yin, J. Shao _et al._ , “A comprehensive evaluation of multi-modal large language models for endoscopy analysis,” _arXiv preprint arXiv:2505.23601_ , 2025. 

   - [41] P. Badger and P. Stoffregen, “Arduino Capacitive Sensing Library,” 2016. [Online]. Available: https://playground.arduino.cc/ Main/CapacitiveSensor/index-2.html 

   - [42] S. Pertuz, D. Puig, and M. A. Garcia, “Analysis of focus measure operators for shape-from-focus,” _Pattern Recognition_ , vol. 46, no. 5, pp. 1415–1432, 2013. 

   - [43] T. E. Lee, J. Tremblay, T. To, J. Cheng, T. Mosier, O. Kroemer, D. Fox, and S. Birchfield, “Camera-to-robot pose estimation from a single image,” in _Intl. Conf. on Robotics and Auto. (ICRA)_ . IEEE, 2020, pp. 9426–9432. 

   - [44] B. Burkhart and A. Deguet, “dVRK hand-eye calibration package,” 2022. [Online]. Available: https://github.com/jhu-dvrk/dvrk camera registration 

   - [45] Z. Cui, J. Cartucho, S. Giannarou, and F. R. y Baena, “Caveats on the first-generation da Vinci Research Kit: latent technical constraints and essential calibrations,” _IEEE Robotics & Automation Magazine_ , 2023. 

   - [46] B. Wen, M. Trepte, J. Aribido, J. Kautz, O. Gallo, and S. Birchfield, “FoundationStereo: Zero-shot stereo matching,” in _Computer Vision and Pattern Recognition (CVPR)_ , 2025, pp. 5249–5260. 

   - [47] Z. Teed and J. Deng, “RAFT: Recurrent all-pairs field transforms for optical flow,” in _European Conf. on Computer Vision_ . Springer, 2020, pp. 402–419. 

   - [48] J. Martin, G. Regehr, R. Reznick, H. Macrae, J. Murnaghan, C. Hutchison _et al._ , “Objective structured assessment of technical skill (OSATS) for surgical residents,” _British Journal of Surgery_ , vol. 84, no. 2, pp. 273–278, 1997. 

   - [49] Y. Tang, Z. Ni, J. Zhou, D. Zhang, J. Lu, Y. Wu _et al._ , “Uncertaintyaware score distribution learning for action quality assessment,” in _IEEE/CVF Conf. on Computer Vision and Pattern Recognition_ , 2020, pp. 9839–9848. 

- [29] P. Kazanzides, Z. Chen, A. Deguet, G. S. Fischer, R. H. Taylor, and S. P. DiMaio, “An Open-Source Research Kit for the da Vinci® Surgical System,” in _IEEE Intl. Conf. on Robotics and Automation (ICRA)_ , 2014, pp. 6434–6439. 

- [30] K. Xu, J. Y. Wu, A. Deguet, and P. Kazanzides, “dVRK-Si: The Next Generation da Vinci Research Kit,” in _IEEE Intl. Symp. on Medical Robotics (ISMR)_ , 2025, pp. 185–191. 

- [31] J. Xu, B. Li, B. Lu, Y.-H. Liu, Q. Dou, and P.-A. Heng, “SurRol: An open-source reinforcement learning centered and dVRK compatible platform for surgical robot learning,” in _IEEE/RSJ Intl. Conf. on Intelligent Robots and Systems (IROS)_ , 2021, pp. 1821–1828. 

- [32] A. Munawar, J. Y. Wu, G. S. Fischer, R. H. Taylor, and P. Kazanzides, “Open simulation environment for learning and practice of robotassisted surgical suturing,” _IEEE Robotics and Automation Letters_ , vol. 7, no. 2, pp. 3843–3850, 2022. 

- [33] Q. Yu, M. Moghani, K. Dharmarajan, V. Schorp, W. C.-H. Panitch, J. Liu _et al._ , “ORBIT-Surgical: An open-simulation framework for learning surgical augmented dexterity,” in _IEEE Intl. Conf. on Robotics and Automation (ICRA)_ , 2024, pp. 15 509–15 516. 

