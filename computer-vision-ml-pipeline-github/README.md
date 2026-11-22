# Computer Vision & Deep Learning — Four Applied Experiments

**Image classification · Object detection · Metric depth estimation · Optical flow**

Four Google Colab notebooks developed for an Image Processing and Computer Vision laboratory at **Kaunas University of Technology**. The project examines four different questions: *What object is in an image? Where is it? How far away is it? How did it move?*

The notebooks contain the original code and saved outputs from the experiments. They were developed as separate tasks, **not** as one integrated production system. Full datasets and trained model weights are not distributed in this repository.

## Project at a glance

| Task | Model / approach | Key result | Notebook |
|---|---|---|---|
| **1. Image classification** | Transfer learning with MobileNetV3 Small | **91.67% test accuracy** | [Open notebook](01_image_classification/01_image_classification.ipynb) |
| **2. Object detection** | YOLOv8n | **0.9499 mAP@50** | [Open notebook](02_object_detection_yolov8/02_object_detection_yolov8.ipynb) |
| **3. Depth estimation** | Depth Anything V2 Metric Indoor Small | **0.059 m mean absolute error** after reference calibration | [Open notebook](03_depth_estimation/03_depth_estimation.ipynb) |
| **4. Optical flow** | Pretrained RAFT Small | **71.29 px overall mean endpoint error** across seven manually evaluated points | [Open notebook](04_optical_flow_raft/04_optical_flow_raft.ipynb) |

> **Interpretation:** These measurements come from small, task-specific educational experiments. They are not comparisons of the four models on a common dataset and should not be treated as production-level benchmarks.

## Repository structure

```text
computer-vision-ml-pipeline/
├── 01_image_classification/
│   └── 01_image_classification.ipynb
├── 02_object_detection_yolov8/
│   └── 02_object_detection_yolov8.ipynb
├── 03_depth_estimation/
│   └── 03_depth_estimation.ipynb
├── 04_optical_flow_raft/
│   └── 04_optical_flow_raft.ipynb
├── DATASETS.md
├── README.md
└── .gitignore
```

## 1. Image classification — MobileNetV3 Small

**Goal:** Classify a single object image as a `duck`, `bottle`, or `cola_can`.

**Dataset:** Three class folders in Google Drive. The initial inventory contained **54 duck images, 51 bottle images, and 52 cola can images** (157 total). The notebook randomly selects **50 per class** using seed `42`, then creates a per-class split of **35 training, 7 validation, and 8 test images**. The resulting experiment has **105 training, 21 validation, and 24 test images**.

**Method**
1. Inspect the class folders and display five examples per class.
2. Resize images to **224 × 224**, convert them to tensors, and normalize with ImageNet means and standard deviations.
3. Load pretrained `torchvision.models.mobilenet_v3_small` weights and replace the final classifier layer for three classes.
4. Fine-tune using cross-entropy loss, the Adam optimizer (`learning rate = 0.0001`), batch size **16**, and **10 epochs**.
5. Save the model with the best validation loss; evaluate it on the held-out test set; inspect incorrect predictions and a confusion matrix.

**Result:** **91.67% test accuracy** on 24 test images (22 correct predictions). Training loss decreased from **1.0871** to **0.0826**, while validation loss decreased from **0.9227** to **0.1130** over the 10 reported epochs.

**What it shows:** Dataset organisation, controlled data splitting, transfer learning, training/validation monitoring, and error analysis.

**Limitation:** The test set is small. The score should not be taken as evidence of robustness to different cameras, backgrounds, or lighting conditions.

## 2. Object detection — YOLOv8n

**Goal:** Locate and label multiple objects within an image using bounding boxes.

**Dataset:** YOLO-formatted annotated images with **three classes: `bottle`, `cola_can`, `duck`**. The notebook reports **78 training images, 17 validation images, and 17 test images**, each with its corresponding label file. This is a **different dataset and task** from Part 1. Some source photos may depict additional physical objects, but the uploaded training configuration explicitly validates only these three detection classes.

**Method**
1. Extract the dataset ZIP from Google Drive and inspect `data.yaml`.
2. Configure train/validation/test paths and verify the three labels.
3. Fine-tune pretrained **YOLOv8n** (`yolov8n.pt`) for **30 epochs**, using image size **640** and batch size **8**.
4. Evaluate the saved best weights on the test split.
5. Inspect precision, recall, mAP, predictions, and example bounding-box IoU values.

### Test-set performance

| Metric | Value |
|---|---:|
| Precision | **0.9831** |
| Recall | **0.8783** |
| mAP@50 | **0.9499** |
| mAP@50:95 | **0.8771** |

The test set contains **17 images and 21 labelled object instances**. In the saved per-class validation output, mAP@50 is approximately **0.891 for bottle**, **0.964 for cola can**, and **0.995 for duck**.

**What it shows:** Annotation-aware dataset preparation, detection training, interpretation of bounding-box metrics, and per-class evaluation.

**Limitation:** High scores on this small test set do not establish generalisation to a broad marketplace or real-world deployment. Training requires the annotated dataset and model weights, which are not bundled here.

## 3. Monocular depth estimation — Depth Anything V2

**Goal:** Estimate the distance of objects from a single RGB image and compare estimates with manual measurements.

**Data:** Three photographed scenes with six labelled objects in each: **can, duck, lip balm, figure toy, bottle, and box**. Manually measured camera-to-object distances are recorded in metres in the notebook.

**Method**
1. Load `depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf` using the Hugging Face `transformers` depth-estimation pipeline.
2. Generate and visualise depth maps for the three photographs.
3. Select approximate image coordinates for each object.
4. Extract median depth from a **7 × 7 pixel neighbourhood** around each selected point.
5. Calibrate predicted depth separately for each image using the **bottle** as a reference object with known distance.
6. Calculate absolute and percentage errors for the other objects, excluding the reference bottle from summary evaluation.

### Distance estimation results

| Image | Mean absolute error | Mean percentage error |
|---|---:|---:|
| Image 1 | 0.053 m | 6.793% |
| Image 2 | 0.080 m | 18.452% |
| Image 3 | 0.043 m | 7.247% |
| **Overall, excluding reference bottles** | **0.059 m** | **10.83%** |

**What it shows:** Model inference, spatial sampling, calibration using ground truth, and quantitative error analysis.

**Limitation:** Results depend on object-point selection and a known reference distance. Calibration means these figures do not demonstrate fully independent, reference-free metric distance estimation.

## 4. Optical flow — RAFT Small

**Goal:** Estimate pixel displacement between two images and compare the model's motion vectors against manually tracked objects.

**Data:** **Three frame pairs** (six images in total). The notebook includes manually selected coordinates for seven object tracks across the three pairs.

**Method**
1. Load pretrained `torchvision.models.optical_flow.raft_small` weights.
2. Resize frames so dimensions are compatible with the model.
3. Compute dense horizontal and vertical motion fields.
4. Visualise frame pairs and dense flow magnitude.
5. Compare predicted motion around manually selected points with displacement calculated from coordinates.
6. Report **endpoint error (EPE)** and plot manual versus predicted motion vectors.

### Flow magnitude

| Frame pair | Mean magnitude | Maximum magnitude |
|---|---:|---:|
| Pair 1 | 16.13 px | 182.09 px |
| Pair 2 | 125.14 px | 173.08 px |
| Pair 3 | 78.24 px | 144.01 px |

### Manual-point endpoint error

| Frame pair | Evaluated objects | Mean EPE |
|---|---:|---:|
| Pair 1 | 1 | 25.25 px |
| Pair 2 | 3 | 12.91 px |
| Pair 3 | 3 | 145.01 px |
| **Overall** | **7** | **71.29 px** |

**What it shows:** Dense motion estimation, vector calculations, interpreting model predictions, and analysing errors across different motion scenarios.

**Limitation:** Performance differs greatly by frame pair, especially for Pair 3. These are a small number of manually selected object points, not a full optical-flow benchmark.

## Technology stack

| Area | Tools |
|---|---|
| Programming and environment | Python, Google Colab, Google Drive |
| Deep learning | PyTorch, torchvision |
| Classification | MobileNetV3 Small |
| Detection | Ultralytics YOLOv8 |
| Depth estimation | Hugging Face Transformers, Depth Anything V2 |
| Optical flow | RAFT Small |
| Data and evaluation | NumPy, pandas, scikit-learn |
| Images and visualisation | Pillow, Matplotlib |

## Data and reproducibility

The original experiments used the author's Google Drive directory:

```text
MyDrive/Lab2_Computational_Intelligence/
```

**Datasets are not included** in this GitHub repository. To reproduce the experiments:

1. Open the relevant `.ipynb` file in [Google Colab](https://colab.research.google.com/).
2. Review the notebook's first cells to identify its expected Drive directory and file names.
3. Mount your own Google Drive where the notebook requests it.
4. Place your own or appropriately licensed images in the expected structure, or adapt the dataset paths.
5. Install packages where prompted and select a suitable Colab runtime.
6. Execute cells in order; for model evaluation, train or provide the model weights expected by that notebook.

**Dataset-specific expectations**
- **Classification:** `Part1_Classification/dataset/{duck,bottle,cola_can}/`, containing class images.
- **Detection:** a YOLO-format ZIP with `data.yaml`, images, and bounding-box labels across `train`, `valid`, and `test` folders.
- **Depth:** three input photographs stored under `Part3_Depth_Estimation/images/`; the corresponding real-world distances are written directly in the notebook.
- **Optical flow:** six images stored under `Part4_Optical_Flow/frames/` with the frame names referenced in the notebook.

See [DATASETS.md](DATASETS.md) for additional notes. Data paths are currently Colab-specific, and the notebooks are **not guaranteed to run unchanged on another computer or Drive account**. Saved notebook outputs document the original run even when the original images or weights are unavailable.

## Skills demonstrated

- Preparing image datasets and applying appropriate train/validation/test splits
- Fine-tuning pretrained neural networks
- Evaluating classification and detection models with task-specific metrics
- Working with bounding boxes, image coordinates, depth maps, and motion vectors
- Comparing predictions with observed measurements
- Investigating limitations and communicating results through tables and visualisations

## Possible next improvements

- Add small, redistributable sample images and representative figures outside the notebooks.
- Make paths configurable rather than tied to a specific Google Drive account.
- Add pinned package versions and a reproducible environment setup.
- Run broader evaluations on unseen images and lighting/background conditions.
- Expand error analysis and document failure cases for each task.

## Author

**Tamara Yegorova**  
MSc student, Artificial Intelligence in Computer Science — Kaunas University of Technology

[GitHub profile](https://github.com/tamyeg)

---




