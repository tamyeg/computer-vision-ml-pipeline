# Computer Vision: Four Applied Machine Learning Tasks

Four Google Colab notebooks developed as part of an Image Processing and Computer Vision laboratory assignment. They cover image classification, object detection, monocular depth estimation, and optical flow. **This repository contains original notebooks and their saved results; full image datasets and trained weights are not included.**

## Projects

| Task | Approach | What the notebook demonstrates |
| --- | --- | --- |
| [1. Image classification](01_image_classification/01_image_classification.ipynb) | Transfer learning using `torchvision.models.mobilenet_v3_small` | Three-class classification (duck, bottle, cola can); dataset split, training, test accuracy and confusion matrix |
| [2. Object detection](02_object_detection_yolov8/02_object_detection_yolov8.ipynb) | YOLOv8n (`ultralytics`) | Detection for the three classes configured in the notebook; training, precision, recall, mAP and IoU analysis |
| [3. Depth estimation](03_depth_estimation/03_depth_estimation.ipynb) | Depth Anything V2 via Hugging Face `transformers` | Monocular depth maps and distance estimates calibrated using a measured reference object |
| [4. Optical flow](04_optical_flow_raft/04_optical_flow_raft.ipynb) | Pretrained RAFT Small in `torchvision` | Dense motion estimates from frame pairs and comparison with manually recorded movement (endpoint error) |

## Reported outcomes

- **Image classification:** 91.67% test accuracy (reported in the laboratory work).
- **YOLOv8 detection:** precision 0.9831, recall 0.8783, mAP@50 0.9499 and mAP@50:95 0.8771 (reported in the laboratory work).
- **Depth and optical flow:** the notebooks show numerical comparisons and visual results; see the respective notebook outputs for details.

These are results from small, task-specific educational datasets and should not be interpreted as production performance or cross-project benchmarks.

## Data access and reproducibility

The notebooks were written for **Google Colab**, using Google Drive at `MyDrive/Lab2_Computational_Intelligence`. Data are **not bundled in this public repository**. To reproduce the experiments, you must provide appropriately structured datasets in your own Drive, review each notebook's first cells, and update its data paths. The detection notebook expects a YOLO-format ZIP containing `data.yaml` and train/valid/test image and label folders. The optical-flow notebook expects the six frame files listed in its `pairs` variable. The depth-estimation notebook expects three images. Classification expects images organised in `duck`, `bottle`, and `cola_can` class folders.

**Limitations:** These notebooks are preserved from their original Colab workflow. They are not yet one-click reproducible without the datasets, and runtime package versions and paths may need adjustment. The YOLO notebook also contains a validation step expecting three classes, so using a dataset with additional classes will require updating that validation and configuration.

## Run in Colab

1. Open a `.ipynb` notebook in Google Colab.
2. Mount your Google Drive if required.
3. Place **your own or appropriately licensed** dataset in the folder layout expected by the notebook, or update the notebook paths.
4. Run cells in order using a runtime suitable for PyTorch. Colab GPU availability varies.
5. Review model outputs, plots and evaluation cells. Trained weights and large input datasets are not provided.

## Privacy and provenance

The notebooks contain saved output figures from the laboratory execution. Before reusing or redistributing any dataset, verify that you have rights to publish it and that no private information appears in images or metadata. No Vinted data were used.

## Author

Tamara Yegorova · Kaunas University of Technology
