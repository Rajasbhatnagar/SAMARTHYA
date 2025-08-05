# SAMARTHYA  
**SAR-based Maritime Analytics for Recognition, Tracking, and Heading Yield Assessment**

SAMARTHYA is a modular deep learning pipeline for detecting vessel wakes and estimating heading angles from Synthetic Aperture Radar (SAR) imagery. Built for maritime situational awareness, the system combines object detection, binary classification, and hybrid regression to support surveillance, reconnaissance, and autonomous monitoring applications.

## Overview

This project processes SAR images to:
- Detect wake signatures using YOLOv11
- Filter out false detections using ResNet18
- Estimate vessel heading using a hybrid ResNet34 + MLP model

The system is fully automated and trained on both real and synthetic SAR wake datasets, requiring no manual labeling. Outputs include bounding boxes and precise heading angles suitable for real-time or batch inference systems.

## Datasets

SAMARTHYA was trained and evaluated using two sources:

### 1. [OpenSAR Wake Dataset](https://drive.google.com/file/d/14VkPYnb1BsmOvw_JTwtVFM-_qVpc4Udu/view)
- Public SAR wake dataset
- Contains labeled SAR images with visible vessel wake patterns

### 2. Synthetic SAR Wake Dataset (SynthSAR)
- Procedurally generated SAR wake imagery
- Simulates vessel wakes across 0°–360° headings
- Includes metadata with heading ground truth (degrees and sin/cos)

All annotations were generated automatically using script-based logic.

## Setup

```bash
git clone https://github.com/yourusername/SAMARTHYA.git
cd SAMARTHYA
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
