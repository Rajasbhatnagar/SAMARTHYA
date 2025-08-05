# SAMARTHYA

**SAMARTHYA**: *SAR‑based Maritime Analytics for Recognition, Tracking, and Heading Yield Assessment*  
A modular deep learning pipeline for detecting vessel wakes and estimating heading using Synthetic Aperture Radar (SAR) imagery.

## Overview

SAMARTHYA processes Sentinel‑1 SAR images to detect vessel wakes, classify valid detections, and estimate vessel heading by fusing visual and geometric features. This enables automated maritime domain awareness for naval, coastal security, and search & rescue applications.

## Dataset

**OpenSARShip** (Sentinel‑1 ship chips dataset; ~11,346 samples) from Shanghai Jiaotong University¹  
  Download via [OpenSAR official site](http://opensar.sjtu.edu.cn/) :contentReference[oaicite:1]{index=1}

**Synthetic SAR Wakes**  
  Procedurally generated wake images spanning full 0°‑360° heading distribution with simulated SAR noise.

## Installation

```bash
git clone https://github.com/yourusername/SAMARTHYA.git
cd SAMARTHYA
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
