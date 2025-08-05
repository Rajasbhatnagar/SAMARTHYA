# SAMARTHYA

**SAMARTHYA**: *SAR-based Maritime Analytics for Recognition, Tracking, and Heading Yield Assessment*  
An end-to-end deep learning pipeline for automated vessel wake detection and heading estimation in Synthetic Aperture Radar (SAR) imagery.

---

## Overview

SAMARTHYA is designed to process SAR images to detect vessel wakes, classify their validity, and estimate vessel heading using a combination of visual and geometric features. The pipeline is built for enhancing maritime situational awareness and is adaptable for integration into UAV, satellite, or coastal surveillance systems.

---

## Dataset

This project uses two primary data sources:

### 1. OpenSAR Wake Dataset
- Publicly available dataset containing SAR images with visible vessel wakes.
- Source: [OpenSAR Project – Wake Data](http://opensar.sjtu.edu.cn/)
- Direct Link: [Download Wake Dataset](http://opensar.sjtu.edu.cn/data/wake.zip)

### 2. Synthetic SAR Wake Dataset (SynthSAR)
- Custom-generated wake images covering a full range of headings (0°–360°).
- Includes simulated SAR texture, noise, and wake geometry.
- Metadata includes ground truth heading in both degrees and (sin, cos) format.

No manual annotations were used; all labels were generated using procedural logic and automated heading generation.

---

## Installation

```bash
git clone https://github.com/yourusername/SAMARTHYA.git
cd SAMARTHYA
python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate.bat on Windows
pip install -r requirements.txt
