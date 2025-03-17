# Knowledge Distillation on DeBERTa-V3-Large with GLUE Tasks

## Overview

This repository provides an implementation of knowledge distillation using the **DeBERTa-V3-large** model, specifically applied to the **GLUE benchmark tasks**. Our approach builds upon the DeBERTa-v3 repository, which you can explore here: [DeBERTa-v3 Repository](https://github.com/microsoft/DeBERTa)

## Getting Started

This repository includes a detailed pipeline for knowledge distillation.

To perform knowledge distillation, use the following command:

```bash
sh run_kd.sh
```

This script requires **two arguments**:

1. **Task Index (1–8)** – Specifies the GLUE task index.
2. **Data Source** – Indicates the dataset source for training.

## Dataset Preparation

Due to storage limitations, the **GLUE dataset** is not included in this repository. You can download the dataset from:


[Google Drive](https://drive.google.com/file/d/1uI6HbkksVLxuOqsx0uB7eeMLsFeQ57cw/view?usp=sharing)
