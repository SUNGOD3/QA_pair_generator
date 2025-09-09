# AutoQA-Gen

[![Google Slides](https://img.shields.io/badge/Google%20Slides-Presentation-blue?logo=google-slides&logoColor=white)](https://docs.google.com/presentation/d/1Ga9ogwiEXKRqdqsDmR9rGu8m2F2ue6CkTfkjj6or3L4/edit?usp=sharing)



## Problem to Solve

The creation of high-quality QA datasets is fundamental for developing and evaluating modern question-answering (QA) systems.
However, existing approaches often face challenges such as:

- Limited flexibility when adapting to different domains.
- High development and annotation costs.
- Lack of systematic quality control in generated QA pairs.
- Most methods are tightly coupled to a specific task or domain. When switching to a new dataset or topic, the entire pipeline often needs to be redesigned.

AutoQA-Gen addresses these challenges by introducing a scalable and extensible framework that standardizes the QA-pair generation process while maintaining flexibility for domain-specific customization.


## System Architecture

AutoQA-Gen transforms raw data into high-quality QA pairs through a six-stage modular pipeline:

1. Initialization – Input setup and preprocessing.
2. Data Expansion – Expanding raw text into structured QA-pairs.
3. Knowledge Graph Construction – Structuring knowledge for better QA grounding.
4. Data Fusion – Combining multiple data sources into a unified representation.
5. Data Filtering – Ensuring quality control by removing low-value or redundant pairs.
6. Data Augmentation – Expanding and diversifying QA pairs for robustness.


## Setup
```bash
git clone https://github.com/SUNGOD3/QA_pair_generator.git
cd QA_pair_generator
pip install -r requirements.txt
```

See demo.py/demo_boolQ/PAQ_demo for how to use it.