# Machine Learning and Deep Learning Playground

This repository serves as a practical testing environment for note-taking and code implementations corresponding to the following curriculum:

1. **[Estadística práctica para Machine Learning con Python](https://youtube.com/playlist?list=PLtKI8MF06ilAy3rwpJNJrl6Zg3SPrDJDW&si=jId4SL4mJzsDEKQK)** - by Franco di Leo.
2. **[Practical Deep Learning for Coders](https://youtube.com/playlist?list=PLfYUBJiXbdtSvpQjSnJJ_PmDQB_VyT5iU&si=-zPB5zezzLZgGcrT)** - Fast.ai (Jeremy Howard).
3. **[PyTorch for Deep Learning & Machine Learning](https://www.youtube.com/watch?v=V_xro1bcAuA)** - freeCodeCamp (Daniel Bourke).

## Objectives
* Establish a rigorous statistical foundation for data analysis before applying predictive models.
* Implement structural components of Deep Learning frameworks using PyTorch and Fast.ai abstractions.
* Verify analytical derivations of backpropagation and optimization bounds through numerical computation.
* Complete the official repository templates and custom exercise sets from each course.

---

## Directory Structure (Work In Progress)

```text
├── 1_practical_statistics/    # Statistical foundations (Franco di Leo)
│   ├── data/                  # .csv files used in notebooks
│   ├── docs/                  # Statistic Cheatsheet 
│   ├── notebooks/             # Dataframes, Graphs (matplotlib and seaborn) and Distributions
├── 2_practical_deep_learning/ # High-level Deep Learning implementations (Fast.ai)
│   ├── notebooks/             # Top-down applications (Computer Vision models)
│   └── answers/               # Text-based questionnaire solutions
└── 3_pytorch_for_ml_and_dl/   # Low-level PyTorch mechanics (freeCodeCamp)
    ├── 00_fundamentals/       # Tensor manipulation, reproducibility, and CUDA device management
    ├── 01_workflow/           # Training loops: Forward pass, loss calculation, backward pass
    ├── 02_classification/     # Binary and multiclass classification (Logits, Sigmoid, Softmax)
    └── 03_computer_vision/    # CNN construction from scratch (TinyVGG architecture replica)