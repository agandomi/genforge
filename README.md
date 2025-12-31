# GenForge: Sculpting Symbolic Solutions with Multi-Population Genetic Programming

![GenForge Logo](GenForge_Logo.png)  <!-- Optional: Add a logo or banner image -->

## Overview

**GenForge** is an open-source Python package for interpretable symbolic modeling through *Genetic Programming (GP)*.  
It introduces a cohesive ecosystem of modules — `gpclassifier`, `gpregressor`, and `spfp` — that together support the evolution of compact, human-readable models for both classification and regression tasks.  
The framework integrates multi-population evolution, semantic partitioning, and ensemble-based symbolic learning to produce transparent, high-performance models.

## Key Features

- **Multi-Population Genetic Programming**  
  Parallel populations evolve diverse solutions that collectively reduce variance and prevent premature convergence.

- **Symbolic Regression and Classification**  
  - `gpregressor`: for continuous-valued prediction using multi-gene symbolic regression.  
  - `gpclassifier`: for categorical prediction with softmax-calibrated symbolic classifiers.

- **Semantic-Preserving Feature Partitioning (SPFP)**  
  The optional `spfp` module partitions high-dimensional feature spaces into complementary, information-preserving “views.”  
  Each view trains a separate GP population; the resulting submodels are then aggregated to yield multi-view ensembles with stronger generalization and interpretability.

- **Parsimony and Calibration**  
  GenForge emphasizes compact symbolic expressions and calibrated confidence estimates for trustable decision boundaries.

- **Reproducibility and Auditability**  
  Built with fixed random seeds, detailed logs, and exportable equations for transparent evaluation and regulatory use.

- **Educational and Research Utility**  
  Modular design and readable structure make it suitable for both academic teaching and applied research in symbolic machine learning, explainable AI, and scientific modeling.

## Authors

- **Mohammad Sadegh Khorshidi**  
  Faculty of Information and IT, University of Technology Sydney  
  Email: msadegh.khorshidi.ak@gmail.com  

- **Navid Yazdanjue**  
  Faculty of Information and IT, University of Technology Sydney  
  Email: navid.yazdanjue@gmail.com

- **Hassan Gharoun**  
  Faculty of Information and IT, University of Technology Sydney  
  Email: hassan.gharoun@student.uts.edu.au

- **Mohammad Reza Nikoo**  
  Department of Civil and Architectural Engineering, Sultan Qaboos University, Muscat, Oman  
  Email: m.reza@squ.edu.om  

- **Fang Chen**  
  Faculty of Information and IT, University of Technology Sydney  
  Email: fang.chen@uts.edu.au  

- **Amir H. Gandomi**  
  Faculty of Information and IT, University of Technology Sydney;  
  University Research and Innovation Center (EKIK), Óbuda University  
  Email: gandomi@uts.edu.au  


## Documentation

A detailed *User Manual* describing installation, configuration, and examples is available here:

 [Download the GenForge User Manual (PDF)](./GenForge_UserManual.pdf)

The manual includes:
- Step-by-step setup and environment configuration
- Example scripts for `gpclassifier`, `gpregressor`, and `SPFPPartitioner`
- Guidance for extending and visualizing GenForge runs

## Reproducible Examples and Capsule
The examples/ directory contains fully reproducible solved examples demonstrating each of the three core modules of GenForge:

- **SPFP** – Semantic-Preserving Feature Partitioning preprocessor for balanced multi-view decomposition.

- **gpclassifier** – Symbolic multi-population genetic-programming classifier.

- **GPRegressor** – Symbolic multi-population genetic-programming regressor.

**Each example directory provides**:

1- The runnable Python script used in the Software Impacts paper capsule.

2- Input datasets and pre-generated results (plots, HTML reports, and logs).

3- An accompanying capsule_readme.txt describing reproduction steps and expected outputs.

These examples reproduce the behavior of the software as presented in the GenForge reproducibility capsule:

- examples/SPFP Example/SPFP_Example.py – Demonstrates feature partitioning.

- examples/ARWPM Classification Example/Example_ARWPM_2.py – Demonstrates classification.

- examples/SFRC Regression Example/SFRC_Example_2.py – Demonstrates regression.

All examples are self-contained; running any script regenerates the corresponding figures and reports found under each folder’s "Produced Results" subdirectory, enabling direct comparison for validation.

## Citation
If you use GenForge in your research, please cite the corresponding papers listed below:
- Khorshidi, M. S., *et al.* (2025). Semantic-Preserving Feature Partitioning for multi-view ensemble learning. *Information Fusion*, 122, 103152. [DOI](https://doi.org/10.1016/j.inffus.2025.103152)  
- Khorshidi, M. S., *et al.* (2025). Multi-population Ensemble Genetic Programming via Cooperative Coevolution and Multi-view Learning for Classification. *arXiv:2509.19339.* [DOI](https://doi.org/10.48550/arXiv.2509.19339)  
- Khorshidi, M. S., *et al.* (2025). From embeddings to equations: Genetic-programming surrogates for interpretable transformer classification. *arXiv:2509.21341.* [DOI](https://doi.org/10.48550/arXiv.2509.21341)  
- Khorshidi, M. S., *et al.* (2025). Domain-Informed Genetic Superposition Programming: A Case Study on SFRC Beams. *arXiv:2509.21355.* [DOI](https://doi.org/10.48550/arXiv.2509.21355)

## Authors

- **Mohammad Sadegh Khorshidi**  
  Faculty of Information and IT, University of Technology Sydney  
  Email: msadegh.khorshidi.ak@gmail.com  

- **Mohammad Reza Nikoo**  
  Department of Civil and Architectural Engineering, Sultan Qaboos University, Muscat, Oman  
  Email: m.reza@squ.edu.om  

- **Fang Chen**  
  Faculty of Information and IT, University of Technology Sydney  
  Email: fang.chen@uts.edu.au  

- **Amir H. Gandomi**  
  Faculty of Information and IT, University of Technology Sydney;  
  University Research and Innovation Center (EKIK), Óbuda University  
  Email: gandomi@uts.edu.au  

## Installation

To install GenForge from source:

```bash
git clone https://github.com/maisamkhorshidi/genforge.git
cd genforge
pip install .




