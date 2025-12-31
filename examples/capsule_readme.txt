CAPSULE TITLE: GenForge Reproducible Capsule

DESCRIPTION:
This capsule reproduces the main results of the Software Impacts paper, including demonstration scripts for the gpclassifier, gpregressor, and SPFPPartitioner modules of the GenForge package. The capsule provides end-to-end examples that showcase the symbolic modeling, multi-population search, and semantic-preserving feature partitioning capabilities of GenForge.

The capsule contains three subdirectories:
1. ARWPM Classification Example
2. SFRC Regression Example
3. SPFP Example

Each folder includes the runnable Python script, input data, and previously generated outputs for verification.

- ARWPM Classification Example: Demonstrates the use of `GPClassifier` on the ARWPM dataset.  
  To reproduce, run `Example_ARWPM_2.py` located in `ARWPM Classification Example/`.  
  Pre-generated results and reports are located under `ARWPM Classification Example/Produced Results`.

- SFRC Regression Example: Demonstrates the use of `GPRegressor` for symbolic regression on the SFRC dataset.  
  To reproduce, run `SFRC_Example_2.py` located in `SFRC Regression Example/`.  
  Pre-generated results and visual diagnostics are stored under `SFRC Regression Example/Produced Results`.

- SPFP Example: Demonstrates the `SPFPPartitioner` functionality for semantic-preserving feature partitioning.  
  To reproduce, run `SPFP_Example.py` located in `SPFP Example/`.  
  The corresponding results are stored under `SPFP Example/Produced Results`.

All demonstration scripts are self-contained, reproducible, and designed for examination of the GenForge software behavior. Each example produces diagnostic plots, HTML reports, and logs identical in structure to those in the corresponding *Produced Results* folders, enabling direct comparison.

Note: No experimental results were reported in the Software Impacts paper; this capsule serves exclusively as a functional demonstration and validation resource for the GenForge package.

REQUIREMENTS:
- Python ≥ 3.10
- numpy, pandas, scipy, scikit-learn, matplotlib, seaborn, joblib
- genforge==1.0

INSTALLATION:
1. **Clone the GenForge repository**:
   ```bash
   git clone https://github.com/maisamkhorshidi/genforge.git
   cd genforge

2. **Create a clean Python environment**:
 python -m venv genforge_env

3. **Activate the environment**:
  genforge_env\Scripts\activate

4. **Upgrade pip and install dependencies**:
  pip install --upgrade pip setuptools wheel
 pip install -r requirements.txt

5. **Install GenForge locally**:
 pip install .

6. **Verify installation**:
 python -c "import genforge; print(genforge.__version__)"

The expected output should be:
1.0


