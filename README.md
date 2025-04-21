# Renewables in Electricity Markets
## Course Assignment 2: Stakeholder Perspective by Group 24


## Installation Guide

### 1. Install Conda (if not already installed)
Make sure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/) installed on your system.

### 2. Create the Conda Environment
Run the following command to create the environment from `env.yaml`:

```bash
conda env create -f env.yaml
```

### 3. Activate the Environment
Once the installation is complete, activate the environment:

```bash
conda activate renewables
```

### 4. Verify Installation
Check that the required packages are installed:

```bash
python -c "import gurobipy, pandas, numpy, matplotlib, seaborn, scipy; print('All packages installed successfully!')"
```

## Running the Exercises
Each step contains a Python script that can be run from the command line with arguments.

Example Usage:
Replace <args> with the appropriate command-line arguments as needed.

Step 1:
``` bash
python src/step1/step1.py
```

Step 2:
``` bash
python src/step2/step2.py
```
