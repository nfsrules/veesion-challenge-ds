# Veesion Data Science Challenge

This is my solution to the Data Science technical test. For maintenability and better usage 


## Directory Structure
<pre> <code> version-challenge-ds/ ├── README.md # Project overview and instructions ├── main.py # Command-line script for Q3 and Q4 ├── requirements.txt # Dependencies ├── data/ # Input dataset (not included) ├── optimizer/ │ ├── __init__.py │ ├── base_models.py # Abstract base classes for camera and multi-camera optimizers │ ├── global_optim.py # Multi-camera optimization (Q3) │ ├── io_utils.py # Functions to load the dataset │ └── local_optim.py # Single-camera optimization model (Q3) ├── results/ # Optimization results ├── tests/ # Unit tests (out of scope due to time constraints) └── __init__.py </code> </pre>

## Answers

### Question 1: Plot #FP vs. Recall curves for each camera and save as separate images.
- **Answer**: Refer to the (`analysis_notebook.ipynb`). The notebook includes data exploration, curve generation using `matplotlib`, and saving plots to the `results/` directory.

### Question 2: Analyze the model's performaance uniformness across cameras
- **Answer**: Refer to Section 2 of the Jupyter notebook (`analysis_notebook.ipynb`).

### Question 3: Optimization: Find an optimal per-camera threshold to reduce the total #FP by a target, without loosing too mane #TP.
- **Greedy/Naive implementations**: 
  - **Camera-level**: Select the threshold that minimizes the #TP lost per #FP avoided (reduce false alarms without loosing too much real theft). Implemented in `optimizer/local_optim.py` (`_fit_cost`).
  - **Multi-camera level**: Sort the cameras by cost ratio (TP lost / FP saved) and optimize them iterativelly until reaching the target #FP reduction. Implemented in `optimizer/global_optim.py` (`MultiCameraOptimizer`).
  - **Running the solution**:
        `python main.py --source data/production_alerts_meta_data.csv --target_fp_reduction 12000 --save_path results/optim.json`

 - **Smarter implementations**: 
  - **Camera-level**: 
  - **Multi-camera level**: 

### Question 4: Deployment: Making sure the code is simple to deploy
- **Task**: Prepare the script for deployment with flexible data sources, command-line execution, and single-camera support.
- **Implementation**: 
  - **CLI**: `main.py` using `argparse` for CL execution at multi and single camera level.
  - **Data Source**: `optimizer/io_utils.py` includes a `DataLoader` class (support only CSV now for lack of time).
  - **Output**: Stateless camera and multi camera optimization classes with easy serializable JSON results `--save_path`.

  - **Running the solution**:
        `python main.py --source data/production_alerts_meta_data.csv --store be-ad-1420-hugo-3 --camera_id 10 --target_fp_reduction 100`

## Setup
1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt