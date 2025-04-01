Veesion Data Science Challenge Submission
===================================================

Hi, I'm Nelson,

This is my solution to the Veesion Data Science Challenge.

All the answers and detailed analysis for **Questions 1 to 4** are in the notebook: `answers.ipynb`.

It includes:
- FP vs Recall plots
- The performance uniformity analysis
- Explanations of the thresholds optimization solutions (greedy vs global)
- Deployment recommendations

### Setup & Run Instructions

1. Create and activate a virtual environment:

   `python -m venv veesion-challenge-venv`

   `source veesion-challenge-venv/bin/activate`


2. Install required dependencies:

   `pip install -r requirements.txt`

3. Run the solutions from CLI (Note that Q1 and Q2 responses are in `answers.ipynb`)


---------------------------------------------------
### Question 3: Optimization

The goal: Given a target reduction in total false positives (#FP),
find a threshold configuration across all cameras that minimizes the loss of true positives (#TP).

A) Greedy Optimization (naive implementation):

For a target 50% un FP reduction run:

`python main.py --source data/production_alerts_meta_data.csv --target_fp_reduction 0.9 --strategy greedy  --save_path results/greedy_optim_50.json`


B) Global Optimization (linear programming formulation solved using `PuLP` library):

For a target 50% un FP reduction run:

`python main.py --source data/production_alerts_meta_data.csv --target_fp_reduction 0.5 --strategy global  --save_path results/global_optim_50.json`


---------------------------------------------------
### Question 4: Deployment-ready Execution


This script supports running optimization for a **single camera**, simulating a real deployment scenario.

You can run one camera optimization like this:

`!python main.py --source data/production_alerts_meta_data.csv --store be-ad-1420-hugo-3 --camera_id 10 --target_fp_reduction 0.5 --strategy global --save_path results/camera_optim_global_150.json`


The complete response is in the notebook: `answers.ipynb`