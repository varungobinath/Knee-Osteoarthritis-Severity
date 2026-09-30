python -m venv venv

.\venv\Scripts\Activate.ps1

pip install -r requirements.txt

python -m streamlit run app.py

pip install streamlit==1.37.1 numpy==1.23.5 matplotlib==3.7.5 pillow==10.4.0 tensorflow==2.13.1


   - Medical Context & KL Grading: Explains Knee Osteoarthritis (KOA) grading using the standard Kellgren-Lawrence scale (Grades 0 to 4: Healthy, Doubtful, Minimal, Moderate,
     Severe).
   - Dataset Analysis: Outlines the 9,786 knee radiograph distribution across train (5,778), val (826), test (1,656), and auto_test (1,526), detailing class imbalance handling via
     balanced class weighting and augmentation.
   - Deep Learning Pipelines:
     - Fine-tuned Xception (67% Balanced Acc), ResNet50 (65% Balanced Acc), and Inception-ResNet-v2 (64% Balanced Acc).
     - Soft voting ensembles with uniform mean, accuracy-weighting, and F1-score weighting (reaching 71.43% Balanced Accuracy and 68.72% Test Accuracy).
   - Explainable AI (Grad-CAM): Details the gradient-weighted class activation mapping implementation, peak local activation detection, and superimposed visual heatmaps.
   - Streamlit Web Application (app/app.py): UI overview, X-ray upload, classification metrics, and heatmaps.
   - Setup & Execution Guides: Installation steps, virtual environment activation (PowerShell and Bash), and execution commands for both Streamlit and Jupyter.