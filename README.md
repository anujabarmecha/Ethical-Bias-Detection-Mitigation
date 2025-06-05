# Ethical-Bias-Detection-Mitigation

A complete framework to detect, evaluate, and mitigate bias in machine learning models, ensuring fair and ethical AI decision-making. This tool provides an interactive interface for uploading datasets, analyzing fairness metrics, applying debiasing techniques, and visualizing the results.

---

## 🚀 Features

- 📂 Upload custom datasets (CSV format)
- 📊 Fairness metric analysis (Disparate Impact, Equal Opportunity, etc.)
- 🛠️ Apply bias mitigation techniques (Reweighing, Prejudice Remover, etc.)
- 📈 Visualize bias before and after mitigation
- 📡 RESTful API backend using FastAPI
- 🌐 Streamlit-based frontend UI
- ✅ Supports `aif360` and `fairlearn` libraries



---

## 🔧 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ethical-bias-framework.git
cd ethical-bias-framework
```
### 2. Install Requirements
```
pip install -r requirements.txt
```
💡 Note: Ensure Python 3.8+ is installed. If using Windows, consider a virtual environment.

### 3. Run Backend (FastAPI)
```
cd backend
uvicorn api:app --reload
```
By default, the API will run at http://127.0.0.1:8000.

### 4. Run Frontend (Streamlit)
In a new terminal:

```
cd frontend
streamlit run app.py
```

### 📂 Sample Usage
Upload your CSV dataset via the UI.

Select the protected attribute (e.g., gender, race).

Choose fairness metrics and mitigation techniques.

View results — bias reports, visualizations, and updated metrics.

### 📊 Supported Bias Mitigation Algorithms
Preprocessing: Reweighing, Optimized Preprocessing

In-processing: Adversarial Debiasing, Prejudice Remover

Post-processing: Equalized Odds, Reject Option Classification
