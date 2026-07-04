# Iris Flower Classification Dashboard 🌸

A premium machine learning web application built using **Python**, **Scikit-Learn**, and **Streamlit** to classify Iris flower species (*Setosa*, *Versicolor*, *Virginica*) in real-time based on their physical measurements.

---

## 🚀 Features

* **Real-Time Classification**: Select sepal and petal measurements via interactive sliders to predict the species instantly.
* **Vibrant Class Confidence Charts**: Displays a live horizontal probability distribution bar chart showing the model's confidence for all three classes.
* **Premium Species Imagery**: Renders custom, high-resolution photographs of the classified species in real-time.
* **Interactive Exploratory Data Analysis (EDA)**: Fully responsive scatter plots visualizing sepal and petal measurements from the dataset, highlighting the user's custom input point with a distinctive red star (⭐).
* **Model Diagnostics**: Detailed metrics showing the performance of the classification model (precision, recall, confusion matrix) and relative feature importances.
* **Clean Code Architecture**: Separated model training pipelines (`train_model.py`), exploratory analysis (`eda_analysis.py`), and the user interface (`app.py`).

---

## 📈 Model Performance

Three machine learning classifiers were evaluated on the training set (80/20 split):

| Classifier | Test Accuracy | Status |
| :--- | :---: | :---: |
| **Support Vector Classifier (SVC)** | **96.67%** | **Selected & Deployed** |
| **Logistic Regression** | 93.33% | Backup |
| **Random Forest** | 90.00% | Backup |

The best-performing model (SVC) is serialized using `joblib` along with the scaling parameters.

---

## 🛠️ Setup & Running Instructions

Follow these steps to run the application locally:

### 1. Clone the repository
```bash
git clone https://github.com/Omranjan110/CodeAlpha_ProjectName.git
cd CodeAlpha_ProjectName
```

### 2. Create and Activate a Virtual Environment
* **Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```
* **Windows (CMD):**
  ```cmd
  python -m venv .venv
  .venv\Scripts\activate.bat
  ```
* **macOS/Linux (Terminal):**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Model Training Pipeline (Optional)
If you want to train the model from scratch:
```bash
python train_model.py
```

### 5. Launch the Web App
```bash
streamlit run app.py
```

---

## 📦 File Structure

```text
├── assets/                  # High-quality flower images for Setosa, Versicolor, Virginica
├── static_plots/            # Generated EDA visualization charts
├── app.py                   # Streamlit web application entry point
├── train_model.py           # Model training and serialization script
├── eda_analysis.py          # Exploratory Data Analysis script
├── requirements.txt         # Package dependencies
├── .gitignore               # Ignored files (venv, pycache, etc.)
└── README.md                # Project documentation
```
