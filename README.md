# Parkinson's Disease Detection System

## Project Overview

Parkinson's Disease Detection System is a machine learning-based project that predicts the possibility of Parkinson's disease using voice measurement features.

This project uses a trained machine learning model to analyze biomedical voice data and classify whether a person is likely to have Parkinson's disease or not.

**Note:** This project is created for educational purposes and is not a medical diagnosis system.

---

## Features

- Machine learning based Parkinson's disease prediction
- Voice feature analysis
- Random Forest classification model
- Model accuracy evaluation
- Prediction confidence score
- Simple command-line interface

---

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Joblib

---

## Machine Learning Algorithm

**Random Forest Classifier**

Random Forest is used for classification because it performs well with complex datasets and multiple input features.

---

## Dataset

The project uses the Parkinson's Disease dataset containing biomedical voice measurements.

### Input Features

- MDVP:Fo(Hz)
- MDVP:Fhi(Hz)
- MDVP:Flo(Hz)
- Jitter
- Shimmer
- NHR
- HNR
- RPDE
- DFA
- PPE

### Output

- 0 → Healthy
- 1 → Parkinson's Disease

---

## Project Structure

```
Parkinson_Disease_Detector/

│
├── app.py                 # Prediction application
├── train_model.py         # Model training script
├── parkinsons.data        # Dataset
├── parkinson_model.pkl    # Trained machine learning model
├── requirements.txt       # Required libraries
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_LINK
```

Go to the project folder:

```bash
cd Parkinson_Disease_Detector
```

Install required libraries:

```bash
pip install -r requirements.txt
```

---

## How to Run

### Step 1: Train the Model

Run:

```bash
python train_model.py
```

This will:

- Load the Parkinson's dataset
- Train the Random Forest model
- Calculate model accuracy
- Save the trained model

---

### Step 2: Run the Detection System

Run:

```bash
python app.py
```

Enter the patient number:

```
Enter patient number: 0
```

Example Output:

```
Patient: phon_R01_S01_1

Result:
Parkinson's Disease Detected

Confidence: 99%
```

---

## Model Performance

The model is evaluated using accuracy score.

Accuracy may vary depending on the training and testing data split.

---

## Future Improvements

- Add a graphical user interface
- Create a web application using Streamlit
- Add patient information management
- Generate PDF reports
- Add database support
- Deploy the application online

---

## Author

Rimita Ghosh

---

## License

This project is created for educational purposes.
