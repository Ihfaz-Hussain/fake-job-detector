#Fake Job Posting Detection (Research Internship)

This repository contains the code and experiments from my research internship at the **University of Nottingham Malaysia**.  
The project explores whether **supervised NLP models** can detect fraudulent job postings using both **textual** (title, description, requirements) and **structural** (logo, telecommuting, salary, etc.) features.

---

## Project Overview
- Preprocess job posting text with **tokenization, stopword removal, stemming, and lemmatization**.
- Engineer **punctuation** and **binary/structured features** from the dataset.
- Compare **three models** fairly on the same train/test split:
  1. Logistic Regression (TF-IDF + structured features)
  2. GRU sequence model (text order + structured features)
  3. LSTM sequence model (text order + structured features)
- Evaluate results using **Accuracy, Precision, Recall, F1**, and **Confusion Matrix**.

---

## Results (Example Run)

| Model                               | Accuracy | Fake Precision | Fake Recall | Fake F1 |
|-------------------------------------|----------|----------------|-------------|---------|
| Logistic Regression (TF-IDF + str.) | 0.966    | 0.602          | 0.902       | 0.722   |
| GRU (text + structured)             | 0.883    | 0.252          | 0.717       | 0.373   |
| LSTM (text + structured)            | 0.806    | 0.161          | 0.717       | 0.263   |

*Observation:* Logistic Regression performs best with TF-IDF + structured features.  
GRU and LSTM capture sequences but struggle due to class imbalance and limited fake samples.

---

## Tech Stack
- **Python** – pandas, numpy, matplotlib  
- **NLP** – NLTK (tokenization, stopwords, stemming, lemmatization)  
- **Machine Learning** – scikit-learn (Logistic Regression, TF-IDF)  
- **Deep Learning** – TensorFlow / Keras (GRU, LSTM)  

---

##How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/fake-job-detection.git
   cd fake-job-detection
