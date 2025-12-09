# The Impact of Batch Size on Gradient Descent Stability and Generalisation  
### A Tutorial Using the Credit Card Fraud Detection Dataset

This repository contains the code, notebook, figures, and report for a tutorial exploring how **batch size** influences the behaviour of gradient descent during neural network training. Using the **Credit Card Fraud Detection dataset**, we compare training dynamics, evaluation metrics, and generalisation performance across four different batch sizes: **16, 64, 256, and 1024**.

This project demonstrates the trade-offs between noisy vs. stable gradients, convergence speed, sensitivity to rare classes, and overall model performance.

---

## 📌 Project Overview

This tutorial replicates the analysis presented in the accompanying PDF report. It includes:

- Explanation of batch size and its role in gradient descent  
- Neural network training with different batch sizes  
- Comparison of training/validation loss curves  
- Comparison of training/validation AUC curves  
- Test accuracy and AUC metrics  
- Confusion matrices for smallest vs. largest batch sizes  
- Discussion on stability, generalisation, and sharp vs. flat minima  

The tutorial aims to help students understand **why batch size matters**, how it affects optimisation dynamics, and how to choose a suitable batch size in practice.

---

## 📊 Dataset

**Dataset:** Credit Card Fraud Detection  
Source: Kaggle  
Link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The dataset contains:

- 284,807 credit card transactions  
- PCA-transformed numerical features (V1–V28), plus Time & Amount  
- Highly imbalanced labels:  
  - `0` → legitimate  
  - `1` → fraud (~0.17%)

### ⚠️ Dataset Not Included in Repository  
GitHub does not allow uploading files larger than 25 MB through the web interface, and the Credit Card Fraud dataset (~150 MB) exceeds this limit.

Please download the dataset manually from Kaggle:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

After downloading, place the file inside:

```
data/creditcard.csv
```
---

## 🧪 Methods Used

### 1. Model Architecture  
A simple multilayer perceptron (MLP):

- Dense (32, ReLU)  
- Dense (16, ReLU)  
- Dense (1, Sigmoid)  

Matches the architecture described in the report (page 4) :contentReference[oaicite:1]{index=1}.

### 2. Batch Sizes Tested  
We train the same model using:

- **16**  
- **64**  
- **256**  
- **1024**  

As justified in the report (page 4) for covering the spectrum from noisy to stable updates.

### 3. Metrics Evaluated  

Following the evaluation methodology outlined in the report (pages 4–5):

- Training loss  
- Validation loss  
- Training AUC  
- Validation AUC  
- Test accuracy  
- Test AUC  
- Confusion matrices (batch 16 vs 1024)

---

## 🛠️ How to Run the Notebook

## 1. Install dependencies

```bash
pip install numpy pandas scikit-learn matplotlib seaborn tensorflow
```

## 2. Ensure dataset is available

## Download creditcard.csv from Kaggle and place it in:

### ⚠️ Dataset Not Included in Repository  

GitHub does not allow uploading files larger than 25 MB through the web interface, and the Credit Card Fraud dataset (~150 MB) exceeds this limit.

Please download the dataset manually from Kaggle:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud



## 3. Run the notebook

```
(24069723)_ML_Assignment.ipynb
```

### All plots generated in the report (loss curves, AUC curves, confusion matrices).
We can see the plots that are used in the report in the plots folder.
---

### 📁 Folder Structure
```

project/
│
├── data/
│   └── creditcard.csv         
│
├── plots/                   
│   ├── Trainingloss_for_diff_batchsizes.png
│   ├── Training_AUC_forDiffBatchSizes_bs64.png
│   ├── Validationloss_for_diff_batchsizes.png
│   ├── ValidationAUC_for_diff_batchsizes.png
│   ├── confusion_matrix_bs16.png
│   ├── confusion_matrix_bs1024.png
│   └── Placeholder.txt
│
├── (24069723)_ML_Assignment.ipynb
├── (24069723)_ML_Report.pdf
├── README.md
└── LICENSE
```

## 📈 Test Results

| Batch Size | Test Accuracy | Test AUC |
|------------|---------------|----------|
| 16         | 0.9399        | 0.9768   |
| 64         | 0.9796        | 0.9764   |
| 256        | 0.9812        | 0.9723   |
| 1024       | 0.9821        | 0.9747   |

➡️ Accuracy remains similar due to class imbalance.  
➡️ AUC reveals real differences in fraud detection ability.

---

## 🧩 Confusion Matrices (Summary)

- **Batch 16** detects **more fraud cases**, meaning fewer false negatives.  
- **Batch 1024** misses more frauds due to overly stable gradient descent.  

(Figures shown on page 8 of the report.)

---

## 🧠 Key Takeaways

- Small batches introduce **useful noise** that improves exploration.  
- Very large batches converge to **sharp minima** and may generalise worse.  
- Medium batch sizes (**64–256**) achieved the **best trade-off**.  
- Batch size should be chosen **strategically**, not arbitrarily.  

---

### 📚 References

- Keskar et al. (2017). *On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima.*  
- Goodfellow, Bengio, Courville (2016). *Deep Learning.*  
- Kaggle Credit Card Fraud Dataset.  
- Bottou (2012). *Stochastic Gradient Descent Tricks.*  

(All references included in report, page 10.)

---

### 📄 License

This project is released under the **MIT License**.

---

### 🙌 Acknowledgements

- Kaggle for the dataset  
- Researchers studying optimisation & batch-size effects  
- University module instructors  

---





