# tbp-skin-detect
Skin Cancer Detection from 3D Total Body Photos

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Evaluator - Partial area under the ROC curve](https://img.shields.io/badge/Optimized--for-pAUC%20Score-yellowgreen?style=for-the-badge)
![Metric Score](https://img.shields.io/badge/Best%20Score-0.16930-2ECC71?style=for-the-badge)
![Rank](https://img.shields.io/badge/Rank-184%20of%202739-brightgreen?style=for-the-badge)
![Solo](https://img.shields.io/badge/Submission-Solo-orange?style=for-the-badge)
![Optuna](https://img.shields.io/badge/Optuna-Tuning-blueviolet?style=for-the-badge)

### **Project Duration**: Aug 15, 2024 - Sep 7, 2024

---

## 🧠 Objective

TThe goal was to build binary classifiers to predict malignant skin lesions from single-lesion crops extracted from 3D total body photos (TBP). This project is part of the `ISIC 2024 - Skin Cancer Detection from 3D-TBP` Kaggle competition. Submissions were evaluated on **partial AUC (pAUC)** for true positive rates (TPR) above **80%**.

---

## 🧩 Approach

You can explore the complete methodology in this notebook: 🔗 [ISIC24 - Heavy Feature Eng with Polars + Boosting + CV + Ensemble Blend](https://github.com/krishnaura45/tbp-skin-detect/blob/main/isic24-feature-boost-ensemble.ipynb)

Key steps followed:

- ✏️ **Feature Engineering**: *spatial insights*
  - Extracted 3D landmark distances using pairwise Euclidean computations across TBP points.
  - Constructed derived geometric features reflecting anatomical symmetry and patient-level spatial variation.

- ⚖️ **Patient-Level Normalization**: *consistency modeling*
  - Applied normalization of features at the patient level to control for inter-subject variability.
  - Included feature columns for image count per patient.

- 📊 **Categorical Handling**: *info retention*
  - Employed OneHotEncoder for categorical variables.
  - Converted them to category dtype for memory efficiency.

- 🧰 **Ensemble Learning**: *reducing model variance*
  - Trained LightGBM, XGBoost, and CatBoost models independently.
  - Combined predictions using a weighted ensemble method for improved pAUC.

- 📊 **Custom Evaluation Metric**: *tailored pAUC*
  - Implemented a custom scoring function to simulate competition-specific pAUC above 80% TPR.
  - This metric guided model selection and cross-validation.

---

## 🏆 Results / Outcomes

![isic2024-leaderboard-scores](https://github.com/user-attachments/assets/b49ed419-cd72-4881-902c-8dd8f6453758)

- ✅ Public Leaderboard Scores:
  - 0.18368, 0.18412, 0.18519

- 🏁 Private Leaderboard Scores:
  - 0.16733, 0.16753, **0.16930** (final best)

- 🥇 Rank Achieved:
  - Placed `184th` out of **3410 participants** and **2739 teams** as a **solo competitor**

---

## 🔗 References

- 🏆 Kaggle Competition: [ISIC 2024 - Skin Cancer Detection with 3D-TBP](https://www.kaggle.com/competitions/isic-2024-challenge)

---

## 🛠️ Tech Stack

- **Language**: Python 🐍
- **Libraries**:
  - `polars` for dataframe operations
  - `pandas`, `numpy` for numerical tasks
  - `stratifiedKFold` for cross validation
  - `sklearn`, `lightgbm`, `xgboost`, `catboost` for modeling
  - `matplotlib`, `seaborn` for visualization
  - `optuna` for hyperparameter tuning
- **Tools**:
  - Jupyter Notebook / Kaggle Notebooks 📓 for experimentation and code
  - Custom Python metric functions for pAUC evaluation

