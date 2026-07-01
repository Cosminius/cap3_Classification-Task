# Chapter 3 – Classification

Exercises and code from Chapter 3 of *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* by Aurélien Géron.

---

## Notebooks

### `mnist_classification.ipynb`

Works through the MNIST handwritten digits dataset (60k training, 10k test samples from OpenML).

- Binary classification: digit 5 vs. rest using `SGDClassifier`
- Custom cross-validation with `StratifiedKFold`
- Precision/recall tradeoff analysis, ROC curve (AUC 0.96)
- Multi-class classification comparing SGD, RandomForest, and SVM
- Confusion matrix error analysis (e.g., where 3s get confused with 5s)
- Multi-label classification with `KNeighborsClassifier` and `ClassifierChain`
- Image denoising using KNN
- Data augmentation by shifting digits in all 4 directions → test accuracy goes from 97.14% to 97.66%

---

### `titanic_survivor.ipynb`

Binary classification to predict Titanic passenger survival (891 training samples, 418 test).

- EDA: missing value analysis, histograms, scatter matrix, title extraction from passenger names
- Preprocessing pipeline: median imputation + scaling for numerics, ordinal + one-hot encoding for categoricals
- `RandomForestClassifier` with 100 estimators
- Cross-validation accuracy: 81.27%, Precision: 0.737, Recall: 0.768
- Confusion matrix and ROC curve

---

### `spam_classifier.ipynb`

Email spam detection using the SpamAssassin public corpus (2500 ham, 500 spam emails).

- Downloads and parses raw emails with Python's `email` library
- Analyzes email structure across the dataset (text/plain, text/html, multipart variants)
- HTML-to-plain-text conversion, URL replacement, number normalization, punctuation stripping
- NLTK `PorterStemmer` for word stemming, `urlextract` for URL detection
- Two custom sklearn transformers: `EmailToWordCounterTransformer` and `WordCounterToVectorTransformer` (top-1000 vocab → sparse matrix)
- Full `Pipeline` combining both transformers + `LogisticRegression`
- Cross-validation accuracy: 99%, Test set Precision: 96.77%, Recall: 94.74%
