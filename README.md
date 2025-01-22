# Email Spam Classification

## 1. Descrizione
Questo progetto mira a costruire un modello di Machine Learning per classificare messaggi di testo in **Spam** o **Non-Spam**.

Il progetto esplora diversi modelli, come **Naive Bayes** e **Logistic Regression**, ottimizzando le performance tramite **GridSearchCV**.

---

## 2. Dataset Utilizzato
- **Nome**: [SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- **Descrizione**: Un dataset di 5572 messaggi etichettati come `spam` o `ham` (non-spam).
- **Formato**:
  - Colonna `label`: Etichetta del messaggio (`spam` o `ham`).
  - Colonna `message`: Testo del messaggio.

---

## 3. Obiettivo
Costruire un modello di classificazione testuale in grado di:
1. Preprocessare il testo per pulirlo e trasformarlo in numeri.
2. Addestrare modelli di Machine Learning su dati preprocessati.
3. Valutare e ottimizzare il modello migliore.

---

## 4. Struttura del Progetto
- **`data/`**:
  - `raw/`: Dati grezzi del dataset.
  - `processed/`: Dati trasformati e preprocessati, inclusi:
    - `tfidf_matrix.npz`: Matrice TF-IDF.
    - `labels.csv`: Etichette preprocessate.

- **`notebooks/`**:
  - `1_EDA.ipynb`: Analisi esplorativa dei dati.
  - `2_Preprocessing.ipynb`: Preprocessamento del testo.
  - `3_Modeling.ipynb`: Addestramento e valutazione dei modelli.

- **`src/`**:
  - `preprocessing.py`: Funzioni di preprocessamento (es. rimozione stopword, lemmatizzazione).
  - `model.py`: Codice per testare il modello salvato.

- **`models/`**:
  - `best_model.pkl`: Modello ottimizzato e salvato.

---

## 5. Modelli Testati
- **Multinomial Naive Bayes**:
  - **Accuracy**: 97.9%
- **Logistic Regression**:
  - **Accuracy**: 96.5%
- **Logistic Regression (Ottimizzata con GridSearchCV)**:
  - **Accuracy**: 98.3%
  - **Migliori Parametri**:
    - `C=10`
    - `solver='lbfgs'`

---

## 6. Risultati
- **Confusion Matrix (GridSearchCV)**:
  ``` plaintext
  [[964   2]
   [ 16 133]]

---

## 7. Osservazioni:
- La **Logistic Regression ottimizzata** ha prodotto le migliori performance.
- I falsi positivi e falsi negativi sono stati ridotti rispetto agli altri modelli.

---

## 8. Autore e Licenza

**Autore**: *Davide Quattrocchi*

**Licenza**: *Questo progetto è rilasciato sotto la licenza MIT.*
