# Transformer-Based Scientific Text Classification

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow?style=for-the-badge)](https://huggingface.co/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)

## 📌 Overview
This project benchmarks the performance of sequential modeling versus transfer learning for the **hierarchical classification of scientific abstracts**.

Using the **[Web of Science (WOS-11967)](https://data.mendeley.com/datasets/9rw3vkcfy4/2)** dataset, the system classifies 11,967 abstracts into 7 parent domains (Level 1) and 34 fine-grained sub-fields (Level 2).

The project compares two distinct architectures:
1.  **Baseline:** A custom **LSTM** built from scratch (implementing manual gate logic) using Pre-trained GloVe Embeddings.
2.  **SOTA:** A fine-tuned **BERT** Transformer (`bert-base-uncased`) using Self-Attention mechanisms.

## 📊 Key Results
The fine-tuned BERT model consistently outperformed the sequential LSTM baseline, achieving a **12.6% accuracy improvement** on the complex sub-field task. This validates the effectiveness of the Transformer architecture in capturing global semantic context compared to the sequential limitations of RNNs.

| Task | Classes | LSTM Accuracy (Baseline) | BERT Accuracy (Fine-Tuned) | Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Level 1 (Parent Domains)** | 7 | 88.60% | **93.61%** | +5.0% |
| **Level 2 (Sub-Fields)** | 34 | 74.98% | **87.55%** | **+12.6%** |

## 🧠 Model Architectures

### 1. Sequential Baseline: Custom LSTM
Instead of using the standard high-level `nn.LSTM` API, this implementation features a **custom `LSTMCell` class built from scratch** to demonstrate a deep understanding of RNN mechanics.
* **Manual Gate Implementation:** Explicitly defined linear layers and forward-pass logic for the **Input ($i_t$)**, **Forget ($f_t$)**, **Cell ($g_t$)**, and **Output ($o_t$)** gates.
* **Embeddings:** Utilizes 300-dimensional pre-trained **[GloVe Vectors](https://nlp.stanford.edu/projects/glove/)**.
* **Optimization:** Standard Adam optimizer with CrossEntropyLoss.
* **Challenges:** Aggressive stop-word filtering initially caused gradient divergence; resolved by reverting to a Regex-based tokenizer to preserve syntactic structure.

### 2. Transformer: Fine-Tuned BERT
A `bert-base-uncased` model fine-tuned for sequence classification using the Hugging Face Transformers library.
* **Architecture:** 12 Transformer layers, 12 Attention heads, 768 hidden units.
* **Optimization:** **AdamW** optimizer (handling weight decay) with **Early Stopping** to prevent overfitting.
* **Input Handling:** Stratified sampling to handle class imbalance; max sequence length set to **256 tokens** (covering 83% of the dataset) to optimize memory usage.

## 🔍 Interpretability & Analysis
To ensure the BERT model was learning meaningful features rather than exploiting noise, I developed an interpretability module.

### Self-Attention Visualization
By extracting attention tensors from **Layer 12, Head 1** (the final semantic block), I visualized how the `[CLS]` token attends to the input sequence (visualizations are generated directly in the notebook).
* **Finding:** The model places high attention weights on domain-specific keywords (e.g., *"genetic"*, *"locus"*) while filtering out functional stop words.

### Error Analysis (Confusion Matrix)
A confusion matrix analysis of the Level 2 predictions revealed that misclassifications were not random.
* **Cluster Analysis:** Errors primarily occurred between semantically adjacent fields, such as **Medical Science vs. Biochemistry**.
* **Outlier Detection:** Identified that "outlier" tokens in short abstracts occasionally misled the attention mechanism.

## 🛠️ Installation & Usage

### Prerequisites
* Python 3.8+
* PyTorch (CUDA recommended for training)
* Jupyter Notebook / Google Colab

### Setup
1.  **Clone the repository**
    ```bash
    git clone https://github.com/SamySabir/scientific-text-classification.git
    cd scientific-text-classification
    ```

2.  **Install dependencies**
    ```bash
    pip install torch torchvision transformers pandas numpy seaborn scikit-learn matplotlib
    ```

3.  **Download Data**
    * Download the **WOS-11967** dataset from [Mendeley Data](https://data.mendeley.com/datasets/9rw3vkcfy4/2).
    * Place the files (`X.txt`, `YL1.txt`, `YL2.txt`) in the `WOS11967/` directory.
    * Download `glove.6B.300d.txt` from the [Stanford NLP website](https://nlp.stanford.edu/projects/glove/) and place it in the root directory.

### Running the Pipeline
Open the notebook `training_pipeline.ipynb` to run the full training loop. The notebook is structured as follows:
1.  **Data Preprocessing:** Label mapping and EDA.
2.  **LSTM Implementation:** Definition of the custom `LSTMCell`.
3.  **Training Loops:** Training both models with validation monitoring.
4.  **Evaluation:** Generating accuracy metrics and visualizations.

## 👨‍💻 Credits
* **Samy Sabir** - Hyperparameter tuning (AdamW), Interpretability (Attention Maps), Confusion Matrix analysis, and Report writing.
* **Vladimir Venkov** - Core Model Development (BERT & LSTM architectures).
* **Alan Fu** - GPU resource management and Training Time analysis.
