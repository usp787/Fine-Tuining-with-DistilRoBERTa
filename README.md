# Emotion Analysis via Parameter-Efficient Fine-Tuning (LoRA)

Authors: Jiarui Zha, Linxuan Li


# 📌 Project Overview

This project explores the effectiveness of Parameter-Efficient Fine-Tuning (PEFT) for multi-label emotion classification. While Large Language Models (LLMs) are capable of general-purpose tasks, they are often computationally expensive and overqualified for specific classification problems.

Our goal is to demonstrate that a small, distilled Encoder-only model (DistilRoBERTa), adapted via Low-Rank Adaptation (LoRA), can achieve competitive performance on complex emotion analysis tasks while training <1% of the total parameters.

# 🚀 Key Features

**Model Architecture**: DistilRoBERTa-base (Encoder-only).

**Fine-Tuning Method**: Low-Rank Adaptation (LoRA) to inject trainable rank-decomposition matrices into attention layers while freezing pretrained weights.

**Task**: Multi-label classification (27 emotion categories + Neutral).

**Loss Function**: BCEWithLogitsLoss for independent probability estimation per class.

# 📂 Dataset: GoEmotions

We utilize the GoEmotions dataset (Demszky et al., 2020), a corpus of 58k Reddit comments.

**Labels**: 27 emotion categories (e.g., Admiration, Remorse, Joy) + Neutral.

**Structure**: Multi-label (a single comment can express multiple emotions).

**Preprocessing**: Subset of ~10,000 samples for efficiency.

**Iterative Stratification**: Used to split data (Train 80% / Val 10% / Test 10%) while preserving the distribution of complex label combinations.

# 🛠 Methodology

1. Model Selection: Why DistilRoBERTa?

Following architectural best practices, we utilize DistilRoBERTa (an encoder) instead of decoder-based models (like DistilGPT-2).

**Bidirectional Attention**: Allows the model to access the full context of the comment simultaneously, which is crucial for understanding sentiment nuance.

**Classification Head**: We replace the standard pooling layer with a custom classification head outputting 28 logits.

2. Multi-Label Strategy

Since emotions are not mutually exclusive, we treat this as 28 independent binary classification problems.

**Activation**: Sigmoid (instead of Softmax). This allows high probabilities for multiple classes simultaneously.

**Threshold**: Probabilities $> 0.5$ are converted to positive predictions.

3. Evaluation Metrics

To account for class imbalance and the multi-label nature of the data, we report:

**Macro-F1 Score**: Averages the F1 score of each class independently. Treating rare classes (e.g., Grief) as equally important to common classes.

**Micro-F1 Score**: Aggregates contributions of all classes. Useful for assessing global accuracy.

📊 Baseline & Expected Results

We compare our LoRA-adapted model against:

**Zero-Shot Baseline**: Pretrained DistilRoBERTa without fine-tuning.

**Full Fine-Tuning (Benchmark)**: Updating all 82M parameters (computationally expensive).

**Target Metrics**:

**F1-Score**: 80-85%

**Parameter Efficiency**: Reduce trainable parameters by >98% compared to full fine-tuning.

# 🔧 Installation & Usage

# Clone the repository
git clone [https://github.com/usp787/DS_5110_Final_Project_LoRA.git](https://github.com/usp787/DS_5110_Final_Project_LoRA.git)


# Requirements:

torch

transformers

peft (for LoRA implementation)

scikit-learn (for iterative stratification and metrics)

datasets

# 📜 References

**Dataset**: Demszky, D., et al. (2020). GoEmotions: A Dataset of Fine-Grained Emotions.

**Method**: Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685v2.

**Base Model**: Sanh, V., et al. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.
