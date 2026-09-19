# 🤖 AI Hallucination Risk Predictor

An end-to-end Machine Learning web application designed to detect and measure hallucination risks in AI-generated answers by comparing them with verified references and retrieved context.

🔗 **Live Demo:** [AI Hallucination Risk Predictor App](https://ai-hallucination-risk-predictor-pgr6mp3vn7u4aattpqmal6.streamlit.app/)

---

## ✨ Features
- **Semantic Similarity Analysis:** Uses `sentence-transformers` to compute deep textual semantic alignment.
- **Entity Matching:** Checks for factual discrepancies and entity mismatches between answers.
- **Machine Learning Risk Scoring:** Predicts risk percentage using a trained custom model (`hallucination_model.pkl`).
- **Claim Verification:** Verifies claims against reference context with real-time feedback.

---

## 🛠️ Tech Stack
- **Frontend / Deployment:** Streamlit, Streamlit Community Cloud
- **Machine Learning & NLP:** PyTorch, `sentence-transformers`, `scikit-learn`
- **Data Manipulation:** `pandas`, `numpy`

---

## 🚀 How to Run Locally

1. Clone the repository:
   ```bash
   git clone [https://github.com/Vikasraj-vrg/AI-Hallucination-Risk-Predictor.git](https://github.com/Vikasraj-vrg/AI-Hallucination-Risk-Predictor.git)
   cd AI-Hallucination-Risk-Predictor
