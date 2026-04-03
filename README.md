# TrustCare AI Chatbot - Disease Detection System

## 🌐 What is this Website?
TrustCare AI is a modern, responsive chat-based web application that acts as an intelligent medical triage assistant. Users can chat with the AI in plain English about their symptoms. The AI will analyze the input, ask intelligent follow-up questions to narrow down the problem, and finally provide a prediction of the most likely medical condition, along with an explanation fetched directly from Wikipedia.

## 🧠 Which Model is it Working On?
The current live application is powered by a **hybrid Natural Language Processing (NLP) and Machine Learning approach**:

1. **The New NLP Model (BERT):** We are using `all-MiniLM-L6-v2` (a lightweight BERT variant via `sentence-transformers`). This allows the chatbot to understand *natural human language*.
2. **TF-IDF & Co-occurrence Matrix:** Once the symptoms are understood, the system uses a Term Frequency-Inverse Document Frequency (TF-IDF) scoring algorithm backed by a massive dataset matrix to calculate the probability of various diseases and dynamically suggest follow-up symptoms.

## 📥 Inputs & 📤 Outputs

**Inputs:**
*   **Initial Input:** A conversational, natural language sentence describing your symptoms (e.g., *"I have a really bad headache and I feel dizzy"*).
*   **Follow-up Input:** Easy-to-use checkbox selections of related symptoms suggested by the AI to refine your diagnosis.

**Outputs:**
*   **Symptom Mapping:** Mentally maps your plain English into exact clinical terms.
*   **Disease Prediction:** Calculates and returns the top possible diseases alongside a percentage confidence score.
*   **Medical Summary:** Automatically fetches a comprehensive 2-sentence medical description of your primary predicted disease directly from Wikipedia's API.

## 📂 File Structure Guide

*   `app.py`: The Flask web server. It handles the API routes (`/api/chat`) and serves the frontend user interface.
*   `chat_engine.py`: The "Brain" of the application. It loads the dataset, initializes the BERT model into memory, and processes all the heavy math (Cosine Similarity, TF-IDF calculation, and Final Predictions).
*   `/templates/` & `/static/`: Contains the Frontend HTML, CSS (featuring a modern Glassmorphism UI), and JavaScript logic for the interactive chat interface.
*   `requirements.txt`: The deployment dependencies needed for cloud hosting (e.g., Render, Railway).
*   `/Dataset/`: The raw CSV database files containing the complex symptom and disease relationship vectors.
*   `BERT_Symptom_Matcher.py` / `PreProcess_SymtomMatching.py`: Isolated logic scripts used to test and refine the new BERT natural language algorithms.

## 📊 Dataset Information
The application's logic is fueled by the curated CSV files located in the `/Dataset/` directory (`dis_sym_dataset_comb.csv` and `dis_sym_dataset_norm.csv`). This is a comprehensive medical multi-label dataset comprising **hundreds of symptom configurations accurately mapped to their corresponding diseases**. The ML Engine uses this relational dataset to learn the statistical co-occurrence of symptoms (e.g., how often *coughing* occurs alongside a *fever*) to formulate its probabilistic diagnosis matrix.

## 🚀 How to Run Locally

### 1. Clone the Repository
Open your terminal and clone the repository from GitHub:
```bash
git clone https://github.com/trustcare-ai/trustcare-ai-chatbot.git
cd trustcare-ai-chatbot
```

### 2. Install Requirements
Ensure you are inside the project folder, then install the necessary Python dependencies:
```bash
pip install -r requirements.txt
```

### 3. Start the Server
Once everything is installed, run the main routing app:
```bash
python app.py
```
*(Note: Start-up may take up to 10-15 seconds the first time, as the BERT model must load its extensive multi-million parameter weights natively into memory).*

### 4. Chat with the AI!
Open your preferred web browser and navigate directly to:
```text
http://127.0.0.1:5000
```

## 🕰️ Why are there Old Models in the repo, and why use the New One?

**The Legacy Models:**
You may notice files like `Model_latest.py`, `TF_IDF_NN.py`, `SymptomSuggestion.py`, and `Model_latest.ipynb`. These contain the original, legacy algorithms. They originally utilized traditional machine learning classifiers like Decision Trees, Random Forests, and K-Nearest Neighbors (KNN).

**The Problem with the Old Models:**
While the old models achieved good mathematical accuracy, they had a critical real-world flaw: they required **exact, hardcoded symptom inputs**. The user had to know the exact internal variable name (like `continuous_sneezing` instead of *"I am sneezing"*) for the system to work. It was essentially a strict database query, not an AI.

**Why we use the New Approach:**
By throwing out the strict classifiers and integrating the new `sentence-transformers` (BERT) model in our `chat_engine.py`, we created a powerful "language bridge". BERT understands the semantic *meaning* behind what the user is typing in everyday English. This completely eliminates the need for rigid, formatted inputs and allowed us to build a true, fluid Chatbot experience!

---

> *Disclaimer: TrustCare AI is a technology demonstration for Data Science and Machine Learning. Always seek the advice of a qualified healthcare provider with any questions you may have regarding a medical condition.*
