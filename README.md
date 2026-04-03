# TrustCare AI - Disease Detection System

TrustCare AI is an advanced, interactive diagnostic assistant that leverages Machine Learning and Natural Language Processing to predict health conditions based on symptoms. 

Originally built as a basic console application, the project has been fully re-architected into a **modern, web-based AI assistant** complete with dynamic language bridging via **BERT** and a gorgeous, responsive Glassmorphic user interface.

## 🌟 Key Capabilities
1. **Natural Language Processing (BERT):** You don't need to conform to strict, hard-coded clinical terminology! Using `sentence-transformers` (`all-MiniLM-L6-v2`), TrustCare seamlessly parses conversational phrases (e.g. *"I have a really bad headache"*) and automatically maps them to formal medical dataset classifications with high accuracy.
2. **Interactive Triage:** Just like a real doctor, the AI utilizes a co-occurrence matrix algorithm. Once it catches your initial symptoms, it searches its database for diseases sharing those symptoms and prompts you with highly relevant follow-up questions (e.g. *"Since you have a fever, do you also have nausea?*").
3. **Advanced Prediction Algorithms:** It leverages massive datasets mathematically using specialized Search-Engine indexing techniques like **TF-IDF Calculation** and **Cosine-Similarity**. Look at the `TF_IDF_NN.py` for the algorithmic backbone!
4. **Automated Dictionary Lookup:** Out of the box integration with the `wikipedia` Python package to extract and deliver 2-sentence medical explanations identifying your top forecasted disease natively in the chat!

## 💻 Tech Stack
*   **Backend / Server:** Python, Flask server (`app.py`), `chat_engine.py`
*   **Machine Learning:** PyTorch, `sentence-transformers`, `scikit-learn`, `pandas`, `numpy`, `nltk`.
*   **Frontend / UI:** Next-generation Vanilla HTML/CSS with JavaScript interactivity, styled entirely with frosted-glass Glassmorphism metrics for maximum aesthetic quality without heavy external libraries.

## 🚀 How to Run Locally

### 1. Install Dependencies
You will need to install the required Python libraries. Open your terminal in the project directory and run:
```bash
py -m pip install flask sentence-transformers torch pandas numpy wikipedia nltk scikit-learn
```

### 2. Boot the Server
Start the Flask web-server bridging the AI architecture with the frontend interface:
```bash
py app.py
```
*(Note: It may take 5–15 seconds to finish spinning up while the PyTorch Sentence-Transformers backend loads the multi-million parameter weights natively into memory).*

### 3. Open the App
Once you see `* Running on http://127.0.0.1:5000` in the terminal, open your favorite web browser and navigate directly to the application:
```text
http://localhost:5000
```

## 🧠 File Structure Guide
*   `app.py`: Standard Flask web server managing endpoints.
*   `chat_engine.py`: The core state-holding interface class containing our Machine Learning integration.
*   `/templates` & `/static`: The Frontend chat client implementation.
*   `Model_latest.py`: The original algorithm comparisons (Decision Trees, K-Nearest Neighbors achieving ~90% accuracy).
*   `/Dataset`: Cleaned CSV references of symptom/disease vectors. 

> *Disclaimer: TrustCare AI is a technology demonstration for Data Science and Machine Learning. Always seek the advice of a qualified healthcare provider with any questions you may have regarding a medical condition.*
