import numpy as np
import pandas as pd
from collections import Counter
import operator
import re
import warnings
import wikipedia
from sentence_transformers import SentenceTransformer, util

warnings.filterwarnings("ignore")

class ChatEngine:
    def __init__(self):
        print("Initializing Backend Chat Engine and BERT Model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.df_comb = pd.read_csv("./Dataset/dis_sym_dataset_comb.csv")
        self.df_norm = pd.read_csv("./Dataset/dis_sym_dataset_norm.csv")

        self.Y = self.df_norm.iloc[:, 0:1]
        self.X = self.df_norm.iloc[:, 1:]

        self.dataset_symptoms = list(self.X.columns)
        self.documentname_list = list(self.df_norm['label_dis'])
        self.diseases = list(set(self.Y['label_dis']))
        self.diseases.sort()

        self.N = len(self.df_norm)
        self.M = len(self.dataset_symptoms)

        # Pre-compute TF-IDF
        self.idf = {}
        for col in self.dataset_symptoms:
            temp = np.count_nonzero(self.df_norm[col])
            self.idf[col] = np.log(self.N / temp)

        self.tf = {}
        for i in range(self.N):
            for col in self.dataset_symptoms:
                key = (self.documentname_list[i], col)
                self.tf[key] = self.df_norm.loc[i, col]

        self.tf_idf = {}
        for i in range(self.N):
            for col in self.dataset_symptoms:
                key = (self.documentname_list[i], col)
                self.tf_idf[key] = float(self.idf[col]) * float(self.tf[key])

        self.D = np.zeros((self.N, self.M), dtype='float32')
        for i in self.tf_idf:
            sym = self.dataset_symptoms.index(i[1])
            dis = self.documentname_list.index(i[0])
            self.D[dis][sym] = self.tf_idf[i]

        dataset_symptoms_clean = [sym.replace('_', ' ') for sym in self.dataset_symptoms]
        self.db_embeddings = self.model.encode(dataset_symptoms_clean, convert_to_tensor=True)
        print("Engine Ready!")

    def parse_initial_message(self, user_input):
        user_symptoms_raw = [s.strip() for s in re.split(r'[,.!]|\band\b|\balso\b|\bwith\b|\bplus\b|\bi have\b', user_input.lower()) if len(s.strip()) > 3]
        if not user_symptoms_raw:
            return {"status": "error", "message": "I didn't catch any clear symptoms. Could you describe them again?"}

        user_embeddings = self.model.encode(user_symptoms_raw, convert_to_tensor=True)
        cosine_scores = util.cos_sim(user_embeddings, self.db_embeddings)

        found_symptoms = set()
        matched_text = []

        for i, raw_sym in enumerate(user_symptoms_raw):
            scores = cosine_scores[i]
            sorted_indices = scores.argsort(descending=True)
            for idx in sorted_indices[:2]:
                score = scores[idx].item()
                if score > 0.40:
                    matched_sym = self.dataset_symptoms[idx]
                    found_symptoms.add(matched_sym)
                    matched_text.append(matched_sym)

        final_symp = list(found_symptoms)
        if not final_symp:
            return {"status": "error", "message": "I couldn't perfectly match what you said to any known medical symptoms. Please try being more specific!"}

        # Calculate Co-occurring
        dis_list = set()
        counter_list = []
        for symp in final_symp:
            dis_list.update(set(self.df_norm[self.df_norm[symp]==1]['label_dis']))
        
        for dis in dis_list:
            row = self.df_norm.loc[self.df_norm['label_dis'] == dis].values.tolist()
            row[0].pop(0)
            for idx, val in enumerate(row[0]):
                if val != 0 and self.dataset_symptoms[idx] not in final_symp:
                    counter_list.append(self.dataset_symptoms[idx])

        dict_symp = dict(Counter(counter_list))
        dict_symp_tup = sorted(dict_symp.items(), key=operator.itemgetter(1), reverse=True)
        
        suggested_symptoms = [tup[0] for tup in dict_symp_tup[:15]]

        return {
            "status": "success",
            "matched": final_symp,
            "message": f"I understood you are experiencing: **{', '.join(final_symp).title()}**.<br>To help me give you a better diagnosis, do you also have any of these common related symptoms?",
            "suggestions": suggested_symptoms
        }

    def predict_diseases(self, final_symp):
        query_vector = self._gen_vector(final_symp)
        d_cosines = [self._cosine_dot(query_vector, d) for d in self.D]
        out = np.array(d_cosines).argsort()[-5:][::-1]
        
        top_diseases = []
        for idx in out:
            dis = self.diseases[idx]
            try:
                # Fetch a reliable, comprehensive 2-sentence summary directly from Wikipedia's API
                wiki_detail = wikipedia.summary(dis, sentences=2, auto_suggest=False)
            except Exception as e:
                wiki_detail = "No additional wikipedia details available."
            top_diseases.append({"name": dis, "score": float(d_cosines[idx]), "wiki": wiki_detail})
            
        return top_diseases

    def _gen_vector(self, tokens):
        Q = np.zeros(self.M)
        counter = Counter(tokens)
        for token in np.unique(tokens):
            tf_val = counter[token]
            idf_temp = self.idf.get(token, 0)
            try:
                ind = self.dataset_symptoms.index(token)
                Q[ind] = tf_val * idf_temp
            except ValueError: pass
        return Q

    def _cosine_dot(self, a, b):
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0 or b_norm == 0: return 0
        return np.dot(a, b) / (a_norm * b_norm)
