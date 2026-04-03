import numpy as np
import pandas as pd
from collections import Counter
import operator
import sys
from Treatment import diseaseDetail

# Try importing sentence_transformers, handle if not installed yet
try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    print("\n[!] Please install sentence-transformers using: py -m pip install sentence-transformers torch")
    sys.exit(1)

print("Initializing BERT Model (this generally takes a few seconds)...")
# Initialize BERT semantic model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load Datasets
df_comb = pd.read_csv("./Dataset/dis_sym_dataset_comb.csv")
df_norm = pd.read_csv("./Dataset/dis_sym_dataset_norm.csv")

Y = df_norm.iloc[:, 0:1]
X = df_norm.iloc[:, 1:]

dataset_symptoms = list(X.columns)
documentname_list = list(df_norm['label_dis'])
diseases = list(set(Y['label_dis']))
diseases.sort()

N = len(df_norm)
M = len(dataset_symptoms)

# Calculate TF-IDF (retained from original for final prediction)
idf = {}
for col in dataset_symptoms:
  temp = np.count_nonzero(df_norm[col])
  idf[col] = np.log(N/temp)

tf = {}
for i in range(N):
  for col in dataset_symptoms:
    key = (documentname_list[i], col)
    tf[key] = df_norm.loc[i, col]

tf_idf = {}
for i in range(N):
  for col in dataset_symptoms:
    key = (documentname_list[i], col)
    tf_idf[key] = float(idf[col]) * float(tf[key])

D = np.zeros((N, M), dtype='float32')
for i in tf_idf:
    sym = dataset_symptoms.index(i[1])
    dis = documentname_list.index(i[0])
    D[dis][sym] = tf_idf[i]

def cosine_dot(a, b):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0: return 0
    return np.dot(a, b) / (a_norm * b_norm)

def gen_vector(tokens):
    Q = np.zeros(M)
    counter = Counter(tokens)
    for token in np.unique(tokens):
        tf_val = counter[token]
        idf_temp = idf.get(token, 0)
        try:
            ind = dataset_symptoms.index(token)
            Q[ind] = tf_val * idf_temp
        except ValueError: pass
    return Q

def tf_idf_score(k, query):
    query_weights = {}
    for key in tf_idf:
        if key[1] in query:
            query_weights[key[0]] = query_weights.get(key[0], 0) + tf_idf[key]
    return sorted(query_weights.items(), key=lambda x: x[1], reverse=True)[:k]

def cosine_similarity_score(k, query):
    query_vector = gen_vector(query)
    d_cosines = [cosine_dot(query_vector, d) for d in D]
    out = np.array(d_cosines).argsort()[-k:][::-1]
    return {lt: float(d_cosines[lt]) for lt in set(out)}

# Generate BERT embeddings for all dataset symptoms (done just once at startup)
print("Encoding dataset symptoms database...")
dataset_symptoms_clean = [sym.replace('_', ' ') for sym in dataset_symptoms]
db_embeddings = model.encode(dataset_symptoms_clean, convert_to_tensor=True)

import re

# -----------------
# USER INTERACTION
# -----------------
print("\n" + "="*50)
user_input = input("Please describe your symptoms in a natural sentence:\n").lower()
# Split by commas and basic conjunctions to chunk symptoms
user_symptoms_raw = [s.strip() for s in re.split(r'[,.!]|\band\b|\balso\b|\bwith\b|\bplus\b', user_input) if len(s.strip()) > 3]

if not user_symptoms_raw:
    print("\nNo valid symptoms found in input. Exiting.")
    sys.exit()

print("\nRunning Semantic BERT search for your symptoms...")
# Encode user symptoms
user_embeddings = model.encode(user_symptoms_raw, convert_to_tensor=True)

# Find top matching database symptoms using PyTorch / SentenceTransformers cosine similarity
cosine_scores = util.cos_sim(user_embeddings, db_embeddings)

found_symptoms = set()
print("\n[BERT Semantic Matches]")
for i, raw_sym in enumerate(user_symptoms_raw):
    # Simple tensor argsort
    scores = cosine_scores[i]
    sorted_indices = scores.argsort(descending=True)
    
    print(f"\nFor your input '{raw_sym}':")
    for idx in sorted_indices[:2]:
        score = scores[idx].item()
        if score > 0.40:  # Good semantic confidence threshold
            matched_sym = dataset_symptoms[idx]
            print(f" -> Auto-Matched to database symptom: '{matched_sym}' (Confidence: {int(score*100)}%)")
            found_symptoms.add(matched_sym)
        elif score > 0.30:
            print(f" -> Weak match: '{dataset_symptoms[idx]}' (Confidence: {int(score*100)}%) - Ignoring")

final_symp = list(found_symptoms)

if not final_symp:
    print("\nSorry, BERT couldn't confidently match your input to any known symptoms. Please be more specific.")
    sys.exit()

# --- Co-Occurring Symptom Discovery Loop ---
dis_list = set()
counter_list = []

# Find all diseases that share these initial matched symptoms
for symp in final_symp:
    dis_list.update(set(df_norm[df_norm[symp]==1]['label_dis']))
   
for dis in dis_list:
    row = df_norm.loc[df_norm['label_dis'] == dis].values.tolist()
    row[0].pop(0)
    for idx,val in enumerate(row[0]):
        if val!=0 and dataset_symptoms[idx] not in final_symp:
            counter_list.append(dataset_symptoms[idx])

# Count how frequently other symptoms occur with the matched items
dict_symp = dict(Counter(counter_list))
dict_symp_tup = sorted(dict_symp.items(), key=operator.itemgetter(1),reverse=True)

# Iteratively suggest top co-occurring symptoms to the user
suggested_symptoms = []
count = 0
for tup in dict_symp_tup:
    count += 1
    suggested_symptoms.append(tup[0])
    if count % 5 == 0 or count == len(dict_symp_tup):
        print("\nCommon co-occuring symptoms:")
        for idx, ele in enumerate(suggested_symptoms):
            print(idx, ":", ele)
        select_list = input("Do you have any of these symptoms? Enter the indices (space-separated), 'no' to stop, '-1' to skip:\n").lower().split()
        if not select_list or select_list[0] == 'no':
            break
        if select_list[0] == '-1':
            suggested_symptoms = [] 
            continue
        for idx_str in select_list:
            if idx_str.isdigit() and int(idx_str) < len(suggested_symptoms):
                final_symp.append(suggested_symptoms[int(idx_str)])
        suggested_symptoms = []

k = 10
print("\n------------------------------")
print("Final list of Symptoms used for prediction:")
for val in final_symp: print("-", val)
print("------------------------------")

topk1 = tf_idf_score(k, final_symp)
topk2 = cosine_similarity_score(k, final_symp)

print(f"\nTop {k} diseases predicted based on TF_IDF Matching :\n")
topk1_index_mapping = {}
for i, (key, score) in enumerate(topk1):
  print(f"{i}. Disease: {key} \t Score: {round(score, 2)}")
  topk1_index_mapping[i] = key

select = input("\nMore details about the disease? Enter index of disease or '-1' to discontinue:\n")
if select != '-1':
    if select.isdigit() and int(select) in topk1_index_mapping:
        dis = topk1_index_mapping[int(select)]
        print()
        print(diseaseDetail(dis))

print(f"\nTop {k} disease based on Cosine Similarity Matching :\n ")
topk2_sorted = dict(sorted(topk2.items(), key=lambda kv: kv[1], reverse=True))
topk2_index_mapping = {}
for j, key in enumerate(topk2_sorted):
  print(f"{j}. Disease: {diseases[key]} \t Score: {round(topk2_sorted[key], 2)}")
  topk2_index_mapping[j] = diseases[key]

select = input("\nMore details about the disease? Enter index of disease or '-1' to discontinue and close the system:\n")
if select != '-1':
    if select.isdigit() and int(select) in topk2_index_mapping:
        dis = topk2_index_mapping[int(select)]
        print()
        print(diseaseDetail(dis))
