import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def build_matrix(user_docs, query):
    words = set()
    for doc in user_docs + [query]:
        words.update(doc.lower().replace(".", "").split())
    words = sorted(words)
    
    term_doc = np.zeros((len(words), len(user_docs)))
    for j, doc in enumerate(user_docs):
        for word in doc.lower().replace(".", "").split():
            if word in words:
                term_doc[words.index(word), j] = 1
    return term_doc, words

n = int(input())
documents = [input().strip() for _ in range(n)] 
query = input().strip() 
k = int(input()) 
matrix, vocab = build_matrix(documents, query)

u, s, vt = np.linalg.svd(matrix, full_matrices=False)
u_k = u[:, :k]
s_k = np.diag(s[:k])
vt_k = vt[:k, :]

doc_reduced = np.dot(s_k, vt_k)
query_vec = np.zeros(len(vocab))

for word in query.lower().split():
    if word in vocab:
        query_vec[vocab.index(word)] = 1

query_reduced = np.dot(np.linalg.inv(s_k), np.dot(u_k.T, query_vec))

similarities = cosine_similarity(query_reduced.reshape(1, -1), doc_reduced.T).flatten()
similarities = np.round(similarities, 2).tolist()
print(similarities)