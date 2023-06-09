from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from init import app
import init
from flask import Flask
# app = Flask(__name__)


def matching(doc_vector, query_vector):
    # enc = OneHotEncoder(handle_unknown='ignore')
    # array_vec_1 = np.array(doc_vector, dtype=object)
    # array_vec_2 = np.array(query_vector, dtype=object)

    cosinesimilarities = cosine_similarity(doc_vector, query_vector).flatten()
    related_doc_id = cosinesimilarities.argsort()[:-10:-1]


# print(related_doc_id)
    return related_doc_id
# search=finalDic(token_dic)
# Search=list(search.values())
    # for i in related_doc_id :
    # #     fin = open("Corpus\\"+str(i+1)+".txt", "rt")
    # #     data=[fin.read()]
    #     data=[documents[i]]
    #     print(data)
