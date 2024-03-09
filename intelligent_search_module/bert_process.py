#Import all the dependencies
# from selectors import EpollSelector
# import gensim
import pandas as pd
# from nltk import RegexpTokenizer
from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer,PorterStemmer
# from os import listdir
# from os.path import isfile, join
# from sklearn.neural_network import MLPRegressor
from scipy.spatial.distance import cosine
# import re
import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')
from sentence_transformers import SentenceTransformer, util
import numpy as np
# import pandas as pd
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

model = SentenceTransformer("intelligent_search_module/model/")

def build_context(df):
    '''
    create input context for bert model
    '''
    context = []
    for row in df.itertuples():
        # print(len())
    #     print(row.cleaned_text)
        context.append(row.cleaned_text)

    return context


def bert_search(corpus, sentence):
    search_results = []
    search_results_scrs = []
    # encode corpus to get corpus embeddings
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
    # sentence = "restrictions to emergency entrance"
    # encode sentence to get sentence embeddings
    sentence_embedding = model.encode(sentence, convert_to_tensor=True)
    # top_k results to return
    top_k=20
    score_thresh = 0.50
    # compute similarity scores of the sentence with the corpus
    cos_scores = util.pytorch_cos_sim(sentence_embedding, corpus_embeddings)[0]
    # print(cos_scores)
    # Sort the results in decreasing order and get the first top_k
    top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]

    for idx in top_results[0:top_k]:

        # print(corpus[idx], " --- (Score: %.4f)" % (cos_scores[idx]))
        if cos_scores[idx] > score_thresh:
            search_results.append(corpus[idx])
            search_results_scrs.append(round(cos_scores[idx].item(),2)*100)
            # print(cos_scores[idx].item())       

    return search_results, search_results_scrs


def output_df(search_lst, search_score_lst,df):
    df_searched = pd.DataFrame()
    if len(search_lst) > 0:
        for i, string in enumerate(search_lst):
            df_sub = df.loc[df['cleaned_text']==string]
            df_sub['search_score'] = search_score_lst[i]
            df_searched = pd.concat([df_searched,df_sub],  ignore_index=True)

        return df_searched
    else:
        return None


# if __name__ == '__main__':
#     main()