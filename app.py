from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import pickle
import sys
from flask import Flask, request
import tensorflow as tf
import flask

# loading the text dictionary file
with open('text_dictionary.pickle', 'rb') as temp_var:
    total_text_dictionary = pickle.load(temp_var)

# code reference from the elastic search documentation 
es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
if es.ping():
    print('Connected to ES!')
else:
    print('Could not connect!')
    sys.exit()

# loading the encoder model
embed = tf.keras.models.load_model('universal-sentence-encoder_4')
def make_vector(query):
    embeddings = embed([query])
    vector = []
    for i in embeddings[0]:
        vector.append(float(i))
    return vector

# connecting to elastic search

# definning a function to normalize the score values of the result.


def final_func1(query):
    def norm_list(lis):
        scores = [x[0] for x in lis]
        ma = max(scores)
        mi = min(scores)
        for i in range(len(lis)):
            lis[i][0] = (lis[i][0] - mi)/(ma - mi)
        return lis
    
    
    request={
            'query':{ 'match':{"total_texts":query } }
            }

    res= es.search(index='database',body=request)
    l1 = []
    for hit in res['hits']['hits']:
        l1.append([hit['_score'] , hit['_id']])
# change the cosine similarity to euclidean distance

    query_vector = make_vector(query)
    request = {"query" : {
                "script_score" : {
                    "query" : {
                        "match_all": {}
                    },
                    "script" : {
                        #"source": "cosineSimilarity(params.query_vector, 'total_vectors') + 1.0",
                        "source": "1 / (1 + l2norm(params.query_vector , 'total_vectors'))" ,
                        "params": {"query_vector": query_vector}
                    }
                }
             }
    }

    res= es.search(index='database',body=request)
    l2 = []
    for hit in res['hits']['hits']:
        l2.append([hit['_score'] , hit['_id']])
    
    l1 = norm_list(l1)
    l2 = norm_list(l2)
    
    # getting the weighted average score for the text search and semantics search
    temp_doc = {}
    for i in l1:
        temp_doc[i[1]]  = i[0]*2
    for i in l2:
        temp_doc[i[1]] = temp_doc.get(i[1] , 0) + i[0]*5
    
    inverse_temp_doc = [(i[1] , i[0])  for i in temp_doc.items()]
    inverse_temp_doc = sorted(inverse_temp_doc , reverse = True)
    return inverse_temp_doc[:10]



# getting the combined search results both semantic and the text based.
app = Flask(__name__)

@app.route('/')
def print_search_engine():
    return 'Search Engine'

@app.route('/index')
def index():
    return flask.render_template('index.html')

@app.route('/return_searches', methods=['POST'])
def return_searches():
    answer_no = 1
    to_return = ''
    for i in final_func1(request.form.to_dict()['query']):
        to_return += '-'*50 + "Answer No:" + str(answer_no) + '-'*50
        to_return += 2*'<br>'
        title = total_text_dictionary[i[1]][0]
        question = total_text_dictionary[i[1]][1]
        to_return+= "title : " + title + 2*'<br>'
        to_return += "question : " + question + 2*'<br>'
        sub_answer = 1
        for i in total_text_dictionary[i[1]][2:]:
            to_return += "subanswer " + str(sub_answer) +' : ' + i + 2*'<br>'
            sub_answer+=1
        answer_no+=1
    return to_return

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)