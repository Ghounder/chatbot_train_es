import random
import pickle
import json
import stanza
import numpy as np
from tensorflow.keras.models import load_model
nlp = stanza.Pipeline(lang='es', processors='tokenize,mwt,pos,lemma')
model = load_model("model.h5")
intents = json.loads(open("database_uncased.json", encoding='utf-8').read())
words = pickle.load(open("words.pkt", "rb"))
classes = pickle.load(open("classes.pkt", "rb"))


def clean_up_sentence(sentence):
    aux = []
    sentence_words = nlp(sentence)
    for sent in sentence_words.sentences:
        for word in sent.words:
            aux.append(word.lemma)
    sentence_words = aux
    return sentence_words


def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    print(sentence_words)
    bag = [0]*len(words) 
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))


def predict_class(sentence, model):
    p = bow(sentence,words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.7
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent":classes[r[0]],"probability": str(r[1])})
    print(return_list)
    return return_list


def get_response(ints, intents_json):
    try:
        tag = ints[0]["intent"]
        print("tag")
        print(tag)
        list_of_intents=intents_json["intents"]
        for i in list_of_intents:
            if(i["tag"] == tag):
                result = random.choice(i["responses"])
                break
    except:
        result = "no puedo entenderte"
    return result
def chatbot_response(msg):
    ints=predict_class(msg,model)
    print(f"ints = {ints}")
    res=get_response(ints,intents)
    return print(res)
chatbot_response("cuantos años tienes?")
