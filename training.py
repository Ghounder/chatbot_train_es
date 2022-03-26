from glob import glob
import stanza
import json
import random
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
stanza.download("es")
nlp = stanza.Pipeline(lang='es', processors='tokenize,mwt,pos,lemma')
words = []
classes = []
documents = []
aux = []
aux2 = []
data_file = open("database_uncased.json", encoding='utf-8').read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nlp(pattern)
        for sent in w.sentences: 
            aux.clear()
            for word in sent.words:
                words.append(word.lemma)
                aux.append(word.lemma)
            documents.append((aux, intent['tag']))
            print(aux)
            aux = ['']
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))
print(words)
print(classes)
print(documents)
print(len(documents), "documents")
print(len(classes), "classes")
print(len(words), "words")

pickle.dump(words, open('words.pkt', 'wb'))
pickle.dump(classes, open('classes.pkt', 'wb'))

training = []
output_empy = [0]*len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    print(pattern_words)

    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empy)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])
random.shuffle(training)
training = np.array(training)
train_x = list(training[:, 0])
train_y = list(training[:, 1])
print("creado la data del training")

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

hist = model.fit(np.array(train_x),
                 np.array(train_y),
                 epochs=300,
                 batch_size=50,
                 verbose=1)
model.save('model.h5', hist)

print("modelo created")
