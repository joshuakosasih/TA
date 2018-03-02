import nltk


def predict(sentence, MAX_SEQUENCE_LENGTH):
    sent = nltk.word_tokenize(sentence)  # tokenize
    se = []
    for it in range(len(sent), MAX_SEQUENCE_LENGTH):  # padding
        se.append(0)

    for token in sent:  # indexing
        se.append(word_index[token])

    se = np.array([se])  # change to np array

    result = model.predict(se)[0]  # get prediction result
    res = []
    for token in result:
        value = np.argmax(token)
        if value == 0:
            res.append('~')
        else:
            key = labels_index.keys()[labels_index.values().index(value)]
            res.append(key)

    return res
    print res