import re
import pickle
from nltk.stem import WordNetLemmatizer

with open('text_classifier', 'rb') as training_model:
    model = pickle.load(training_model)

documents = []

with open("/Users/sskmal/PycharmProjects/NlpSentimentAnalysis/review_polarity/txt_sentoken/neg/cv000_29416.txt") as file:
    a = file.readlines()

with open("/Users/sskmal/PycharmProjects/NlpSentimentAnalysis/review_polarity/txt_sentoken/pos/cv012_29576.txt") as file:
    b = file.readlines()

print(a)
X = [' '.join(a), ' '.join(b)]

stemmer = WordNetLemmatizer()

for sen in range(0, len(X)):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(X[sen]))

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)

    # Converting to Lowercase
    document = document.lower()

    # Lemmatization
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)

    documents.append(document)

with open('text_vectorizer', 'rb') as vectorizer:
    Vectormodel = pickle.load(vectorizer)

with open('text_tfidfconverter', 'rb') as TFIDConvertor:
    TFIDConvertormodel = pickle.load(TFIDConvertor)

vector = Vectormodel.transform(documents).toarray()
finalvector = TFIDConvertormodel.transform(vector).toarray()

y_pred2 = model.predict(finalvector)

for i in range(len(y_pred2)):
    x = 'negative' if y_pred2[i] == 0 else 'positive'
    print("______________________")
    print(str(X[i][:100]) + "[ .... is a " + x + " comment ]")
