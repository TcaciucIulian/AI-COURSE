import numpy as np
from sklearn import preprocessing
from sklearn import svm
from sklearn.metrics import f1_score, classification_report
import pdb
import re


train_data = np.genfromtxt('train_samples.txt', encoding='utf-8', dtype=None , delimiter='\t', names=('id','samples'),comments=None)
train_labels = np.genfromtxt('train_labels.txt', encoding='utf-8', dtype=None , delimiter='\t', names=('id','label'))
test_data = np.genfromtxt('test_samples.txt', encoding='utf-8', dtype=None , delimiter='\t', names=('id','samples'),comments=None)
val_label = np.genfromtxt('validation_labels.txt', encoding='utf-8', dtype=None , delimiter='\t', names=('id','label'))

class Bag_of_words:

    def __init__(self):
        self.words = []
        self.vocabulary_length = 0

    def build_vocabulary(self, data):
        for word in data:
            res = word.split()
            for cuv in res:
                if cuv not in self.words:
                    self.words.append(cuv)

        self.vocabulary_length = len(self.words)
        self.words = np.array(self.words)

    def get_features(self, data):
        features = np.zeros((len(data), self.vocabulary_length), dtype='uint8')
        for document_idx, document in enumerate(data):
            res = document.split()
            for word in res:
                if word in self.words:
                    features[document_idx, np.where(self.words == word)[0][0]] += 1
        return features


def normalize_data(training_data, testing_data, type=None):
    scaler = None
    if type == 'standard':
        scaler = preprocessing.StandardScaler()

    elif type == 'min_max':
        scaler = preprocessing.MinMaxScaler()

    elif type == 'l1':
        scaler = preprocessing.Normalizer(norm='l1')

    elif type == 'l2':
        scaler = preprocessing.Normalizer(norm='l2')

    if scaler is not None:
        scaler.fit(training_data)
        scaled_train_data = scaler.transform(training_data)
        scaled_test_data = scaler.transform(testing_data)
        return (scaled_train_data, scaled_test_data)
    else:
        print("No scaling was performed. Raw data is returned.")
        return (training_data, testing_data)


y = np.array(train_labels['label'][:2656])
y1= np.array(val_label['label'])
plt.plot(y, y1)
plt.show()

bow_model = Bag_of_words()
bow_model.build_vocabulary(train_data['samples'])
print(len(bow_model.words))

print(len(train_data))
print(len(train_labels))
print(len(test_data))
print(len(val_label))

inceput = 0
sfarsit = 500
index = 0
predicted_labels_svm = np.zeros(2623)
while ( sfarsit < 2623 ):
    index += 1
    print('Predictia numarul', index)
    train_features = bow_model.get_features(train_data['samples'][inceput:sfarsit])
    test_features = bow_model.get_features(test_data['samples'][inceput:sfarsit])
    print(train_features.shape)
    print(test_features.shape)
    scaled_train_data, scaled_test_data = normalize_data(train_features, test_features, type='l2')
    print(scaled_train_data.shape)
    print(scaled_test_data.shape)
    svm_model = svm.SVC(C=100, kernel='linear')
    svm_model.fit(scaled_train_data, train_labels['label'][inceput:sfarsit])
    predicted_labels_svm[inceput:sfarsit] = svm_model.predict(scaled_test_data)
    print(predicted_labels_svm[inceput:sfarsit])
    inceput = sfarsit
    sfarsit = sfarsit + 500
    print(len(predicted_labels_svm))



train_features = bow_model.get_features(train_data['samples'][2600:2623])
test_features = bow_model.get_features(test_data['samples'][2600:2623])
print(train_features.shape)
print(test_features.shape)
scaled_train_data, scaled_test_data = normalize_data(train_features, test_features, type='l2')
print(scaled_train_data.shape)
print(scaled_test_data.shape)
svm_model = svm.SVC(C=2**50, kernel='linear')
svm_model.fit(scaled_train_data, train_labels['label'][2600:2623])
print(len(svm_model.predict(scaled_test_data)))
predicted_labels_svm[2600:2623] = svm_model.predict(scaled_test_data)


print(compute_accuracy(np.asarray(val_label['label'][:2623]), predicted_labels_svm))
print('f1 score', f1_score(np.asarray(val_label['label'][:2623]), predicted_labels_svm))

np.savetxt("test_labels.csv", np.column_stack((test_data['id'], predicted_labels_svm)), delimiter=",", fmt='%s')