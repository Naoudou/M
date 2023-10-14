# M
KFOLD



from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression



iris = load_iris()

x = iris.data
y = iris.target
logreg = LogisticRegression(solver='lbfgs',class_weight='balanced', max_iter=10000)
kf = KFold(n_splits=5)
score = cross_val_score(logreg,x,y,cv=kf)
print("Cross validation score are {}".format(score))
print("Average Cross Validation score :{}".format(score.mean()))


STACK MODEL


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


#loading the data
iris = pd.read_csv("iris.csv", encoding="unicode_escape")
#storing feature matrix in x
x=iris.iloc[:,0:4].values
y=iris.iloc[:,4].values

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=0)



# KNN
from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(x_train,y_train)

#predicting

y_pred = knn_clf.predict(x_test)


knn_accuracy = accuracy_score(y_test, y_pred)

print("KNN gives"+ str(knn_accuracy*100)+"% accuracy")

#GAUSSIAN NAIVE BAYES
from sklearn.naive_bayes import GaussianNB

NB_clf = GaussianNB()
NB_clf.fit(x_train,y_train)

#predicting

y_pred = NB_clf.predict(x_test)

#testing accuracy

NB_accuracy = accuracy_score(y_test, y_pred)

print("Naive bayes gives :"+ str( NB_accuracy*100)+ "% Accuracy")

# DECISION TREE
from sklearn import tree

dt_clf = tree.DecisionTreeClassifier(criterion="gini",min_samples_split=30,splitter="best")

dt_clf.fit(x_train, y_train)


#predicting

y_pred = dt_clf.predict(x_test)

#testing accuracy

dt_accuracy = accuracy_score(y_test, y_pred)

print("Decision tree gives :"+ str( dt_accuracy*100)+ "% Accuracy")


## Stacking

from sklearn.ensemble import StackingClassifier 
from sklearn.linear_model import LogisticRegression

estimators = [('knn',knn_clf),
             ('NB',NB_clf),
             ('dt',dt_clf)]

stack_model = StackingClassifier(estimators=estimators,final_estimator=LogisticRegression())
stack_model.fit(x_train, y_train)
y_pred = stack_model.predict(x_test)

#testing accuracy

stack_model_accuracy = accuracy_score(y_test, y_pred)
print("Stack Model gives :"+ str( stack_model_accuracy*100)+ "% Accuracy")




GRADIENT BOOST



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier


data = pd.read_csv("Breast_Cancer2.csv", encoding="ISO-8859-1")
x= data.iloc[:,1:8] #df.iloc[0:3,1:7]
y=data.iloc[:,10]
data
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
y_train = sc.fit_transform(y_train)

lr_list =[0.05,0.075,0.1,0.25,0.5,0.75,1]
for learning_rate in lr_list:
    gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=learning_rate,
                                        max_features=2,max_depth=2,random_state=0)
    gb_clf.fit(x_train,y_train)
print("Learning Score:", learning_rate)
print("Accuracy score (training):{0:.3f}".format(gb_clf.score(x_train,y_train)))
print("Accuracy score (Validation):{0:.3f}".format(gb_clf.score(x_train,y_train)))




ADABOOST


from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics


iris = datasets.load_iris()
x=iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=0)

abc = AdaBoostClassifier(n_estimators=55, learning_rate=2)

model = abc.fit(x_train, y_train)
y_pred = model.predict(x_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred)*100,"%")




TEXT CLASSIFICATION

SOURCE CODE:
from sklearn.datasets import fetch_20newsgroups
twenty_train=fetch_20newsgroups(subset='train',shuffle=True)
twenty_train.target_names
print("\n".join(twenty_train.data[0].split("\n")[:3]))
from sklearn.feature_extraction.text import CountVectorizer
count_vect=CountVectorizer()
X_train_counts=count_vect.fit_transform(twenty_train.data)
X_train_counts.shape
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer=TfidfTransformer()
X_train_tfidf=tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape
from sklearn.naive_bayes import MultinomialNB
clf=MultinomialNB().fit(X_train_tfidf,twenty_train.target)
from sklearn.pipeline import Pipeline
text_clf=Pipeline([('vect',CountVectorizer()),
                   ('tfidf',TfidfTransformer()),
                   ('clf',MultinomialNB())])
text_clf=text_clf.fit(twenty_train.data,twenty_train.target)
import numpy as np
twenty_test=fetch_20newsgroups(subset='test',shuffle=True)
predicted=text_clf.predict(twenty_test.data)
np.mean(predicted==twenty_test.target)






TEXT CLUSTERING

SOURCE CODE:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

documents=["This little kitty came to play when I was eating at a restaurant.","Merley has the best squooshy kitten belly.","Google translate app is incredible.","If you open 100 tabs in google you get a smiley face.","Best cat photo I've ever taken.","Climbing ninja cat.", "Impressed with google map feedback.","Key promoter extension for Google Chrome."]
vectorizer=TfidfVectorizer(stop_words='english')
X=vectorizer.fit_transform(documents)
true_k=2
model=KMeans(n_clusters=5,init='k-means++',max_iter=100,n_init=1)
model.fit(X)

print("Top terms per cluster:")
order_centroid=model.cluster_centers_.argsort()[:,::-1]
terms=vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroid[i,:10]:
        print(' %s' %terms[ind]),
    print
print("\n")
print("Prediction")

Y=vectorizer.transform(["chrome browser to open."])
prediction=model.predict(Y)
print(prediction)

Y=vectorizer.transform(["My cat is hungry."])
prediction=model.predict(Y)
print(prediction)


SENTIMENTAL ANALYSIS

SOURCE CODE:
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sentiment = SentimentIntensityAnalyzer()
text_1 = "The book was a perfect balance between wrtiting style and plot."
text_2 =  "The pizza tastes terrible."
sent_1 = sentiment.polarity_scores(text_1)
sent_2 = sentiment.polarity_scores(text_2)
print("Sentiment of text 1:", sent_1)
print("Sentiment of text 2:", sent_2)
import pandas as pd
data = pd.read_csv('Finance_data.csv')
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
text_counts = cv.fit_transform(data['Sentence'])
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(text_counts, data['Sentiment'], test_size=0.25, random_state=5)
from sklearn.naive_bayes import MultinomialNB
MNB = MultinomialNB()
MNB.fit(X_train, Y_train)
racy score of the model
from sklearn import metrics
predicted = MNB.predict(X_test)
accuracy_score = metrics.accuracy_score(predicted, Y_test)
print("Accuracuy Score: ",accuracy_score)


RECOMMANDER SYSTEM
SOURCE CODE:
import pandas as pd
data=pd.read_csv('U:/pcsf18/Python/Recom_System.csv')
data.head()
description=data['Gender'] +' '+ data['Subject'] +' '+ data['Stream'] +' '+ data['Marks'].apply(str)
description[0]
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
def create_similarity_matrix(new_description,overall_description):
    overall_description.append(new_description)
    tfidf= TfidfVectorizer(stop_words="english")
    tfidf_matrix=tfidf.fit_transform(overall_description)
    tfidf_matrix.shape
    cosine_sim=linear_kernel(tfidf_matrix,tfidf_matrix)
    return cosine_sim
    
def get_recommendations(new_description,overall_description):
    cosine_sim= create_similarity_matrix(new_description, overall_description)
    sim_scores=list(enumerate(cosine_sim[-1]))
    sim_scores=sorted(sim_scores,key=lambda x:x[1],reverse=True)
    sim_scores=sim_scores[1:10]
    indices=[i[0]for i in sim_scores]
    return data.iloc[indices]
new_description=pd.Series('male physics science 78')
get_recommendations(new_description,description)




CNN

SOURCE CODE:

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
(X_train,Y_train) , (X_test,Y_test)=mnist.load_data()
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2],1))
X_test = X_test.reshape((X_test.shape[0], X_train.shape[1], X_test.shape[2],1))
print(X_train.shape)
print(X_test.shape)
X_train = X_train/225
X_test = X_test/225
model = Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
model.add(MaxPool2D(2,2))
model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X_train,Y_train,epochs=10)
model.evaluate(X_test,Y_test)


RNN

SOURCE CODE:

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
data = [[i for i in range(100)]]
data = np.array(data, dtype=float)
target = [[i for i in range(1,101)]]
target = np.array(target, dtype=float)
data = data.reshape((1,1,100))
target = target.reshape((1,1,100))
X_test= [i for i in range(100,200)]
X_test= np.array(X_test).reshape((1,1,100));
Y_test= [i for i in range(101,201)]
Y_test= np.array(Y_test).reshape((1,1,100));
model = Sequential()
model.add(LSTM(100,input_shape=(1,100),return_sequences=True))
model.add(Dense(100))
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
model.fit(data,target,epochs=10, batch_size=1, verbose=2, validation_data=(X_test,Y_test))
predict = model.predict(X_test)


