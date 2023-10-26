EXERCISE:01  DATA HANDLING
CODE:
import pandas as pd
surveys_df=pd.read_csv("data2.csv")
surveys_df['Age']
surveys_df.Name
surveys_Name=surveys_df['Name']
surveys_Name
surveys_Name=surveys_df[['Name','Age']]
surveys_Name
surveys_df[0:3]
surveys_df[:5]
surveys_df[-1:]
surveys_copy=surveys_df
surveys_copy[0:3]=0
surveys_copy=surveys_df.copy()
surveys_df.loc[:,'Name':'month']
surveys_df[-1:-4]
surveys_df.loc[:,'Name':'month':2]
surveys_df.loc[:,:'month']
surveys_df.loc[:,'Name'::3]
surveys_df.loc[:,['Name','Salary','Year']]
surveys_df.iloc[0:3,1:4]
surveys_df.iloc[[0,7],:]
surveys_df.iloc[2,4]
surveys_df.iloc[0,3]
surveys_df[surveys_df.Salary==30000]
surveys_df[surveys_df.Salary>30000]
surveys_df[(surveys_df.Empno>=204)&(surveys_df.Salary<=20000)]
surveys_df[surveys_df.Name=='akash']
surveys_df[surveys_df.month.isin(['February','April','September'])&(surveys_df.Year==2002)]
surveys_df[(surveys_df.Year.between(2004,2011))&(surveys_df.month=='September')]

 
OUTPUT:
 

 
EXERCISE:02    LINEAR REGRESSION
CODE:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv('sales.csv')
dataset.head()
x=dataset.iloc[:,:-1]
y=dataset.iloc[:,1]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
print(regressor.intercept_)
print(regressor.coef_)
y_pred=regressor.predict(x_test)
df=pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
df
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)
from sklearn import metrics
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,y_pred))
print('Mean Squared Error:',metrics.mean_squared_error(y_test,y_pred))
print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

 
OUTPUT:
 
 
EXERCISE:3 SVM
CODE:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
data=pd.read_csv('kyphosis.csv')
x=data.iloc[:,[1,2,3]]
y=data.iloc[:,0]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
from sklearn.svm import SVC
clf=SVC(kernel="linear")
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
print(str(accuracy*100)+"% accuracy")
cm1=confusion_matrix(y_test,y_pred,labels=['absent','present'])
print(cm1)
df_confusion=pd.crosstab(y_test,y_pred)
print(df_confusion)
df_confusion1=pd.crosstab(y_test,y_pred,rownames=['Actual'],colnames=['Predicted'],margins=True)
print(df_confusion1)
sensitivity1=cm1[1,1]/(cm1[1,0]+cm1[1,1])
print('Sensitivity:',sensitivity1)
specificity1=cm1[0,0]/(cm1[0,0]+cm1[0,1])
print('Specificity:',specificity1)

 
OUTPUT:
 
 
EXERCISE:4 DECISION TREE
CODE:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
data=pd.read_csv('kyphosis.csv')
x=data.iloc[:,[1,2,3]]
y=data.iloc[:,0]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
clf=tree.DecisionTreeClassifier(criterion='gini',min_samples_split=30,splitter='best')
clf=clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
df=pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
df
accuracy=accuracy_score(y_test,y_pred)
print(str(accuracy*100)+"% accuracy")
cm1=confusion_matrix(y_test,y_pred,labels=['absent','present'])
print(cm1)
df_confusion=pd.crosstab(y_test,y_pred)
print(df_confusion)
df_confusion1=pd.crosstab(y_test,y_pred,rownames=['Actual'],colnames=['Predicted'],margins=True)
print(df_confusion1)
sensitivity1=cm1[1,1]/(cm1[1,0]+cm1[1,1])
print('Sensitivity:',sensitivity1)
specificity1=cm1[0,0]/(cm1[0,0]+cm1[0,1])
print('Specificity:',specificity1)

 
OUTPUT:
 
 
EXERCISE:5 KNN
CODE:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
data=pd.read_csv('kyphosis.csv')
x=data.iloc[:,[1,2,3]]
y=data.iloc[:,0]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
from sklearn.neighbors import KNeighborsClassifier
clf=KNeighborsClassifier(n_neighbors=5)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
df=pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
df
accuracy=accuracy_score(y_test,y_pred)
print(str(accuracy*100)+"% accuracy")
cm1=confusion_matrix(y_test,y_pred,labels=['absent','present'])
print(cm1)
df_confusion=pd.crosstab(y_test,y_pred)
print(df_confusion)
df_confusion1=pd.crosstab(y_test,y_pred,rownames=['Actual'],colnames=['Predicted'],margins=True)
print(df_confusion1)
sensitivity1=cm1[1,1]/(cm1[1,0]+cm1[1,1])
print('Sensitivity:',sensitivity1)
specificity1=cm1[0,0]/(cm1[0,0]+cm1[0,1])
print('Specificity:',specificity1)

 
OUTPUT:
 
 
EXERCISE:6 LOGISTIC REGRESSION
CODE:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
data=pd.read_csv('kyphosis.csv')
x=data.iloc[:,[1,2,3]]
y=data.iloc[:,0]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
from sklearn.linear_model import LogisticRegression
clf=LogisticRegression(random_state=0)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
df=pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
df
accuracy=accuracy_score(y_test,y_pred)
print(str(accuracy*100)+"% accuracy")
cm1=confusion_matrix(y_test,y_pred,labels=['absent','present'])
print(cm1)
df_confusion=pd.crosstab(y_test,y_pred)
print(df_confusion)
df_confusion1=pd.crosstab(y_test,y_pred,rownames=['Actual'],colnames=['Predicted'],margins=True)
print(df_confusion1)
sensitivity1=cm1[1,1]/(cm1[1,0]+cm1[1,1])
print('Sensitivity:',sensitivity1)
specificity1=cm1[0,0]/(cm1[0,0]+cm1[0,1])
print('Specificity:',specificity1)
 
OUTPUT:
 
 
EXERCISE:7 KMEANS
CODE :
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
df=pd.read_csv("kmeans.csv")
df.head()
from sklearn.cluster import KMeans
km=KMeans(n_clusters=4,init='k-means++',n_init=10)
km.fit(df)
y_hc=km.fit_predict(df)
y_hc
df["Cluster"]=y_hc
df.head()
df
x=df
df1=df.sort_values(['Cluster'])
df1
print(df1)
zero_member=df[df.Cluster==0]
print(zero_member)
first_member=df[df.Cluster==1]
print(first_member)
second_member=df[df.Cluster==2]
print(second_member)
third_member=df[df.Cluster==3]
print(third_member)

 
OUTPUT:
 

 
EXERCISE:8 ELBOW METHOD
CODE:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
df=pd.read_csv("kmeans.csv")
df.head()
from sklearn.cluster import KMeans
sum_of_squared_distance=[]
K=range(1,15)
for k in K:
           km=KMeans(n_clusters=k)
           km=km.fit(df)
           sum_of_squared_distance.append(km.inertia_)
plt.plot(K,sum_of_squared_distance,'bx-')
plt.xlabel('k')
plt.ylabel('sum_of_squared_distance')
plt.title('Elbow Method For Optimal k')
plt.show()

 
OUTPUT:
 

 
EXERCISE:9 HIERARCHICAL CLUSTERING
CODE:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset=pd.read_csv('Mall_Customers.csv')
x=dataset.iloc[:,[3,4]].values
import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(x,method='ward'))
plt.title('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()
from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_hc=hc.fit_predict(x)
dataset["Cluster"]=y_hc
dataset.head()
dataset
df1=dataset.sort_values(['Cluster'])
df1()
print(df1)
zero_member=dataset[dataset.Cluster==0]
print(zero_member)
first_member=dataset[dataset.Cluster==1]
print(first_member)
second_member=dataset[dataset.Cluster==2]
print(second_member)
third_member=dataset[dataset.Cluster==3]
print(third_member)
fourth_member=dataset[dataset.Cluster==4]
print(fourth_member)
plt.scatter(x[y_hc==0,0],x[y_hc==0,1],s=100,c='red',label='Cluster 0')
plt.scatter(x[y_hc==1,0],x[y_hc==1,1],s=100,c='blue',label='Cluster 1')
plt.scatter(x[y_hc==2,0],x[y_hc==2,1],s=100,c='green',label='Cluster 2')
plt.scatter(x[y_hc==3,0],x[y_hc==3,1],s=100,c='cyan',label='Cluster 3')
plt.scatter(x[y_hc==4,0],x[y_hc==4,1],s=100,c='magenta',label='Cluster 4')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
 
OUTPUT:
    
EXERCISE:10 RANDOM FOREST
CODE:
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
data=pd.read_csv('kyphosis.csv')
data.head()
x=data.iloc[:,[1,2,3]]
y=data.iloc[:,0]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=10,criterion='gini',random_state=0)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
print(str(accuracy*100)+"% accuracy")
cm1=confusion_matrix(y_test,y_pred,labels=['absent','present'])
print(cm1)
df_confusion=pd.crosstab(y_test,y_pred)
print(df_confusion)
df_confusion1=pd.crosstab(y_test,y_pred,rownames=['Actual'],colnames=['Predicted'],margins=True)
print(df_confusion1)
sensitivity1=cm1[1,1]/(cm1[1,0]+cm1[1,1])
print('Sensitivity:',sensitivity1)
specificity1=cm1[0,0]/(cm1[0,0]+cm1[0,1])
print('Specificity:',specificity1)
 
OUTPUT:
 
 
EXERCISE:11 STACKING
CODE:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
iris=pd.read_csv("Breast_Cancer2.csv",encoding='unicode_escape')
x=iris.iloc[:,1:8]
y=iris.iloc[:.10]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
from sklearn.neigbhors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)
knn_accuracy=accuracy_score(y_test,y_pred)
print("KNN: "+str(knn_accuracy*100)+"% accuracy")
from sklearn.naive_bayes import GaussianNB
NB_clf=GaussianNB()
NB_clf.fit(x_train,y_train)
y_pred=NB_clf.predict(x_test)
NB_accuracy=accuracy_score(y_test,y_pred)
print("Naive Bayes: "+str(NB_accuracy*100)+"% accuracy")
from sklearn import tree
dt_clf=tree.DecisionTreeClassifier(criterion='gini',min_samples_split=30,splitter='best')
dt_clf.fit(x_train,y_train)
y_pred=dt_clf.predict(x_test)
dt_accuracy=accuracy_score(y_test,y_pred)
print("Decision Tree: "+str(dt_accuracy*100)+"% accuracy")
 
OUTPUT:
 
 
EXERCISE:12 ADABoost
CODE:
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
iris=datasets.load_iris();
x=iris.data
y=iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
abc=AdaBoostClassifier(n_estimators=55,learning_rate=2)
model=abc.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test,y_pred)*100,"%")

 
OUTPUT:
 
 
EXERCISE:13 GRADIENT BOOST
CODE:
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
data=pd.read_csv("Breast_Cancer2.csv")
x=data.iloc[:,1:8]
y=data.iloc[:,10]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
lr_list=[0.05,0.075,0.1,0.25,0.5,0.75,1]
for learning_rate in lr_list:
    gb_clf=GradientBoostingClassifier(n_estimators=20,learning_rate=learning_rate,max_features=2,max_depth=2,random_state=0)
    gb_clf.fit(x_train,y_train)
print("Learning Score: ",learning_rate)
print("Accuracy Score (Training): {0:.3f}".format(gb_clf.score(x_train,y_train)))
print("Accuracy Score (Validation): {0:.3f}".format(gb_clf.score(x_test,y_test)))
 
OUTPUT:
 

 
EXERCISE:14 KFOLD CROSS VALIDATION
CODE:
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score,KFold
from sklearn.linear_model import LogisticRegression
iris=load_iris()
x=iris.data
y=iris.target
logreg=LogisticRegression()
kf=KFold(n_splits=5)
score=cross_val_score(logreg,x,y,cv=kf)
print("Cross Validation Scores are {}".format(score))
print("Average Cross Validation Score:{}".format(score.mean()))

 
OUTPUT:
 
 
EXERCISE:15 TEXT SIMILARITY
CODE:
import en_core_web_sm
from numpy import dot
from numpy.linalg import norm
import numpy as np

nlp=en_core_web_sm.load()
doc_list=['I love this sandwich','this is an amazing place',
          'I feel very good about these beers',
          'this is my best work',
          'what an awesome view',
          'i do not like the restaurant',
          'i am tired of this stuff',
          'i cant deal with this',
          'he is my sworn enemy',
          'my boss is horrible',
          'i hate this sandwhich.']

query=input()
sim_scores=[]
q_vector=nlp(query).vector
for doc in doc_list:
    doc_vector=nlp(doc).vector
    cos_sim=dot(q_vector,doc_vector)/(norm(q_vector)*norm(doc_vector))
    sim_scores.append(cos_sim)
most_similar=doc_list[sim_scores.index(max(sim_scores))]
print("\n Most similar:\n",most_similar)
top_index=list(np.argsort(sim_scores)[-5:])
top_index.reverse()
print("\n Most similar documents")
for i in top_index:
    print(doc_list[i])
 
OUTPUT:
 
 
EXERCISE:16 TEXT CLASSIFICATION
CODE:
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
 
OUTPUT:
  
EXERCISE:17 TEXT CLUSTERING
CODE:
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
 
OUTPUT:
Cluster 0:
 eating
 kitty
 little
 came
 restaurant
 play
 ve
 feedback
 face
 extension
Cluster 1:
 google
 incredible
 app
 translate
 feedback
 impressed
 map
 key
 promoter
 chrome

print(prediction)
[1]

print(prediction)
[2]

 
EXERCISE:18 SENTIMENT ANALYSIS
CODE:
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
from sklearn import metrics
predicted = MNB.predict(X_test)
accuracy_score = metrics.accuracy_score(predicted, Y_test)
print("Accuracuy Score: ",accuracy_score)
OUTPUT:
 
 
EXERCISE:19 RECOMMENDER SYSTEM
CODE:
import pandas as pd
data=pd.read_csv('Recom_System.csv')
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
   cosine_sim=create_similarity_matrix(new_description, overall_description)
sim_scores=list(enumerate(cosine_sim[-1]))
sim_scores=sorted(sim_scores,key=lambda x:x[1],reverse=True)
sim_scores=sim_scores[1:10]
    indices=[i[0]for i in sim_scores]
    return data.iloc[indices]
new_description=pd.Series('male physics science 78')
get_recommendations(new_description,description)
 
OUTPUT:
  
EXERCISE:20 CNN
CODE:
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train=x_train.reshape((x_train.shape[0],x_train.shape[1],x_train.shape[2],1))
x_test=x_test.reshape((x_train.shape[0],x_train.shape[1],x_train.shape[2],1))
print(x_train.shape)
print(x_test.shape)
x_train=x_train/225
x_test=x_test/225
model=Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
model.add(MaxPool2D(2,2))
model.add(Flatten)
model.add(Dense(100,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=10)
model.evaluate(x_test,y_test)
 
OUTPUT:
 
 
EXERCISE:21 RNN
CODE:
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
data = [[i for i in range(100)]]
data = np.array(data,dtype=float)

target =  [[i for i in range(1,101)]]
target = np.array(target,dtype=float)
data = data.reshape((1,1,100))
target =target.reshape((1,1,100))
x_test =[[i for i in range(100,200)]]
x_test = np.array(x_test).reshape((1,1,100))
y_test =[[i for i in range(101,201)]]
y_test = np.array(x_test).reshape((1,1,100))

model = Sequential()
model.add(LSTM(100,input_shape=(1,100),return_sequences=True))
model.add(Dense(100))
model.compile(loss='mean_absolute_error',optimizer='adam',metrics=['accuracy'])
model.fit(data,target,epochs=10,batch_size=1, verbose = 2,validation_data=(x_test,y_test))
predict = model.predict(x_test)
 
OUTPUT:
 
