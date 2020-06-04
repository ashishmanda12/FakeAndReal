import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report ,accuracy_score
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import TfidfVectorizer 


# Opening the news.csv file

df = pd.read_csv('news.csv')


# filtering the dataframe

df = df.filter(['title','text','label'], axis=1)
filt = (df['label'] == 'FAKE') | (df['label'] == 'REAL')
df = df[filt] 


# Assigning FAKE = 0 AND REAL =1

df.loc[df["label"]=="FAKE","label"]=0
df.loc[df["label"]=="REAL","label"]=1



# Concatenating the title and text columns

df["text"]= df["title"].astype(str)+df["text"]
df = df.filter(['text','label'], axis=1)


# Ploting no. of real news and no. of fake news

print(df["label"].value_counts())
print(sns.countplot(df["label"]))
plt.show()


# Assigning the values of X , y, X_train, y_train, X_test, y_test

X = df['text']
y = df['label'].astype('int')

X_train ,X_test,y_train,y_test = train_test_split(X,y ,test_size =0.2,random_state =42)




# Converting words to numeric values by using TfidfVectorizer  

cv = TfidfVectorizer(min_df =1,stop_words = 'english')


X_train_cv = cv.fit_transform(X_train)
X_train_cv = X_train_cv.toarray()

X_test_cv = cv.transform(X_test)
X_test_cv = X_test_cv.toarray()



# Initilising PassiveAggressiveClassifier

clf = PassiveAggressiveClassifier(max_iter=1000, random_state=0, tol=1e-3)
clf.fit(X_train_cv,y_train)
pred = clf.predict(X_test_cv)


# Printing Classification_report and Confusion matrix

print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))

# Printing accuracy score
 
cm = accuracy_score(y_test,pred)
print(cm)