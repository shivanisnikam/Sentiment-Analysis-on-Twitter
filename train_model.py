def main():
    #--------------------------------------------------------------
    # Include Libraries
    #--------------------------------------------------------------
    
    import pandas as pd
    from sklearn.model_selection import train_test_split
    #import sklearn
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn import metrics
    #from pandas_ml import ConfusionMatrix
    from matplotlib import pyplot as plt
    from sklearn.linear_model import PassiveAggressiveClassifier
    import itertools
    import numpy as np
    import re
    import warnings
    import pickle
    import dill
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    #--------------------------------------------------------------
    # Importing dataset using pandas dataframe
    #--------------------------------------------------------------
    df = pd.read_csv("Fake_real.csv")
    
    # Set index 
    df = df.set_index("Unnamed: 0")
    
    #--------------------------------------------------------------
    # Separate the labels and set up training and test datasets
    #--------------------------------------------------------------
    X = df.text
    y = df.sent

    
    # Make training and test sets 
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=2000)
    
    ##==============================================================
    ##==============================================================
    #--------------------------------------------------------------
    # Building the Count and Tfidf Vectors 4. Feature Extraction
    #--------------------------------------------------------------
    
    # Initialize the `tfidf_vectorizer`
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(2,3),analyzer=lambda x:x, max_df=0.9,stop_words='english')    # This removes words which appear in more than 70% of the articles
    
    # Fit and transform the training data 
    tfidf_train = tfidf_vectorizer.fit_transform(X_train) 
    
    # Transform the test set 
    tfidf_test = tfidf_vectorizer.transform(X_test)

    
    print(pd.DataFrame(tfidf_train.A, columns=tfidf_vectorizer.get_feature_names()))


    #--------------------------------------------------------------
    # Function to plot the confusion matrix 
    #--------------------------------------------------------------
    
    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig("A.png")
    #--------------------------------------------------------------
    # Applying Passive Aggressive Classifier
    #--------------------------------------------------------------
    
    clf_PAC = PassiveAggressiveClassifier()  #(n_iter=50)
    
    clf_PAC.fit(tfidf_train, y_train)
    
    pred = clf_PAC.predict(tfidf_test)
    score = metrics.accuracy_score(y_test, pred)
    print("PassiveAggressiveClassifier acuracy:   %0.3f" % round(score*100,2))
    A="PassiveAggressiveClassifier accuracy:   %0.3f" % round(score*100,2)
    
    
#    pickle.dump(vectorizer, open("vectorizer.pickle", "wb"))
    with open('clf_PAC.pkl', 'wb') as f:
        pickle.dump(clf_PAC, f)
#dill.dump(lambda x:x+1, open('yeah.p', 'wb'))
    with open('vectorizer.pkl', 'wb') as f:
        dill.dump(tfidf_vectorizer, f)


    #--------------------------------------------------------------
    # Naive Bayes classifier for Multinomial model 
    #--------------------------------------------------------------
    
    clf_MNB = MultinomialNB() 
    
    clf_MNB.fit(tfidf_train, y_train)                       # Fit Naive Bayes classifier according to X, y
    
    pred = clf_MNB.predict(tfidf_test)                     # Perform classification on an array of test vectors X.
    score = metrics.accuracy_score(y_test, pred)
    print("Multinomial Naive Bayes accuracy:   %0.3f" % round(score*100,2))
    
    cm = metrics.confusion_matrix(y_test, pred, labels=[1, 0,-1])
    plot_confusion_matrix(cm, classes=[1, 0,-1])
#    print(cm)
    ## now you can save it to a file
    with open('clf_MNB.pkl', 'wb') as f:
        pickle.dump(clf_MNB, f)

    B= "Multinomial Naive Bayes accuracy:   %0.3f" % round(score*100,2)

   
    msg1= A  + '\n' + B
    
    return msg1

#main()
