import tkinter as tk
from tkinter import ttk, LEFT, END
from PIL import Image , ImageTk 
import numpy as np
import time
global fn
import textwrap
import pickle
import nltk
#import preprocessor.api as p
#from preprocessor.api import clean, tokenize, parse
#import nltk
from nltk import word_tokenize
import string
from nltk.corpus import stopwords

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import csv

import re
import pandas as pd
#import matplotlib.pyplot as plt
import seaborn as sns
#from pylab import savefig
import dill
sns.set()

import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)

#import matplotlib as plt
from train_model import main
#from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

#####=========================1.Scraping Tweets========================================================
import tweepy
global twit_sel


twit_sel='@RajatSharmaLive'

# Access Credentials
consumer_key = 'U6EGQgda0uChCDbrGDgFmqTbx'
consumer_secret = 'slNKflyQSHpStZXvS1lSXS9MDGb1tg4MyAo1VqVF78I5bZfdR4'
access_token = '1149581774583808001-LmKDH5OHVIT71lDbY5iyiEX9EtyMcq'
access_token_secret = 'hfRBp2FqR0PgmJQoKPFLbhqcyZ5w1zGTPEKWTcfU6BgMP'

# OAuthHandler object
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
# set access token and secret
auth.set_access_token(access_token, access_token_secret)
# create tweepy API object to fetch tweets
api = tweepy.API(auth)

##############################################+=============================================================
root = tk.Tk()
root.configure(background="white")
#root.geometry("1300x700")


w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("Fake News Analysis")
#root.rowconfigure(0, weight = 1)
##############################################+=============================================================
#####For background Image
image2 =Image.open('Fake1.jpg')
image2 =image2.resize((w-700,h), Image.ANTIALIAS)

background_image=ImageTk.PhotoImage(image2)

background_label = tk.Label(root, image=background_image)

background_label.image = background_image

background_label.place(x=700, y=50) #, relwidth=1, relheight=1)
#height=1, width=35,
lbl = tk.Label(root, text="Fake news detection on twitter using machine learning algorithm", font=('times', 20,' bold '),justify=tk.LEFT, wraplength=1100 ,bg="white",fg="lightblue4")
lbl.place(x=200, y=5)


frame_display = tk.LabelFrame(root, text=" --Tweets-- ", width=250, height=650, bd=5, font=('times', 15, ' bold '),bg="white",fg="red")
frame_display.grid(row=0, column=0, sticky='s')
frame_display.place(x=200, y=40)

frame_right = tk.LabelFrame(root, text=" --Preprocessing-- ", width=550, height=650, bd=5, font=('times', 15, ' bold '),bg="white",fg="red")
frame_right.grid(row=0, column=0, sticky='e')
frame_right.place(x=730, y=40)
frame_right.rowconfigure(0, weight = 1)

frame_info = tk.LabelFrame(root, text=" --Information-- ", width=650, height=350, bd=5, font=('times', 15, ' bold '),bg="white",fg="red")
frame_info.grid(row=0, column=0, sticky='s')
frame_info.place(x=730, y=290)



frame_alpr = tk.LabelFrame(root, text=" --Process-- ", width=200, height=650, bd=5, font=('times', 10, ' bold '),bg="lightblue4")
frame_alpr.grid(row=0, column=0, sticky='nw')
frame_alpr.place(x=5, y=0)

###########################################################################################################
#HappyEmoticons
emoticons_happy = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3'
    ])
# Sad Emoticons
emoticons_sad = set([
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
    ])
#Emoji patterns
emoji_pattern = re.compile("["
         u"\U0001F600-\U0001F64F"  # emoticons
         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
         u"\U0001F680-\U0001F6FF"  # transport & map symbols
         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
         u"\U00002702-\U000027B0"
         u"\U000024C2-\U0001F251"
         "]+", flags=re.UNICODE)

#combine sad and happy emoticons
emoticons = emoticons_happy.union(emoticons_sad)
#################################3. Text Pre-processing##################################

def Aclean_tweets(tweet):

    stop_words = set(stopwords.words('english'))
#    print(tweet)
    word_tokens = word_tokenize(tweet)
#    print(word_tokens)
    #after tweepy preprocessing the colon symbol left remain after      #removing mentions
    tweet = re.sub(r':', '', tweet)
    tweet = re.sub(r'‚Ä¶', '', tweet)

    #replace consecutive non-ASCII characters with a space
    tweet = re.sub(r'[^\x00-\x7F]+',' ', tweet)

    # remove retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)

    # remove hashtags
    tweet = re.sub(r'#', '', tweet)
    # removing digits
    tweet = re.sub(r'[0-9]', "", tweet)
    # removing Non-Word Characters
    tweet = re.sub(r"\W", "", tweet, flags=re.I)
    tweet = re.sub(r"[,@\'?\.$%_]", "", tweet, flags=re.I)

    #Removing a Single Character
    tweet = re.sub(r"\s+[a-zA-Z]\s+", " ", tweet)

    try:
        tweet = tweet.decode("utf-8-sig").replace(u"\ufffd", "?")#Unicode Character 'REPLACEMENT CHARACTER'
    except:
        tweet = tweet
    #remove emojis from tweet
    tweet = emoji_pattern.sub(r'', tweet)

    #filter using NLTK library append it to a string
    filtered_tweet = [w for w in word_tokens if not w in stop_words and w not in emoticons and w not in string.punctuation]
    filtered_tweet = []
    #looping through conditions
    for w in word_tokens:
    #check tokens against stop words , emoticons and punctuations
        if w not in stop_words and w not in emoticons and w not in string.punctuation:
            stem_word = stemmer.stem(w) # stemming word

            filtered_tweet.append(stem_word)
    return (" ".join(filtered_tweet)).strip()
#    return ' '.join(filtered_tweet)
    #print(word_tokens)
    #print(filtered_sentence)return tweet
############################################################################################################
canvas2=tk.Canvas(frame_right,bg='#FFFFFF',width=500,height=200,scrollregion=(0,0,1000,4000))

hbar=tk.Scrollbar(frame_right,orient=tk.HORIZONTAL)
hbar.pack(side=tk.BOTTOM,fill=tk.X)
hbar.config(command=canvas2.xview)

vbar=tk.Scrollbar(frame_right,orient=tk.VERTICAL)
vbar.pack(side=tk.RIGHT,fill=tk.Y)
vbar.config(command=canvas2.yview)

canvas2.config(width=500,height=200)
#
canvas2.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
canvas2.pack(side=LEFT,fill=tk.BOTH,expand=1)

#############################################################################################################

canvas=tk.Canvas(frame_display,bg='#FFFFFF',width=250,height=500,scrollregion=(0,0,1000,4000))

hbar=tk.Scrollbar(frame_display,orient=tk.HORIZONTAL)
hbar.pack(side=tk.BOTTOM,fill=tk.X)

hbar.config(command=canvas.xview)
vbar=tk.Scrollbar(frame_display,orient=tk.VERTICAL)
vbar.pack(side=tk.RIGHT,fill=tk.Y)
vbar.config(command=canvas.yview)
canvas.config(width=500,height=550)

canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
canvas.pack(side=LEFT,expand=True,fill=tk.BOTH)
#    canvas.rowconfigure(1, weight = 1)
#    canvas.destroy



wrapper = textwrap.TextWrapper(width=100)

#####========================================================================================================

def clear_img():
    
    img11 = tk.Label(frame_info, background='white',width=650,height=350)
    img11.place(x=0, y=0)

def update_label(str_T):
    result_label = tk.Label(frame_info, text=str_T, width=80, font=("italic", 10),bg='white',fg='blue' )
#    result_label.config(text="")
#    result_label.config(text=str_T)
    result_label.place(x=0, y=0)

def update_label_image(imname):
    
    IMAGE_SIZE=250

    img = Image.open(imname)
    img = img.resize((IMAGE_SIZE+200,IMAGE_SIZE))
    img = np.array(img)

    x1 = int(img.shape[0])
    y1 = int(img.shape[1])
    im = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=im)
    img = tk.Label(frame_info, image=imgtk, height=x1, width=y1+100)
    img.image = imgtk
    img.place(x=0, y=70)

    
def display_tweet(twitter_handle):

    iro=10
    icol=10
    print(twitter_handle)

#    tree.delete(*tree.get_children())
    
    canvas.delete("all")
    
#        twitter_handle=mylistbox.curselection()
    frame_display.config(text="----" + twitter_handle + "----")
#    wrapper = textwrap.TextWrapper(width=100)
#        details=tweet_frm.show_twee()
    tweets = api.user_timeline(twitter_handle, count=30, tweet_mode='extended')
    for t in tweets:
        twit=t.full_text
        word_list = wrapper.wrap(text=twit)
        for element in word_list: 
            canvas.create_text(iro, icol,font="times", anchor="w", text=element)
            icol=icol+30
################################$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
def Display_twits_fun():
    global twit_sel


    if twit_sel!="":
        display_tweet(twit_sel)
#        Example(frame_display,twit_sel).pack(side="top", fill="both", expand=True)
    else:
        print("Please Select Twitter Handle to analysis data")
    

################################################################################################################
itemsforlistbox=['@RajatSharmaLive','@TimesNow','@ndtv','@CNNnews18','@republic']

def CurSelet(event):
    global twit_sel
    twit_sel =""
    widget = event.widget
    selection=widget.curselection()
    picked = widget.get(selection[0])
    twit_sel = picked 
#    Example(frame_display,twit_sel).pack(side="top", fill="both", expand=True)
    
###############################2. Identifying Sentiment type####################################################################################
analyser = SentimentIntensityAnalyzer()
from googletrans import Translator
translator = Translator()

def sentiment_analyzer_scores(text, engl=True):
    if engl:
        trans = text
    else:
        trans = translator.translate(text).text

    score = analyser.polarity_scores(trans)
    lb = score['compound']
    if lb >= 0.05:
        return 1
    elif (lb > -0.05) and (lb < 0.05):
        return 0
    else:
        return -1
  
def list_tweets(user_id, count, prt=False):
    tweets = api.user_timeline(
        "@" + user_id, count=count, tweet_mode='extended')
    tw = []
    for t in tweets:
        tw.append(t.full_text)
        if prt:
            print(t.full_text)
            print()
    return tw


def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)        
    return input_txt


def clean_tweets(lst):
    # remove twitter Return handles (RT @xxx:)
    lst = np.vectorize(remove_pattern)(lst, "RT @[\w]*:")
    # remove twitter handles (@xxx)
    lst = np.vectorize(remove_pattern)(lst, "@[\w]*")
    # remove URL links (httpxxx)
    lst = np.vectorize(remove_pattern)(lst, "https?://[A-Za-z0-9./]*")
    # remove special characters, numbers, punctuations (except for #)
    lst = np.core.defchararray.replace(lst, "[^a-zA-Z0-9#]", " ")

    return lst

def anl_tweets(lst, title='Tweets Classification', engl=True ):
    sents = []
    for tw in lst:
        try:
            st = sentiment_analyzer_scores(tw, engl)
            sents.append(st)
        except:
            sents.append(0)
    ax = sns.distplot(sents,kde=False,bins=3)
    ax.set(xlabel='Fake                Neutral                 Real',
           ylabel='#Tweets',
          title="Tweets of @"+title)
    
    figure = ax.get_figure()    
    figure.savefig('sns_conf.png')  #, dpi=400)
#    sns.savefig("output.png")
    return sents

###=========================================================================================
#
def twitter_stream_listener(file_name,
                            filter_track,
                            follow=None,
                            locations=None,
                            languages=["en"],
                            time_limit=40):
    
    class CustomStreamListener(tweepy.StreamListener):
        def __init__(self, time_limit):
            self.start_time = time.time()
            self.limit = time_limit
            # self.saveFile = open('abcd.json', 'a')
            super(CustomStreamListener, self).__init__()
        def on_status(self, status):
            if (time.time() - self.start_time) < self.limit:
                print(".", end="")
                # Writing status data
                with open(file_name, 'a',encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        status.author.screen_name, status.created_at,
                        status.text
                    ])
            else:
                print("\n\n[INFO] Closing file and ending streaming")
                return False
            
        def on_error(self, status_code):
            if status_code == 420:
                print('Encountered error code 420. Disconnecting the stream')
                # returning False in on_data disconnects the stream
                return False
            else:
                print('Encountered error with status code: {}'.format(
                    status_code))
                return True  # Don't kill the stream

        def on_timeout(self):
            print('Timeout...')
            return True  # Don't kill the stream
    # Writing csv titles
    print(
        '\n[INFO] Open file: [{}] and starting {} seconds of streaming for {}\n'
        .format(file_name, time_limit, filter_track))
    with open(file_name, 'w',encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['author', 'date', 'text'])
    streamingAPI = tweepy.streaming.Stream(
        auth, CustomStreamListener(time_limit=time_limit))
    streamingAPI.filter(
        track=filter_track,
        follow=follow,
        locations=locations,
        languages=languages,
    )
    f.close()





def process_data():
    
    
    
    from os import path,remove
    global twit_sel
    iro=10
    icol=10

####==================================================================   
    if str(path.exists('Raw_tweet_Data.csv'))==True:
        print("remove")
        remove('Raw_tweet_Data.csv')
        
        
    filter_track=[twit_sel,'wall']
    file_name="Raw_tweet_Data.csv"

    twitter_stream_listener(file_name,filter_track,time_limit=180)
######++=========================================================================    

    df_tws = pd.read_csv(file_name)
    df_tws.shape
    
#    print(df_tws.head())
    
    
    df_tws['text'] =  clean_tweets(df_tws['text'])

    print(df_tws.head())
    
    df_tws['sent'] = anl_tweets(df_tws.text)

    df_tws_full = df_tws[df_tws.text != '']
    
    df_tws=df_tws_full.head(20)
#####################################################################################################################
###################################################################################################################
#    print(len(df_tws))
    
    def status(inno):
        if inno == 1:
            return "Real"
        elif inno == -1:
            return "Fake"
        elif inno == 0:
            return "Nutral"
#    tree.delete(*tree.get_children()) 
    canvas2.delete("all")
    canvas2.create_line([(80, 0), (80, 1000)], fill='red', tags='grid_line_w')

    for i in range(len(df_tws)):
        if df_tws['text'][i] != "":
            str2=df_tws['text'][i]
            print(str2)
            canvas2.create_text(30,icol,fill="blue",anchor="w",font="Times 10",text=status(int(df_tws['sent'][i])))
            
#            word_list = wrapper.wrap(text=df_tws['text'][i])
#            lst=str2[0].replace(u"\u2022", "*")
#            lst=str2[1].replace(u"\u2022", "*")
            clean_text = Aclean_tweets(str2)
            word_list = wrapper.wrap(text=clean_text)

            print("--" +  str(word_list[0]))
#            print(clean_tweets(lst))
#            ptweet=clean_tweets(str2)
#            print(ptweet)
            try:
                canvas2.create_text(100,icol,fill="blue",anchor="nw",font="Times 10",text=str(word_list[0]))
            except:
                canvas2.create_text(100,icol,fill="blue",anchor="nw",font="Times 10",text=str("N/A"))
            canvas2.create_line([(0, icol-12), (800, icol-12)], fill='black', tags='grid_line_h')

        icol=icol+50


    if str(path.exists('Fake_real.csv'))==True:
        print("remove")
        remove('Fake_real.csv')
               
    df_tws_full.to_csv("Fake_real.csv")
    clear_img()
    update_label("Processed Data Saved in Fake_real.csv File")
    update_label_image("sns_conf.png")
    
#################################################################################################################

def train_model():
    clear_img()
    update_label("Model Training Start...............")
    
    start = time.time()

    X= main()
    
    end = time.time()
        
    ET="Execution Time: {0:.4} seconds \n".format(end-start)
    
    msg="Model Training Completed.."+'\n'+ X + '\n'+ ET

    update_label(msg)
    update_label_image("A.png")


def plot_window():
   
    def update_lbl(textv):
        label = tk.Label(toproot,text="--- Test Result---",bg="lightblue",fg="dark green",font=('times', 10, ' bold '))
        label.config(text=textv)
        label.pack()

        
    def test_news():
        from nltk.util import ngrams
        from sklearn.feature_extraction.text import TfidfVectorizer
        M=""
#       ===========================================================
        def generate_ngrams(s, n):

            # Convert to lowercases
            s = s.lower()
            # Replace all none alphanumeric characters with spaces
            s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)
            # Break sentence in the token, remove empty tokens
            tokens = [token for token in s.split(" ") if token != ""]
            # Use the zip function to help us generate n-grams
            # Concatentate the tokens into ngrams and return
            ngrams = zip(*[tokens[i:] for i in range(n)])
                
            return [" ".join(ngram) for ngram in ngrams]

#        ==========================================================
        update_lbl("Model Testing Start...............")
        
        start = time.time()
  
        s=[ent1.get()]

        with open('vectorizer.pkl', 'rb') as f:
            tfidf_vectorizer = dill.load(f)
           
        output = tfidf_vectorizer.transform(s)

        with open('clf_PAC.pkl', 'rb') as f:
            clf = pickle.load(f)

        
        Predict_get_value=clf.predict(output)
        X=Predict_get_value
        if X[0] == 1:
            M = "Real"
        elif X[0] == -1:
            M =  "Fake"
        elif X[0] == 0:
            M = "Nutral"
#        X=Model_prediction(output)
        X1="This News is {0}".format(M)
        end = time.time()
            
        ET="Execution Time: {0:.4} seconds \n".format(end-start)
        
        msg="Testing Completed.."+'\n'+ X1 + '\n'+ ET
        
        update_lbl(msg)
        
        
        
    toproot = tk.Toplevel(background="lightblue")
    toproot.geometry("500x200+210+150")
#    toproot.overrideredirect(1)
    toproot.title("------Model Testing Window------")
    # width=650, height=350, bd=5, font=('times', 15, ' bold '),bg="white",fg="red")

    label = tk.Label(toproot,text="Enter Any Sentence ",bg="lightblue",fg="dark green",font=('times', 15, ' bold '))
    label.pack()
    ent1= tk.Entry(toproot,width=80, fg="dark green")
    ent1.pack()
#    label = tk.Label(toproot,text="--- Test Result---",bg="lightblue",fg="dark green",font=('times', 10, ' bold '))
#    label.pack()
#    counter_label(label)
    frame_topw = tk.Frame(toproot, width=200, height=100, bd=5, bg="lightblue4")
#    frame_topw.grid(row=0, column=0, sticky='nw')
    frame_topw.pack(side=tk.BOTTOM)
    
    buttonA = tk.Button(frame_topw ,bg="lightblue",text='Prediction', width=25, command=test_news)
    buttonA.pack(side=tk.LEFT)

    buttonB = tk.Button(frame_topw,bg="lightblue", text='Exit', width=25, command=toproot.destroy)
    buttonB.pack(side=tk.RIGHT)
    
    ent1.focus_set()
        
def window():
    
    root.destroy()


button1 = tk.Button(frame_alpr, text=" Display Tweets ", command=Display_twits_fun,width=18, height=1, font=('times', 12, ' bold '),bg="white",fg="black")
button1.place(x=10, y=40)

mylistbox=tk.Listbox(frame_alpr,width=22,height=6,font=('times',12),fg='red',bg='lightblue')
mylistbox.bind('<<ListboxSelect>>',CurSelet)
mylistbox.place(x=5,y=120)

for items in itemsforlistbox:
    mylistbox.insert(tk.END,items)


#
button4 = tk.Button(frame_alpr, text="Process Tweet Data", command=process_data,width=18, height=1, font=('times', 12, ' bold '),bg="white",fg="black")
button4.place(x=10, y=280)

button5 = tk.Button(frame_alpr, text="Train Model", command=train_model,width=18, height=1,bg="white",fg="black", font=('times', 12, ' bold '))
button5.place(x=10, y=350)
#
button6 = tk.Button(frame_alpr, text="Test Model", command=plot_window,width=18, height=1,bg="white",fg="black", font=('times', 12, ' bold '))
button6.place(x=10, y=420)

exit = tk.Button(frame_alpr, text="Exit", command=window, width=18, height=1, font=('times', 12, ' bold '),bg="red",fg="white")
exit.place(x=10, y=480)



root.mainloop()
