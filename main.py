import pandas as pd
import os
import PySimpleGUI as sg
import string
from collections import Counter
import nltk
from natsort import natsorted
#nltk.download('stopwords')
#nltk.download('punkt')
from nltk.corpus import stopwords
from nltk import word_tokenize

def articleClean(file):
    stop_words = set((stopwords.words('english')) + list(string.punctuation))

    symbols = ["\n", "’", "“", "”", "'", "`", "-", "[", "]", "»", ":"] 

    article = ""
    
    for i in symbols:
        file = file.replace(i, " ")

    def listToString(s): #function that converts arr of strings into one string (used later)
        str1 = " "
        return (str1.join(s))
    
    word_tokens= word_tokenize(file.lower()) #converts arcivle into array of words

    filtered_article = [] 

    for w in word_tokens: # check if word from article is the extra , if not assign the word in the clean words array
        if w not in stop_words and w.isalpha() == True:
            filtered_article.append(w)

    article = listToString(filtered_article)

    return article

all_articles = []
genres = []

os.chdir("/home/farkhad/Documents/University/Data_Mining/article_classification/articles/technology/training/")
tech_res_art = []
for i in range(1,101):
    tech_res = open("training_tech" + str(i) + ".txt", "r").read()
    tech_res = articleClean(tech_res)
    tech_res_art.append(tech_res)
    all_articles.append(tech_res)
    genres.append("technology")

#PRINT PASS

os.chdir("/home/farkhad/Documents/University/Data_Mining/article_classification/articles/sport/training/")
sport_res_art = []

for i in range(1,101):
    sport_res = open("training_sport" + str(i) + ".txt", "r").read()
    sport_res = articleClean(sport_res)
    sport_res_art.append(sport_res)
    all_articles.append(sport_res)
    genres.append("sport")

#PRINT PASS

stop_words = stopwords.words('english')
word_tokens = word_tokenize("hello my name is Pierre")
filtered_data = []
print(word_tokens)
for word in word_tokens:
    if word not in stop_words:
        filtered_data.append(word)
print(filtered_data)

d = {'articles' : all_articles, 'genres' : genres}
df = pd.DataFrame(data=d)
#df=df.transpose()


df["articles"] = df["articles"].apply(lambda str : str.lower())


sum_s = 0
sum_t = 0
for i in df['genres']:
    if i == "sport":
        sum_s +=1
    else:
        sum_t +=1

prob_s = sum_s/len(df)
prob_t = sum_t/len(df)
print(prob_s)
print(prob_t)

df["bow"]  = df["articles"].str.split().apply(Counter)

sum_bow_s = {}
sum_bow_t = {}
i=0
for bow in df["bow"]:
    for key in bow:
        if df["genres"][i] == "sport":
            if key in sum_bow_s:
                sum_bow_s[key] += bow[key]
            else:
                sum_bow_s[key] = bow[key]
        else:
            if key in sum_bow_t:
                sum_bow_t[key] +=bow[key]
            else:
                sum_bow_t[key] = bow[key]
    i += 1

#PRINT PASS

term_occurences_s = 0
term_occurences_t = 0
for key in sum_bow_s:
    term_occurences_s += sum_bow_s[key]
for key in sum_bow_t:    
    term_occurences_t += sum_bow_t[key]

#print(term_occurences_t) 68593
#print(term_occurences_s) 64906

temp = []
unique_term = 0
for bow in df["bow"]:
    for key in bow:
        if key not in temp:
            unique_term += 1
            temp.append(key)

#print(temp)
#print(unique_term) 19813

sum_bow = {}
term_occurences = 0
for bow in df["bow"]:
    for key in bow:
        if key in sum_bow:
            sum_bow[key] += bow[key]
        else:
            sum_bow[key] = bow[key]
        term_occurences += bow[key]

prob_bow = {}
# Adding +1 for better results
for key in sum_bow:
    prob_bow[key] = (sum_bow[key]/term_occurences)

def split(words):
    return words.split()

def lower(words):
    return words.lower()

def probSport(article):
    article = split(lower(article))
    likelihood = 1
    evidence = 1
    for term in article:
        if term in sum_bow_s:
            likelihood *= (sum_bow_s[term]+1)/(term_occurences_s+unique_term)
        else:
            likelihood *= 1/(term_occurences_s+unique_term)
        evidence *= prob_bow[term]
    likelihood_prior = likelihood*prob_s
    posterior = likelihood_prior/evidence
    return posterior

def probTech(article):
    article = split(lower(article))
    likelihood = 1
    evidence = 1
    for term in article:
        if term in sum_bow_t:
            likelihood *= (sum_bow_t[term]+1)/(term_occurences_t+unique_term)
        else:
            likelihood *= 1/(term_occurences_t+unique_term)
        evidence *= prob_bow[term]
    likelihood_prior = likelihood*prob_t
    posterior = likelihood_prior/evidence
    return posterior

#print(probTech("technology")) 
#print(probSport("football player car team "))


layout = [[sg.Text('Enter article number: ')],      
                 [sg.InputText()],      
                 [sg.Submit(), sg.Cancel()]]      



window = sg.Window('NaiveBaise predictor.', layout)    

event, values = window.read()    
window.close()

text_input = values[0]

final_output = ''
prob_sport = probSport(text_input)
prob_tech = probTech(text_input)
if prob_tech > prob_sport:
    final_output = "It is a tech article"
else: 
    final_output = "It is a sport article"
sg.popup(final_output)