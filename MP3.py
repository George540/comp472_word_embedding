##########################################
## MP3.py (Mini-Project 3 COMP 472)
## A program to use different word embeddings 
## to answer the Synonym Test automatically and compare the
## performance of different models
## Created by Team Oranges
##########################################
'''
Using large amounts of unannotated plain text, word2vec learns relationships between words automatically. 
The output are vectors, one vector per word, with remarkable linear relationships that allow us to do things like:

vec(“king”) - vec(“man”) + vec(“woman”) =~ vec(“queen”)
vec(“Montreal Canadiens”) – vec(“Montreal”) + vec(“Toronto”) =~ vec(“Toronto Maple Leafs”).
'''
import gensim.downloader as api
import pandas as pd
import csv

#TASK 1: Evaluation of the word2vec-google-news-300 Pre-trained Model
#Q1
'''
First, use gensim.downloader.load to load the
word2vec-google-news-300 pretrained embedding model.
'''
word2vec_model = api.load('word2vec-google-news-300')
#pd.set_option('display.max_rows', 80)
df = pd.read_csv("synonyms.csv")

#Creating a dictionary from the words in word2vec_model
googleDict = {}
for index, word in enumerate(word2vec_model.index_to_key):
    googleDict[index] = word

'''
Second, use the similarity method from Gensim to compute the cosine 
similarity between 2 embeddings (2 vectors) and find the closest 
synonym to the questionword.
'''
fields = ['Question-word,', 'answer-word,', 'guess-word,', 'result']
filename = 'word2vec-google-news-300-details.csv'
f = open(filename, 'w', newline='')
writer = csv.writer(f)
row = fields
writer.writerow(row)

for i in range(0 , 80):
    #get cell in specified columns to fill the variables below
    qWord = df.at[i, 'question']
    answer = df.at[i, 'answer']
    opt0 = df.at[i, '0']
    opt1 = df.at[i, '1']
    opt2 = df.at[i, '2']
    opt3 = df.at[i, '3']
    #check if word exists in the googledict then 
    #get the highest similar word via its 
    #value using the .similarity() function
    if qWord in googleDict.values() and answer in googleDict.values() and opt0 in googleDict.values() and opt1 in googleDict.values() and opt2 in googleDict.values() and opt3 in googleDict.values():
        most_similar = opt0
        max = word2vec_model.similarity(qWord, opt0)
        temp = word2vec_model.similarity(qWord, opt1)

        if temp > max:
            most_similar = opt1
            max = temp

        temp = word2vec_model.similarity(qWord, opt2)
        
        if temp > max:
            most_similar = opt2
            max = temp

        temp = word2vec_model.similarity(qWord, opt3)

        if temp > max:
            most_similar = opt3
            max = temp

        if most_similar == answer: result = 'correct'
        else: result = 'wrong'

        print('-' * 75)
        print('ROW ', i, ' ')
        row = [qWord,answer,most_similar,result]
        writer.writerow(row)
        print('Question-word:', qWord,' answer:', answer,' guess word: ', most_similar, ' ,', result)
        print()
    else:
        print('-' * 75)
        print('ROW ', i, ' ')
        row = [qWord,answer,'guess','guess']
        writer.writerow(row)
        print('A word in this row was not found in the dictionary... Skipping it!')
        print()
f.close()

#TASK 1 Q2
df = pd.read_csv("word2vec-google-news-300-details.csv")
#count the nymber of times the model correctly guess the synonym
correct_guesses = df.loc[df.result == 'correct', 'result'].count()
#count the nymber of times the model answered incorrectly
wrong_guesses = df.loc[df.result == 'wrong', 'result'].count()
#count the nymber of times the model answered without guessing
unattempted_guesses = df.loc[df.result == 'guess', 'result'].count() 

total_answered = correct_guesses + wrong_guesses
accuracy = round(correct_guesses / total_answered, 2)

fields = ['word2vec-google-news-300,', len(googleDict), str(correct_guesses)+',', str(unattempted_guesses)+',', accuracy]
filename = 'analysis.csv'
f = open(filename, 'w', newline='')
writer = csv.writer(f)
row = fields
writer.writerow(row)
f.close()