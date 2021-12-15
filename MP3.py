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
import matplotlib.pyplot as plt
import gensim.downloader as api
import pandas as pd
import csv

#TASK 1: Evaluation of the word2vec-google-news-300 Pre-trained Model
#Q1
'''
First, use gensim.downloader.load to load the
word2vec-google-news-300 pretrained embedding model.
'''
def load_model(model):
    return api.load(model)

def task_one_and_two(modelstr, df):
    word2vec_model = load_model(modelstr)
    #pd.set_option('display.max_rows', 80)

    #Creating a dictionary from the words in word2vec_model
    dict = {}
    for index, word in enumerate(word2vec_model.index_to_key):
        dict[index] = word

    '''
    Second, use the similarity method from Gensim to compute the cosine 
    similarity between 2 embeddings (2 vectors) and find the closest 
    synonym to the questionword.
    '''
    fields = ['Question-word,', 'answer-word,', 'guess-word,', 'result']
    filename = modelstr + '-details.csv'
    f = open(filename, 'w', newline='')
    writer = csv.writer(f)
    row = fields
    writer.writerow(row)

    for i in range(0 , 20):
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
        if qWord in dict.values() and answer in dict.values() and opt0 in dict.values() and opt1 in dict.values() and opt2 in dict.values() and opt3 in dict.values():
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
    df = pd.read_csv(filename)
    #count the nymber of times the model correctly guess the synonym
    correct_guesses = df.loc[df.result == 'correct', 'result'].count()
    #count the nymber of times the model answered incorrectly
    wrong_guesses = df.loc[df.result == 'wrong', 'result'].count()
    #count the nymber of times the model answered without guessing
    unattempted_guesses = df.loc[df.result == 'guess', 'result'].count() 
    
    total_answered = correct_guesses + wrong_guesses
    if(total_answered == 0):
        accuracy = 0
    else: 
        accuracy = round(correct_guesses / total_answered, 2)

    fields = [modelstr, len(dict), str(correct_guesses)+',', str(unattempted_guesses)+',', accuracy]
    filename = 'analysis.csv'
    f = open(filename, 'a', newline='')
    writer = csv.writer(f)
    row = fields
    writer.writerow(row)
    f.close()


print('--------------------------Welcome To MP3--------------------------')
print('                          by Team ORANGES                          ')

f = open('analysis.csv', 'w', newline='')
f.write('model-name,size-of-vocab,number-of-correct-labels,answers-without-guessing,accuracy')
df = pd.read_csv("sample_40.csv")


#Task 1 q1, q2
task_one_and_two('word2vec-google-news-300', df)

#Task 2 q1: new models different corpora and same emb size
task_one_and_two('glove-twitter-100', df)
task_one_and_two('glove-wiki-gigaword-100', df)

#Task 2 q2: new models same corpora but dif emb size
task_one_and_two('glove-twitter-25', df)
task_one_and_two('glove-twitter-50', df)

#Graphing Results and Analysis
df = pd.read_csv('analysis.csv')
temp = {'model-name': 'human-gold-standard', 'accuracy': 0.8557}
df = df.append(temp, ignore_index=True)
baseline = df["accuracy"].mean()
temp = {'model-name': 'random-baseline', 'accuracy': baseline}
df = df.append(temp, ignore_index=True)
df.plot(kind='bar' , x='model-name', y='accuracy')
plt.tight_layout()
plt.savefig("accuracy.pdf", dpi = 100)