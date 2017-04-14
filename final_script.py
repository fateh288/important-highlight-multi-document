import numpy as np
import pandas as pd
import nltk
import re
import os
import sys
import codecs
from sklearn import feature_extraction
import mpld3
import scipy.cluster.hierarchy as hcluster
import glob
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from operator import itemgetter
import pdfkit

def tokenize(paragraph) :
    tokenList = [word for sentence in nltk.sent_tokenize(paragraph) for word in nltk.word_tokenize(sentence)]
    
    filteredTokens = []
    
    for token in tokenList:
        if re.search('[a-zA-Z]', token):
            filteredTokens.append(token)
    return filteredTokens

#Next step would be to remove the useless tokens. Useless tokens are raw puntuation, numeric tokens, etc.
def stem(filteredTokens) :
    stemList = [stemmer.stem(tok) for tok in filteredTokens]
    return stemList

def tokenizeAndStem(paragraph) :
    tokens = [word for sentence in nltk.sent_tokenize(paragraph) for word in nltk.word_tokenize(sentence)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):  #normal regex search
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

filePath = './books/'
for filename in glob.glob(filePath + "*.pdf"):
    os.system('pdftotext %s %s.txt'%(filename,os.path.splitext(filename)[0]))

fileCounter = len(glob.glob(filePath + "*.txt"))

paragraphs = []
topics = []
book_sentence_pairs=[]
for index, filename in enumerate(glob.glob(filePath + '*.txt')):
    print(filename)
    file = open(filename, 'r')
    txt = file.read()
    txt = txt.replace('\n', '').split('.')
    txt = filter(None, txt)
    print len(txt)
    for i in np.arange(0, len(txt)):
        topics.append('book: %d sentence: %d' % (index, i))
        book_sentence_pairs.append([index,i])
        paragraphs.append(txt[i])

stopwords = nltk.corpus.stopwords.words('english')
stemmer = SnowballStemmer("english")

reload(sys)
sys.setdefaultencoding('utf8')

tokenizedParagraphList = []
stemmedParagraphList = []
for i in paragraphs :
    tokenizedParagraph = tokenize(i)
    tokenizedParagraphList.extend(tokenizedParagraph)
    stemmedParagraphList.extend(stem(tokenizedParagraph))

vocabFrame = pd.DataFrame({'words': tokenizedParagraphList}, index = stemmedParagraphList)
print 'there are ' + str(vocabFrame.shape[0]) + ' items in vocab_frame'

#for tfidf, there are two important things, max_df & min_df;
#max_df - When building the vocabulary ignore terms that have a document frequency strictly higher than the given threshold (corpus-specific stop words).
#min_idf - When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold. This value is also called cut-off in the literature.
tfidfVectorizer = TfidfVectorizer(stop_words='english', use_idf=True, tokenizer=tokenizeAndStem, ngram_range=(1,3))
#Now fit the vectorizer to synopses
tfidf_matrix = tfidfVectorizer.fit_transform(paragraphs)
#Now we calculate the cosine similarity as follows. This will gives us the distance which wil help us in clustering in the later stage.
dist = 1 - cosine_similarity(tfidf_matrix)
#heirarchical clustering
thresh=1.05*np.average(dist)
clusters = hcluster.fclusterdata(dist, thresh, criterion="distance")

numClusters = max(clusters)

#preperation of data for further processing
clusterStatements = []
clusterIndices = []
for i in range(0,numClusters,1) :
    clusterStatements.append([])
    clusterIndices.append([])
for i in range(0, len(paragraphs), 1) :
    clusterStatements[clusters[i]-1].append(paragraphs[i])
    clusterIndices[clusters[i]-1].append(i)

#tfidf inside each of the clusters to find the most unique statement inside each of the statements
final_sentence_indices=[]
for k in range(0, numClusters, 1) :
    tfidfVectorizerCluster = TfidfVectorizer(stop_words='english', use_idf=True, tokenizer=tokenizeAndStem, ngram_range=(1,3))
    tfidfMatrixCluster = tfidfVectorizerCluster.fit_transform(clusterStatements[k])
    distCluster = 1 - cosine_similarity(tfidfMatrixCluster)
    maxClusterDistance=0
    farthestSentence=0
    for i in range(0, len(distCluster), 1) :
        temp=0
        for j in range(0, len(distCluster[i]), 1) :
            temp+=distCluster[i][j]
        farthestSentence = farthestSentence if maxClusterDistance<temp else i
        maxClusterDistance = maxClusterDistance if maxClusterDistance<temp else temp
    final_sentence_indices.append(clusterIndices[k][farthestSentence])

#sorting each of the pairs on the book number and sentence number
final_book_sentence_pairs=[]
for i in final_sentence_indices:
    final_book_sentence_pairs.append(book_sentence_pairs[i])
final_book_sentence_pairs=sorted(final_book_sentence_pairs,key=itemgetter(0,1))

#marking of the sentences, conversion to html followed by conversion to pdf
for index, filename in enumerate(glob.glob(filePath + '/*.txt')):
    file = open(filename, 'r')
    txt = file.read()
    txt = txt.replace('\n', '').split('.')
    for i,line in enumerate(final_book_sentence_pairs):
        if(final_book_sentence_pairs[i][0]==index):
            txt[line[1]]='<mark>'+txt[line[1]]+'</mark>'
    txt='.'.join(txt)
    Html_file= open(os.path.splitext(filename)[0]+'.html',"w")
    Html_file.write(txt)
    Html_file.close()
    pdfkit.from_url(os.path.splitext(filename)[0]+'.html', os.path.splitext(filename)[0]+'_marked'+'.pdf')

#new pdf's with marked sentences with 'oldname'_marked.pdf created in the books folder
