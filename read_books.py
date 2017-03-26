import glob
import numpy as np

filePath='./books'
fileCounter = len(glob.glob(filePath+"*.txt"))


paragraphs=[]
topics=[]
for index,filename in enumerate(glob.glob(filePath+'/*.txt')):
	print(filename)
	file = open(filename,'r')
	txt= file.read()
	txt=txt.replace('\n','').split('.')
	txt=filter(None,txt)
	for i in np.arange(0,len(txt)):
		topics.append('book:%d sentence:%d'%(index,i))
		paragraphs.append(txt[i])

print len(topics)
print len(paragraphs)
print topics
print paragraphs[2]