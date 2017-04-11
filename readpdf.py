stringLines = []
with open('/home/kaushik/Downloads/SVM.txt') as f:
    content = f.readlines()
#print content
for lines in content :
    flag= True
    for character in lines :
        if not (character.isalnum() or character == ' ' or character == '.' or character == ',' or character == '\n') :
            flag = False
            break
    if flag: stringLines.append(lines.replace('\n', ''))
stringLines = filter(None,stringLines)
print stringLines