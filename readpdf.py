txt = []
with open('./segnet1.txt') as f:
    content = f.readlines()
#print content
for lines in content :
    flag= True
    for character in lines :
        if not (character.isalnum() or character == ' ' or character == '.' or character == ',' or character == '\n') :
            flag = False
            break
    if flag: txt.append(lines.replace('\n', ''))
txt = filter(None,txt)
txt=''.join(txt)
print txt
print(len(txt))
