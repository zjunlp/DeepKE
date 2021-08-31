from bert import Ner
model = Ner("out_ner/")

text= "Irene, a master student in Zhejiang University, Hangzhou, is traveling in Warsaw for Chopin Music Festival."
print("Text to predict Entity:")
print(text)
print('Results of NER:')

result = model.predict(text)
for k,v in result.items():
    if v:
        print(v,end=': ')
        if k=='PER':
            print('Person')
        elif k=='LOC':
            print('Location')
        elif k=='ORG':
            print('Organization')
        elif k=='MISC':
            print('Miscellaneous')
