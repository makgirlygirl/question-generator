from googletrans import Translator

translator = Translator()
#%%
k=[]
j=[]
c=[]

for i in mcq_distractors:
    kor=translator.translate(i, src='en', dest='ko').text
    eng=translator.translate(kor, src='ko', dest='en').text
    k.append(eng)
for i in mcq_distractors:
    jap=translator.translate(i, src='en', dest='ja').text
    # zh_cn=translator.translate(i, src='ja', dest='zh-cn').text
    eng=translator.translate(jap, src='ja', dest='en').text
    j.append(eng)

for i in mcq_distractors:
    zh_cn=translator.translate(i, src='en', dest='zh-cn').text
    eng=translator.translate(zh_cn, src='zh-cn', dest='en').text
    c.append(eng)

# print(a)
for i in range(4):
    print(mcq_distractors[i])
    print(k[i])
    print(j[i])
    print(c[i])
    print('\n\n')
# %%
