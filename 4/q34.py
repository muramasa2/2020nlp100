from q30 import read_wakati

wakati_list = read_wakati()

noun_list=[]
noun_lists=[]
for sentence in wakati_list:
    noun_flag = False
    for idx, wakati_dict in enumerate(sentence):
        if wakati_dict['pos'] == '名詞':
            noun_list.append(wakati_dict['surface'])
            flag = True

        else:
            if len(noun_list)>=2:
                noun_lists.append(''.join(noun_list))
                flag=False
            noun_list=[]

print(noun_lists)
