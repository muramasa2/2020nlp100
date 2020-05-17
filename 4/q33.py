from q30 import read_wakati

wakati_list = read_wakati()

noun_list=[]
noun_lists=[]
for sentence in wakati_list:
    flag=0
    for idx, wakati_dict in enumerate(sentence):

        if wakati_dict['pos']=='名詞':

            flag += 1
            noun_list.append(wakati_dict['surface'])
            # print(noun_list)
            if flag==3 and len(noun_list)>=2:
                # print(noun_list)
                noun_lists.append(''.join(noun_list))
                noun_list=[]
                flag=0
                break

        if flag==1:
            if wakati_dict['surface']=='の':
                noun_list.append(wakati_dict['surface'])
                # print(noun_list)
                flag += 1
        else:
            noun_list=[]
            flag=0

print(noun_lists)
