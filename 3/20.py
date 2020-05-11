import json
import codecs

for row in codecs.open('3\jawiki-country.json', encoding='utf8'):
    article_dict = json.loads(row)
    if article_dict['title']=='イギリス':
        print(article_dict['text'])
