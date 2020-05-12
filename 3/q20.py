import json
import codecs

def extract_british():
    for row in codecs.open('3\jawiki-country.json', encoding='utf8'):
        article_dict = json.loads(row)
        if article_dict['title']=='イギリス':
            return article_dict['text']

if __name__ == '__main__':
    text = extract_british()
    print(text)
