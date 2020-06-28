import json
import sys
import re
from urllib import request, parse

sys.path.append('./3')
from q25 import baseinf

baseinf = baseinf()
flag = baseinf['国旗画像']
url = 'https://commons.wikimedia.org/w/api.php'
params = {'action': 'query', 'prop': 'imageinfo', 'iiprop': 'url',
            'format': 'json', 'titles': f'File:{flag}'}

req = request.Request(f'{url}?{parse.urlencode(params)}')
with request.urlopen(req) as res:
    body = res.read()

print(re.search('"url":"(.+?)"', body.decode()).group(1))
