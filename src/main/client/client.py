import requests
import json

url = 'http://127.0.0.1:6789/start'
data = '{"n":10, "action":[0,1,3,4,2,9,8,7,6,5,0]}'
r = requests.post(url, data)

if r.ok:
    print(r.text)
else:
    print(r.raise_for_status())

