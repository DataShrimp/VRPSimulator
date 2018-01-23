import requests
import random
import json

def get_batch_response(data, command=""):
    url = 'http://127.0.0.1:6789/' + command
    r = requests.post(url, data)

    ret = None
    if r.ok:
        ret = json.loads(r.text)
    else:
        print(r.raise_for_status())

    return ret

if __name__ == "__test__":
    for i in range(10000):
        n = 10
        x = [x for x in range(1, n)]
        random.shuffle(x)
        x.append(0)
        x.insert(0, 0)
        data = '{"n":' + str(n) + ', "action":' + str(x) + '}'
        r = get_batch_response(data)
        if r and r['reward'] < 0.5:
            # 输出监督训练数据集
            print(r['city'])
    print('Done')

if __name__ == "__main__":
    data = '{"n":10,"action":0}'
    r = get_batch_response(data, "run")
    print(r)
