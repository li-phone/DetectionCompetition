import requests
from urllib.parse import urlparse

session = requests.Session()
resp = session.get('http://jd.com')
resp.encoding = 'utf8'
print(resp.status_code)
urlParam = urlparse(resp.url)
print(resp.text)
'dhu.edu.cn' in urlParam.netloc
data = {
    'username': '141140117',
    'password': 'Lifeng.141140117'
}
postUrl = urlParam.scheme + '://' + urlParam.netloc + '//post.php'
resp = session.post(postUrl, data)
resp.encoding = 'utf8'
print(resp.text)
