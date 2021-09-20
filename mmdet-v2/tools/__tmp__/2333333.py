session = requests.Session()
try:
    resp = session.get('http://1.2.3.4?cmd=redirect&arubalp=12345', timeout=1)  # 无效ip，会跳转到登录地址
    resp.encoding = 'utf8'
    print(resp.status_code)
    urlParam = urlparse(resp.url)
    print(resp.text)
    if 'dhu.edu.cn' in urlParam.netloc:
        print('开始认证')
        username = input('请输入用户名：')
        password = getpass.getpass('请输入密码：')
        data = {
            'username': username,
            'password': password
        }
        postUrl = urlParam.scheme + '://' + urlParam.netloc + '//post.php'
        resp = session.post(postUrl, data)
        resp.encoding = 'utf8'
        if '登录成功，欢迎您' in resp.text:
            print('认证成功！')
        else:
            print('认证失resp败！')
    else:
        print('已认证，无需重复认证')
except requests.exceptions.ConnectTimeout:
    print("未跳转登陆链接，请检查是否已联网")