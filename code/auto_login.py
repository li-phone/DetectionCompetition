from selenium import webdriver
import time
import os
import requests
import base64


def auto_login(actions, url='https://portalwy.dhu.edu.cn/portalcloud/page/1/PC/chn/Login.html', decode='base64'):
    driver = webdriver.Chrome()
    driver.get(url)
    if str(driver.title).find('百度') != -1:
        return True
    time.sleep(6)
    for act in actions:
        if act['action'] == 'send_keys':
            act_input = act['input']
            if decode == 'base64':
                act_input = base64.b64decode(act_input)
                act_input = str(act_input, 'utf-8')
            driver.find_element_by_id(act['id']).send_keys(act_input)
        elif act['action'] == 'click':
            driver.find_element_by_id(act['id']).click()
    time.sleep(3)
    driver.close()
    return False


actions = [
    dict(id='userphone', input='MjE4MTc2Mg==', action='send_keys'),
    dict(id='password', input='TGlmZW5nLjE0MTE0MDExNw==', action='send_keys'),
    dict(id='mobilelogin_submit', input=None, action='click'),
]
for i in range(10):
    try:
        ret = auto_login(actions, url='http://www.baidu.com')
        if ret:
            print('login successfully!')
            break
    except:
        print('login error!')
