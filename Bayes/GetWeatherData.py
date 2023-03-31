import re
import csv
import time
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.support.select import Select

weather_list = []


def get_url():
    url = 'http://www.weather.com.cn/weather/101270101.shtml'
    return url


def get_urlText(url):
    try:
        kv = {'user-agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=kv)
        r.raise_for_status()
        r.encoding = r.apparent_encoding  # 使其encoding更准确
        # print (r.text)  #1
        return r.text
    except:
        print('error 1')
        return


def get_parseText(parse_url):
    try:
        soup = BeautifulSoup(parse_url, 'html.parser')
        lists = []
        lists = soup.find('ul', 't clearfix').find_all('li')
        for elem in lists:
            date = elem.find('h1').get_text()
            weather = elem.find('p', 'wea').get_text()
            temperature = elem.find('p', 'tem').find('i').get_text()
            win = re.findall('(?<= title=").*?(?=\")', str(elem.find('p', 'win').find('em')))
            # *？匹配前面那个子表达式0/1次，最小匹配  ？= 捕获以title= 开头的内容  ？=查找“前面的。
            wind = '-'.join(win)
            # print(wind)
            wind_lev = elem.find('p', 'win').find('i').get_text()
            global weather_list
            weather_list.append([date, weather, temperature, wind, wind_lev])
    except:
        print('error 2')
        return


def prints(weather_list):
    titles = ['日期', '天气', '温度', '风向', '风级']
    with open('weather.csv', 'w', encoding='utf8') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(titles)
        for row in weather_list:
            f_csv.writerow(row)


def main():
    url = get_url()
    parse_text = get_urlText(url)
    get_parseText(parse_text)
    prints(weather_list)


main()


