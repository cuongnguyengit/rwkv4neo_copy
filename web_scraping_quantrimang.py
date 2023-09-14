import requests
import pandas as pd
from bs4 import BeautifulSoup
from bs4 import BeautifulSoup
import re
import urllib3
from random import randint

from urllib3 import request
import time
from urllib.parse import urlparse

dict_links = {}

def parse_main_url(url='http://www.example.test/foo/bar'):
    domain = urlparse(url).netloc
    return domain
# Disable all kinds of warnings
urllib3.disable_warnings()
import ssl

regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"

def check_url(url):
    url = str(url)
    if url.startswith("http") and not url.endswith(".png") and not url.endswith(".jpg") and url not in dict_links and len(url) <= 199:
        if "lang=" in url and "lang=vi" in url:
            return True
        elif "lang=" in url:
            return False
        else:
            return True
    return False

# Avoid SSL Certificate to access the HTTP website
ssl._create_default_https_context = ssl._create_unverified_context

USER_AGENTS = [
    "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; AcooBrowser; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0; Acoo Browser; SLCC1; .NET CLR 2.0.50727; Media Center PC 5.0; .NET CLR 3.0.04506)",
    "Mozilla/4.0 (compatible; MSIE 7.0; AOL 9.5; AOLBuild 4337.35; Windows NT 5.1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
    "Mozilla/5.0 (Windows; U; MSIE 9.0; Windows NT 9.0; en-US)",
    "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Win64; x64; Trident/5.0; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 2.0.50727; Media Center PC 6.0)",
    "Mozilla/5.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0; WOW64; Trident/4.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 1.0.3705; .NET CLR 1.1.4322)",
    "Mozilla/4.0 (compatible; MSIE 7.0b; Windows NT 5.2; .NET CLR 1.1.4322; .NET CLR 2.0.50727; InfoPath.2; .NET CLR 3.0.04506.30)",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN) AppleWebKit/523.15 (KHTML, like Gecko, Safari/419.3) Arora/0.3 (Change: 287 c9dfb30)",
    "Mozilla/5.0 (X11; U; Linux; en-US) AppleWebKit/527+ (KHTML, like Gecko, Safari/419.3) Arora/0.6",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.2pre) Gecko/20070215 K-Ninja/2.1.1",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN; rv:1.9) Gecko/20080705 Firefox/3.0 Kapiko/3.0",
    "Mozilla/5.0 (X11; Linux i686; U;) Gecko/20070322 Kazehakase/0.4.5",
    "Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.8) Gecko Fedora/1.9.0.8-1.fc10 Kazehakase/0.5.6",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_3) AppleWebKit/535.20 (KHTML, like Gecko) Chrome/19.0.1036.7 Safari/535.20",
    "Opera/9.80 (Macintosh; Intel Mac OS X 10.6.8; U; fr) Presto/2.9.168 Version/11.52",
]

def getdata(url, retry=3):
    random_agent = USER_AGENTS[randint(0, len(USER_AGENTS) - 1)]
    headers = {
        'User-Agent': random_agent,
        "accept-charset": "UTF-8"
    }
    for i in range(retry):
        try:
            r = requests.get(url, headers=headers, timeout=(50, 60))
            rec = r.encoding
            return bytes(r.text, rec).decode('utf-8')
            # return r.text
        except Exception as e:
            print(e)
    return ""


def parse_web(url="https://www.geeksforgeeks.org/"):
    htmldata = getdata(url)
    soup = BeautifulSoup(htmldata, 'html.parser')
    out = ''
    last = ''
    for data in soup.find_all("p"):
        try:
            tmp = re.sub(r'\s+', data.get_text(), ' ').strip()
            if len(tmp) < 10 or (last and last == tmp):
                continue
        except:
            continue
        out += tmp + "\n"
        last = tmp
    links = []
    rurl = parse_main_url(url)
    for link in soup.findAll('a'):
        link = href = link.get('href')

        # if href and check_url(link) and parse_main_url(url) in link:
        #     links.append(link)

        if href:
            link = rurl + href if rurl not in link else href
            if not str(link).startswith("http"):
                link = "https://" + link
            if check_url(link) and href != "/":
                links.append(link)

    return out, links


web = "https://quantrimang.com/"

list_links = [
    web
]

# txt = open("G:\Project\Develop\DATA\TEXT/telecom_web/telecom_website.txt", 'w', encoding='utf-8')
txt = open("../data/quantrimang.txt", 'a', encoding='utf-8')

while len(list_links) > 0:
    url = list_links.pop(0)
    if url in dict_links:
        print(f"Url={url} was scraped")
        continue
    if not str(url).startswith(web):
        # wtxt.write(f"{url}\n")
        continue
    print(f"Parse url={url}")
    out, links = parse_web(url)
    # print(out)
    dict_links[url] = 1
    print(f"\tText Size: {len(out)}, links len: {len(links)}, all links len: {len(list_links)}, all scraped links: {len(dict_links)}")
    # print(f'\tNext 5 link:')
    # for link in list_links[:5]:
    #     print(f"\t\t{link}")
    print("================================ ================================== ============================")
    if len(out) > 500 or str(url).endswith(".html") or str(url).endswith(".htm"):
        try:
            txt.write(out.strip() + "\n\n\n\n")
        except:
            pass

    for link in links:
        if link not in list_links and link not in dict_links and str(link).startswith(web) and 'lich-am' not in str(link):
            list_links.append(link)
