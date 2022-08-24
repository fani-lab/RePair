import requests
import json
from bs4 import BeautifulSoup

# http://web.archive.org/web/20220814194236/https://www.westchestergov.com/
r =requests.get('http://archive.org/wayback/available?url=https://www.westchestergov.com/')
# soup = BeautifulSoup(r.content, 'html.parser')
# print(soup.title.string)

# json_convert = json.loads(r.content)
# print(json_convert["archived_snapshots"]["closest"]["url"])
# print(BeautifulSoup(requests.get(json_convert["archived_snapshots"]["closest"]["url"]).content,'html.parser').title.string)

class AOL_Extract:
    def __init__(self,url,title,item_rank):
        self.url = url
        self.title = title
        self.item_rank = item_rank

