import json
import time
import os
import requests
from bs4 import BeautifulSoup

seed_urls = [
    "https://wiki.sportskeeda.com/naruto/jutsu",
    "https://wiki.sportskeeda.com/naruto/characters",
    "https://wiki.sportskeeda.com/naruto/teams",
    "https://wiki.sportskeeda.com/naruto/clans",
    "https://wiki.sportskeeda.com/naruto/all-pages",
]
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36"
}
page_tag = "wiki-tag-pages"

output_path = "data/sportseeker/"
os.makedirs(output_path, exist_ok=True)

with open(os.path.join(output_path, "seeds.json"), "w") as f:
    for url in seed_urls:
        tag = url.split("/")[-1]
        html = requests.get(url, headers=headers)
        soup = BeautifulSoup(html.text, "lxml")
        # Find all links under div id wiki-tag-pages
        links = soup.select("#wiki-tag-pages ul li a")

        for link in links:
            tmp = {
                "link": link.attrs["href"].strip(),
                "tag": tag,
            }
            f.write(json.dumps(tmp) + "\n")

        time.sleep(1)
