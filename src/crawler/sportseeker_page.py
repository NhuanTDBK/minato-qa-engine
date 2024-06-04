import json
import time
import os
import requests

from tqdm import tqdm
from bs4 import BeautifulSoup

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36"
}
page_tag = "wiki-tag-pages"

output_path = "data/sportseeker/"
os.makedirs(output_path, exist_ok=True)

page_objs = []
with open(os.path.join(output_path, "seeds.json")) as f:
    for line in f:
        page_objs.append(json.loads(line.strip()))


# Preload all pages
all_crawled_pages = set()
try:
    with open(os.path.join(output_path, "all_pages.json")) as f:
        for line in f:
            tmp = json.loads(line.strip())
            all_crawled_pages.add(tmp["link"])
except FileNotFoundError:
    pass


page_objs = [
    page_obj for page_obj in page_objs if page_obj["link"] not in all_crawled_pages
]
data = []
with open(os.path.join(output_path, "all_pages.json"), "a") as f:
    for page_obj in tqdm(page_objs):
        for _ in range(3):
            try:
                url = page_obj["link"]
                html = requests.get(url, headers=headers)
                soup = BeautifulSoup(html.text, "lxml")
                # Find all links under div id wiki-tag-pages
                paragraphs = soup.select(".wiki-content p")
                texts = [paragraph.text.strip() for paragraph in paragraphs]
                # Remove empty paragraph
                texts = [text for text in texts if text]

                if len(texts) == 0:
                    print(paragraphs)
                assert len(texts) > 0, "Check this page " + url

                # Find all images
                imgs_el = soup.select(".wiki-content img")
                img_links = []
                # Filter images without Edit icon, youtube-cover
                for i in range(len(imgs_el)):
                    if imgs_el[i].attrs.get("alt", "") not in [
                        "Edit icon",
                        "youtube-cover",
                    ]:
                        img_links.append(imgs_el[i].attrs["src"])

                title = soup.select(".wiki-header-title h1")[0].text.strip()

                tmp = {
                    "text": ".".join(texts),
                    "img_links": img_links,
                    "title": title,
                    "tag": page_obj["tag"],
                    "link": url,
                }
                f.write(json.dumps(tmp) + "\n")
                time.sleep(1)
                break

            except Exception as e:
                print("Error: ", url, e)
                time.sleep(5)
