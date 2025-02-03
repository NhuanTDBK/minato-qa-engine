import os
import json
from tqdm import trange
from datetime import timedelta, datetime
from mediawiki import MediaWiki, exceptions

from src.writer.json import JsonWriter

mediawiki = MediaWiki(
    "https://naruto.fandom.com/api.php",
    rate_limit=True,
    rate_limit_wait=timedelta(milliseconds=50),
)

# script to fetch all pages from the wiki, based on json file all_pages.json
# it will have rate limit and long time running
# Each 1000 rows, create a json file to write the content of the pages
# Error when quering page
#  mediawiki.exceptions.PageError – if page provided does not exist
# mediawiki.exceptions.DisambiguationError – if page provided is a disambiguation page
# mediawiki.exceptions.RedirectError – if redirect is False and the pageid or title provided redirects to another page

categories = json.load(open("all_pages.json"))
all_content = []
data_path = "data/{}".format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
os.makedirs(data_path, exist_ok=True)
checkpoint_idx = 0
if os.path.exists("checkpoint.txt"):
    checkpoint_idx = int(open("checkpoint.txt").read())
    print(f"Checkpoint at {checkpoint_idx}")
# Checkpoint when coming back

# Every time crawling, the content is append into file, file will be rotated every 1000 rows

with JsonWriter(data_path, rotate=1000) as writer:
    for i in trange(checkpoint_idx, len(categories), 1, initial=checkpoint_idx):
        try:
            category = categories[i]
            page = mediawiki.page(title=category)
            writer.write({"title": category, "wikitext": page.wikitext})
            with open("checkpoint.txt", "w") as f:
                f.write(str(i))
        except exceptions.PageError as e:
            print(f"PageError: {category}")
        except exceptions.DisambiguationError as e:
            print(f"DisambiguationError: {category}")
        except exceptions.RedirectError as e:
            print(f"RedirectError: {category}")
