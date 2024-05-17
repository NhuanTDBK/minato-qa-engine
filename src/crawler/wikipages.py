import json
import requests

base_url = "https://naruto.fandom.com/api.php"
params = {"action": "query", "list": "allpages", "aplimit": 500, "format": "json"}

all_pages = []

while True:
    response = requests.get(base_url, params=params).json()
    all_pages.extend(page["title"] for page in response["query"]["allpages"])

    print("Fetched", len(all_pages), "pages")

    if "continue" in response:
        params.update(response["continue"])
    else:
        break

json.dump(all_pages, open("all_pages.json", "w"), indent=4)
