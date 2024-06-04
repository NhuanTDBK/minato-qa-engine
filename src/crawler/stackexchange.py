import os
import requests
import io
from tqdm import tqdm

links = [
    "https://archive.org/download/stackexchange_20240402_bis",
    "https://archive.org/download/stackexchange_20240402",
    "https://archive.org/download/stackexchange_20240305",
    "https://archive.org/download/stackexchange_20231208",
    "https://archive.org/download/stackexchange_20230912",
    "https://archive.org/download/stackexchange_20230614",
    "https://archive.org/download/stackexchange_20230308",
    "https://archive.org/download/stackexchange_20221206",
    "https://archive.org/download/stackexchange_20221005",
    "https://archive.org/download/stackexchange_20220606",
    "https://archive.org/download/stackexchange_20220307",
    "https://archive.org/download/stackexchange_20211206",
    "https://archive.org/download/stackexchange_20210907",
    "https://archive.org/download/stackexchange_20210607",
    "https://archive.org/download/stackexchange_20210301",
    "https://archive.org/download/StackExchange2009-2011/stackexchange-export-2011-06-01/",
    "https://archive.org/download/stackexchange-snapshot-2018-03-14",
    "https://archive.org/download/StackExchange2009-2011",
]
prefix = "anime.stackexchange.com.7z"
output_path = "data/anime_stackexchange"

for link in links:
    date = link.split("/")[-1]
    file_link = f"{link}/{prefix}"
    try:
        print("Download from", file_link)
        response = requests.get(file_link, stream=True)
        # Get file size, create a progress bar
        total_length = response.headers.get("content-length")
        pbar = tqdm(total=int(total_length), unit="B", unit_scale=True)
        with open(f"{output_path}/{date}.7z", "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    pbar.update(len(chunk))
    # Catch if link cannot accessible
    except Exception as e:
        print("Error:", e)
        continue
