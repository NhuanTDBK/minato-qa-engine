# This script to download all images from the wiki, based on json file all_images.json
# Each image will resize to 256x256 and save to data/images folder
# we use asyncio to download images concurrently

from glob import glob
import random
import io
import os
import asyncio
from PIL import Image, UnidentifiedImageError
from aiohttp import ClientSession
from tqdm import trange
from datetime import datetime
from redis import StrictRedis, asyncio as redis_asyncio

contents = [w.strip() for w in open("notebooks/image_names.txt").readlines()]
all_images = [content for content in contents]

data_path = "data/images/{}".format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
os.makedirs(data_path, exist_ok=True)

IMAGE_SIZE = (384, 384)
CACHE_DOWNLOAD_FILES = set()

files = glob("data/images/*/*")
for file in files:
    CACHE_DOWNLOAD_FILES.add(file.split("/")[-1])


USERNAME = "Nhuantranduc@ImageScanning"
PASSWORD = "giet2oavaiaq2ehf2dt3aklneuoi8bso"
AUTHENTICATE_URL = "https://www.mediawiki.org/w/api.php"


async def get_login_token(session, username, password):
    params = {
        "action": "login",
        "lgname": username,
        "lgpassword": password,
        "format": "json",
    }
    async with session.get(AUTHENTICATE_URL, params=params) as response:
        data = await response.json()
        return data.get("login", {}).get("token")


async def authenticate(session, username, password, login_token):
    params = {
        "action": "login",
        "lgname": username,
        "lgpassword": password,
        "lgtoken": login_token,
        "format": "json",
    }
    async with session.post(AUTHENTICATE_URL, data=params) as response:
        data = await response.json()
        return data.get("login", {}).get("result") == "Success"


async def download_image(
    session: ClientSession,
    url: str,
):

    # timeout 10s
    for _ in range(3):
        try:
            img_name = url.split("/")[-1].lower()
            is_downloaded = img_name in CACHE_DOWNLOAD_FILES
            if is_downloaded:
                return

            async with session.get(url) as response:
                if response.status != 200:
                    return
                image = await response.read()
                image = Image.open(io.BytesIO(image))
                image = image.resize(IMAGE_SIZE)
                image.save(os.path.join(data_path, img_name))
                # Rate limit
                # random time sleep to avoid being banned
                random_sleep = random.randint(1, 10)
                await asyncio.sleep(random_sleep)

                return
        except asyncio.TimeoutError:
            print("Timeout: ", url)
            await asyncio.sleep(5)
            continue

        except UnidentifiedImageError as e:
            print("Error: ", url, e)
            return

        except Exception as e:
            print("Error: ", url, e)
            return


async def download_images():
    async with ClientSession(read_timeout=60, conn_timeout=60) as session:
        login_token = await get_login_token(session, USERNAME, PASSWORD)
        await authenticate(session, USERNAME, PASSWORD, login_token)
        print("get login cookie")
        batch_size = 30
        for i in trange(0, len(all_images), batch_size):
            tasks = []
            for j in range(i, min(i + batch_size, len(all_images))):
                tasks.append(download_image(session, all_images[j]))
            await asyncio.gather(*tasks)


# Start event loop
loop = asyncio.get_event_loop()
loop.run_until_complete(download_images())
loop.close()
