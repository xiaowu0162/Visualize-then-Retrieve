from openai import OpenAI
from ratelimiter import RateLimiter
from retrying import retry
import urllib
import base64
from constants import OPENAI_API_KEY, ORGANIZATION


if ORGANIZATION:
    client = OpenAI(api_key=OPENAI_API_KEY, organization=ORGANIZATION)
else:
    client = OpenAI(api_key=OPENAI_API_KEY)


def save_img_from_url(url, fname):
    urllib.request.urlretrieve(url, fname)
    return


# getting Dalle-3's generation
@retry(stop_max_delay=3000, wait_fixed=1000)
@RateLimiter(max_calls=600, period=60)
def get_dalle_response(prompt, quality="standard", n=1):
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality=quality,
        n=n,
    )
    return response.data[0].url


# getting Image-1's generation
@retry(stop_max_delay=3000, wait_fixed=1000)
@RateLimiter(max_calls=600, period=60)
def get_image1_response(prompt, quality='high', n=1):
    response = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        # size="1024x1024",
        quality=quality,
        n=n,
    )
    images_bytes = []
    for img_dict in response.data:
        img_bytes = base64.b64decode(img_dict.b64_json)
        images_bytes.append(img_bytes)
    return images_bytes
