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

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
@retry(stop_max_attempt_number=10)
@RateLimiter(max_calls=1200, period=60)
def generate_visual_rag_answer_chatgpt(system_prompt, utt, img_paths, model="gpt4o"):
    if model == "gpt4o": # -mini":
        model = "gpt-4o-2024-08-06"
    content=[
                {
                    "type": "text",
                    "text": utt,
                }
            ]
    for i, img_path in enumerate(img_paths):
        base64_image = encode_image(img_path)
        content += [{
                        "type": "text",
                        "text": 'Image {}:\n'.format(i + 1),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "quality": "high"
                        },
                    }]
    if system_prompt is not None:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": content,
                }
            ],
            max_tokens=1000,
        )
    else:
        response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
        max_tokens=1000,
    )
    output = response.choices[0].message.content
            
    print(output)
    return output

@retry(stop_max_attempt_number=10)
@RateLimiter(max_calls=1200, period=60)
def generate_chatgpt_text(utt):
    response = client.chat.completions.create(
        # model="gpt-3.5-turbo-0125",
        model="gpt-4o-mini", # "gpt-4o-2024-05-13",
        messages=[
            {'role': 'system', 'content': "You are a helpful assistant designed to output JSON that answers the following question with proper reference to the provided documents. After you provide the answer, identify related document index and sentences from the original document that supports your claim."},
            {"role": "user", "content": utt}
        ]
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content

@retry(stop_max_attempt_number=10)
@RateLimiter(max_calls=1200, period=60)
def generate_chatgpt_original(utt, model='gpt4o'):
    if model == 'gpt4o':
        model = "gpt-4o-2024-08-06" 
    # print(utt)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": utt}
        ]
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content


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
