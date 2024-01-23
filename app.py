from fastapi import FastAPI, UploadFile, HTTPException, File
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io
from functions import get_name_image, add_new_user
from typing import List
import os
import numpy as np
import json

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

def process_images(image1: Image.Image, image2: Image.Image) -> Image.Image:
    
    add_new_user(image=image1)
    get_name_image(img= image2)

def make_average_embeddings():

    avg_emb = 0
    for emb_path in os.listdir(r'generated_embeddings/'):
        emb = np.load(os.path.join(r'generated_embeddings', emb_path))
        avg_emb += emb
    avg_emb = avg_emb / len(os.listdir(r'generated_embeddings'))
    return avg_emb

def create_images_json(folder_path):
    image_files = os.listdir(folder_path)
    image_files = ['static/' + i for i in image_files if i.endswith('.jpg')]
    json_content = json.dumps(image_files, indent=2)

    with open(os.path.join(folder_path, 'images.json'), 'w') as json_file:
        json_file.write(json_content)

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(content=open("static/index2.html").read(), status_code=200)

@app.post("/process_images")
async def process_images_route(files1: List[UploadFile] = File(...), files2: List[UploadFile] = File(...)):
    try:
        empty_image_directory()
        # Process the list of images
        for i, file in enumerate(files1):
            img = await file.read()
            img = Image.open(io.BytesIO(img))

            add_new_user(img, f'image_embedding-{i}') 

        avg_emb = make_average_embeddings()
        
        np.save(r'generated_embeddings/curr_embeddings.npy', avg_emb)
        
        
        for i, file in enumerate(files2):
            img = await file.read()
            img = Image.open(io.BytesIO(img))
            get_name_image(img, image_name= f'generated_image_{i}.jpg')
        
        for i in os.listdir(r'generated_embeddings'):
            os.remove(f'generated_embeddings/{i}')

        create_images_json(r'static/')
        
        return HTMLResponse(content=open('static/display.html').read(), status_code=200)
        
    except Exception as e:
        # Handle exceptions appropriately
        raise HTTPException(status_code=500, detail=str(e))

def empty_image_directory():
    images = [image for image in os.listdir(r'static/') if image.endswith('.jpg')]
    if len(images) > 0:
        for image in images:
            os.remove(f'static/{image}')

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host= 'localhost', port= 8001)
    #create_images_json(r'static/')
