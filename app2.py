from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io
from functions import get_name_image, add_new_user

app = FastAPI()

# Mount static files (e.g., stylesheets) to the "/static" path
app.mount("/static", StaticFiles(directory="static"), name="static")

def process_images(image1: Image.Image, image2: Image.Image) -> Image.Image:
    # Your image processing logic here
    # For example, you can simply concatenate two images horizontally
    add_new_user(image=image1)
    get_name_image(img= image2)

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(content=open("static/index2.html").read(), status_code=200)

@app.post("/process_images")
async def process_images_route(file1: UploadFile = UploadFile(...), file2: UploadFile = UploadFile(...)):
    try:
        # Open and process the uploaded images
        image1 = Image.open(io.BytesIO(await file1.read()))
        image2 = Image.open(io.BytesIO(await file2.read()))
        #breakpoint()
        # Process the images
        process_images(image1, image2)
        return HTMLResponse(content= open('static/display.html').read(), status_code= 200)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing images: {str(e)}")

if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host= 'localhost', port= 8001)
