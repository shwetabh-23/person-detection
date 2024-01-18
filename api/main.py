from typing import Any
import cv2
from fastapi import APIRouter, FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

import numpy as np
# from app.api import api_router
# from app.config import settings, setup_app_logging

# setup logging as early as possible

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

root_router = APIRouter()

@root_router.get("/")
def index(request: Request) -> Any:
    return HTMLResponse(content=open("index.html", "r").read(), status_code=200)

@root_router.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    
    try:
        return HTMLResponse(content=open('upload_pic.html', 'r').read(), status_code=200)
        # file_content = await file.read()
        # image = process_image(file_content)
        # return image
    except Exception as e:
        return {"error": str(e)}

# @root_router.post('/find-person/')
# async def find_person(image, name):

app.include_router(root_router)

def process_image(file_content):
    image = np.frombuffer(file_content, dtype= np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image
    #breakpoint


if __name__ == "__main__":
    # Use this for debugging purposes only
    import uvicorn

    uvicorn.run(app, host="localhost", port=8001)
