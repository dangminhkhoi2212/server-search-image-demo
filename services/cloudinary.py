import logging
import cloudinary
import cloudinary.uploader
import os
from dotenv import load_dotenv
load_dotenv()

CLOUDINARY_NAME = os.getenv('CLOUDINARY_NAME')
CLOUDINARY_API_KEY = os.getenv('CLOUDINARY_API_KEY')
CLOUDINARY_API_SECRET = os.getenv('CLOUDINARY_API_SECRET')
cloudinary.config(
    cloud_name=CLOUDINARY_NAME,
    api_key=CLOUDINARY_API_KEY,
    api_secret=CLOUDINARY_API_SECRET
)


def upload_to_cloudinary(image_path):
    """Upload an image to Cloudinary and return its URL."""
    folder = 'qdrant_images'
    try:
        response = cloudinary.uploader.upload(
            image_path,  # Path to the image file
            folder=folder  # Specify the folder on Cloudinary
        )
        # Get the secure URL of the uploaded image
        return response.get("secure_url")
    except Exception as e:
        logging.error(f"Error uploading image to Cloudinary: {e}")
        return None
