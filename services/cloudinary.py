import logging
import cloudinary
import cloudinary.uploader
cloudinary.config(
    cloud_name='dakwyskfm',
    api_key="253915897887594",
    api_secret="6cuhDK6jLdGK_vALQ6wY38zv3yA"
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
