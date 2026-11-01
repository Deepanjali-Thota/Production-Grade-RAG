import os
from dotenv import load_dotenv
import boto3

load_dotenv()

BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
LOCAL_DIR = "data/production_baseline"

def list_and_download_s3_docs():
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )
    
    response = s3.list_objects_v2(Bucket=BUCKET_NAME)
    
    if "Contents" not in response:
        print("No files found in the bucket.")
        return

    os.makedirs(LOCAL_DIR, exist_ok=True)
    
    print(f"Connected to S3 bucket '{BUCKET_NAME}'. Found files:")
    for obj in response["Contents"]:
        key = obj["Key"]
        print(f" - {key}")
        
        if key.endswith(".pdf"):
            local_path = os.path.join(LOCAL_DIR, os.path.basename(key))
            if not os.path.exists(local_path):
                print(f"   Downloading {key} -> {local_path}")
                s3.download_file(BUCKET_NAME, key, local_path)

if __name__ == "__main__":
    list_and_download_s3_docs()

