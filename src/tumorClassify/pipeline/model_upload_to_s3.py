import os
import boto3
from dotenv import load_dotenv
from pathlib import Path
#import logging
from tumorClassify import logger as logging

# Set up basic configuration for logging
#logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')

# Load environment variables from .env file
load_dotenv()

# Check and retrieve environment variables
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION')

if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION]):
    logging.error("Missing one or more environment variables: 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_REGION'")
    exit(1)

# Establish connection to S3
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

def check_bucket_exists(bucket_name):
    """Check if an S3 bucket exists and create if it does not exist"""
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        logging.info(f"Bucket {bucket_name} exists.")
    except boto3.exceptions.botocore.exceptions.ClientError as e:
        error_code = int(e.response['Error']['Code'])
        if error_code == 404:
            logging.info(f"Bucket {bucket_name} does not exist. Creating bucket.")
            create_bucket(bucket_name)

def create_bucket(bucket_name):
    """Create an S3 bucket in a specified region"""
    try:
        if AWS_REGION == 'us-east-1':  # The us-east-1 region does not require a location parameter
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            s3_client.create_bucket(Bucket=bucket_name,
                                    CreateBucketConfiguration={'LocationConstraint': AWS_REGION})
        logging.info(f"Bucket {bucket_name} created successfully.")
    except boto3.exceptions.botocore.exceptions.ClientError as e:
        logging.error(f"Failed to create bucket: {e}")

def upload_file_to_s3(file_path, bucket_name, object_name=None):
    """Upload a file to an S3 bucket

    :param file_path: File to upload
    :param bucket_name: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_path is used
    """
    if object_name is None:
        object_name = file_path.name

    try:
        check_bucket_exists(bucket_name)
        response = s3_client.upload_file(
            Filename=str(file_path),
            Bucket=bucket_name,
            Key=object_name
        )
        logging.info(f"File {file_path} uploaded to {bucket_name} as {object_name}")
    except Exception as e:
        logging.error(f"Upload failed: {e}")

# Path to the model file
checkpoint_dir = Path("artifacts") / "models"
model_checkpoint_path = checkpoint_dir / "model.h5"

# Name of your S3 bucket
bucket_name = 'cancerapp-203918887737'

# Upload the file
upload_file_to_s3(model_checkpoint_path, bucket_name)

