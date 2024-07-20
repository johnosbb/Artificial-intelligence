from google.cloud import storage

# Initialize a client
client = storage.Client()

# Get the bucket
bucket = client.bucket('trax-ml')

# Get the blob (file)
blob = bucket.blob('models/translation/ende_wmt32k.pkl.gz')

# Download the file
blob.download_to_filename('ende_wmt32k.pkl.gz')

print("File downloaded successfully.")
