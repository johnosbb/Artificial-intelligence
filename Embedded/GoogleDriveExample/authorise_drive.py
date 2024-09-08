from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import json
import os


gauth = GoogleAuth()
gauth.settings['client_config_file'] = './config/client_secrets.json'
# Try to load saved client credentials
gauth.LoadCredentialsFile(".config/credentials.txt")

if gauth.credentials is None:
    # Authenticate if they're not there
    gauth.LocalWebserverAuth()
elif gauth.access_token_expired:
    # Refresh them if expired
    gauth.Refresh()
else:
    # Initialize the saved creds
    gauth.Authorize()

# Save the current credentials to a file
gauth.SaveCredentialsFile("./config/credentials.txt")

drive = GoogleDrive(gauth)