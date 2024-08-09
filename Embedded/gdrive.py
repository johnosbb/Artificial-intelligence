import serial
import time
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import json
import os



def load_token_references(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
    

def get_token_reference(token_name, references):
    return references.get(token_name)


def serial_readline(obj):
    try:
        data = obj.readline()
        return data.decode("utf-8")
    except Exception as e:
        print(f"Error reading line: {e}")
        return ""


ser = serial.Serial()
ser.port = 'COM10'
ser.baudrate = 9600
try:
    ser.open()
    ser.reset_input_buffer()
except serial.SerialException as e:
    print(f"Serial error: {e}")
    ser.close()
    exit()




current_directory = os.getcwd()
print("Current Working Directory:", current_directory)


# Path to the JSON file containing token references
token_references_file = './config/tokens.json'

token_name = 'gdrive_tinyml'  
token_references = load_token_references(token_references_file)

drive_id = get_token_reference(token_name, token_references)

print(f"Drive ID: {drive_id}")



gauth = GoogleAuth()
gauth.settings['client_config_file'] = './config/client_secrets.json'
# Try to load saved client credentials
gauth.LoadCredentialsFile("./config/credentials.txt")

if gauth.credentials is None:
    # Authenticate if they're not there
    gauth.LocalWebserverAuth()
elif gauth.access_token_expired:
    # Refresh them if expired
    gauth.Refresh()
else:
    # Initialize the saved creds
    gauth.Authorize()

drive = GoogleDrive(gauth)

text = ""
start_s = time.time()
elapsed_s = 0
while(elapsed_s < 10):
    text += serial_readline(ser)
    end_s = time.time()
    elapsed_s = end_s - start_s
    print(f"start time {start_s} end time {end_s}")

full_path = "./data/test3.log"
with open(full_path, "w") as text_file:
    text_file.write(text)


filename = os.path.basename(full_path)

# Create the file metadata for Google Drive
gfile = drive.CreateFile({'parents': [{'id': drive_id}], 'title': filename})
gfile.SetContentFile(full_path)
gfile.Upload()












