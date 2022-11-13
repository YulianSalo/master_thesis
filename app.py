from flask import Flask
import os

UPLOAD_FOLDER = f'{os.getcwd()}/static/uploads/'
DOWNLOAD_FOLDER = f'{os.getcwd()}/static/downloads/'

# UPLOAD_FOLDER = f'static/uploads/'
# DOWNLOAD_FOLDER = f'static/downloads/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 8096 * 8096