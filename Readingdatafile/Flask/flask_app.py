from flask import Flask
import re
import openai
import urllib
import numpy as np
from PIL import Image
from tensorflow import keras
from urllib import request
import subprocess
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import subprocess
from flask import jsonify

# set API key
openai.api_key = "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"


def generateScripts(slides):
    print("Completing...")
    # list for save generated scripts
    generatedScripts = []
    for slide in slides:
        # create the prompt, substring with regex for remove S1: S2: notation
        promptForGPT = "Write a presentation script->\\n" + \
            re.sub('S\d:', '', slide)
        # call openai for completion
        completion = openai.Completion.create(
            engine="text-davinci-003",
            prompt=promptForGPT,
            temperature=0.7,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            max_tokens=200)
        # append to generated scripts list
        generatedScripts.append(completion.choices[0].text)
    return generatedScripts


app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Welcome to ScriptGenAI!'


@app.route('/generate/<string:fileName>/<string:accessToken>', methods=['GET'])
def generateAll(fileName, accessToken):
    in_file = urllib.request.urlopen(
        "https://firebasestorage.googleapis.com/v0/b/sdgp-squadr.appspot.com/o/files%2F" + fileName + "?alt=media&token=" + accessToken)
    content = in_file.read().decode()
    # split into slides
    slides = content.split('<[[[start]]]>')
    slides = list(filter(None, slides))
    generatedScripts = generateScripts(slides)
    return ',\n'.join(generatedScripts)


@app.route('/predict', methods=['GET'])
def predict():
    result = subprocess.run(["python", "ReadGraphOCR.py"], capture_output=True, text=True)
    output = result.stdout
    output = output.replace(",", "\t")  # replace comma separator with tab separator
    output = output.replace("  ", "\t")  # replace double spaces with tab separator
    output = output.replace("\n\n", "\n")  # remove extra line breaks
    output = output.replace("\n", "<br>")  # replace line breaks with HTML line breaks
    return output



'''
@app.route('/predict', methods=['GET'])
def predict():
    graphInfo = ReadGraphOCR.readGraph('1420.png')
    return graphInfo
'''

@app.route('/check')
def check():
    URL = "https://firebasestorage.googleapis.com/v0/b/sdgp-squadr.appspot.com/o/files%2Fsample_slide_deck.txt?alt=media&token=208755ae-61ca-4f52-9559-10c84d80d9ca"
    response = request.urlretrieve(URL, "presentation.pptx")
    return "Noice!"


if __name__ == "__flask_app__":
    app.run(debug=True)
