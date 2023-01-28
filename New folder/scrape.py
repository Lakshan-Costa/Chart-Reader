
import requests
'''
# Open the text file containing the URLs
with open('urls.txt', 'r') as file:
    # Read the URLs into a list
    urls = file.readlines()

count =0
# Iterate over the URLs
for url in urls:
    
    # Remove any leading or trailing whitespace
    url = url.strip()
    # Split the URL into parts
    parts = url.split('/')
    # Get the last part, which should be the file name
    filename = parts[-1]
    # Send a GET request to the URL
    response = requests.get(url)
    # Open a file with the same name and write the contents of the GET request
    open("image"+str(count)+".jpg", 'wb').write(response.content)
    count+=1
'''
from google_images_search import GoogleImagesSearch

gis = GoogleImagesSearch('API_KEY', 'CX')
gis.search_keywords("Vertical Bar charts")
gis.results_per_page = 100 #default is 100
gis.offset = 0
gis.execute()
image_urls = gis.results()

import requests
import os

folder_path = 'path/to/save/image'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

for image_url in image_urls:
    response = requests.get(image_url.url)
    file_path = os.path.join(folder_path, image_url.url.split("/")[-1])
    open(file_path, "wb").write(response.content)
