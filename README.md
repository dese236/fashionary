# fashionary
## Background :
The application is using 2 servers, nodeJS(using React) for the frontend running on port 3000,and Python Flask in the backend Running on port 5000.
The servers are running independetly , and they need to be awaked seperatly.
The communiction between the servers is preformed by RestAPI calls.

## Clone the project to yout pc !!!

## Using the app:
In order to use the app you need to Downloaf the features.npy file from :
 <Link>
and add it to the following Diractory:
fashionary/fashionary-back/static.

## Running the App :

-Frontend :
from the main directory cmd(or IDE terminal) run the following commands.
    1. $cd fashionary-front
    2. $npm i
    3. $npm start

-Backend
from the main directory cmd(or IDE terminal) run the following commands.
    1. $cd fashionary-back
    2. make sure you have the python libraries installed on your pc.
    $pip install <library_name> , if you need to install any of the python libraries.
    3. $python app.py 

Before using the UI make sure both servers are up.

## Date Processing :
-you can see the processing in the data_pocessing.py file located in :
fashionary/fashionary-back

## Image Processing :
-you can see the Image processing in the image_preprocessing.py file located in :
fashionary/fashionary-back

## image search functionality :
- Search methods are preformed in clip_text_image_search.py file located in :
fashionary/fashionary-back

## Flask routing :
- Flask app is getting RestApi requests from nodeJS and the routing is defined in app.py file.