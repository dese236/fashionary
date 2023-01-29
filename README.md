# Fashionary - Personalied visual browsing

## Background :
Welcome to Fashionary! This is the go-to place for all latest fashion trend :)
This project is a final project we created during our studies in the course:
Intelligent Interactive System at the Technion.
In this project you can find a new and improved personalized browsing experience using CLIP model.
ENJOY!

The application is using 2 servers: 
* nodeJS (using React) for the frontend running on port 3000.
* Python Flask in the backend Running on port 5000.

The servers are running independetly, they need to be awaked seperatly.
The communiction between the servers is preformed by RestAPI calls.

## Clone the project to yout pc !!!

## Using the app:
In order to use the app you need to do the following steps:
1. Download the features_clip.npy file from: <https://drive.google.com/drive/u/1/folders/1A3Yh6eCY5ziFy4DrC2e5Nz-HIyA4skfY>
2. Add this file to the following Diractory: fashionary/fashionary-back/static.
At the same link you can find a short video that shows how to run the project and use the system.

## Running the App :

-Frontend :
From the main directory cmd (or IDE terminal) run the following commands.
1. $cd fashionary-front
2. $npm i
3. $npm start

-Backend
From the main directory cmd (or IDE terminal) run the following commands.
1. $cd fashionary-back 
2. make sure you have the python libraries installed on your pc.
    $pip install <library_name> , if you need to install any of the python libraries.
3. $python app.py 

Before using the UI make sure both servers are up.

## Date Processing :
- You can see the processing in the data_pocessing.py file located in :
  fashionary/fashionary-back

## Image Processing :
- You can see the Image processing in the image_preprocessing.py file located in:
  fashionary/fashionary-back

## Image search functionality:
- Search methods are preformed in clip_text_image_search.py file located in:
  fashionary/fashionary-back

## Flask routing:
- Flask app is getting RestApi requests from nodeJS and the routing is defined in app.py file.
