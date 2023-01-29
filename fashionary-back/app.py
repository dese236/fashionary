import ast
import json
from flask import request
from flask import Flask , render_template , url_for , redirect , session
import clip_text_image_search
from flask_cors import CORS , cross_origin

app = Flask(__name__)
CORS(app, supports_credentials=True)
app.config['SECRET_KEY'] = 'a6bd69faba671892e7dd05c20f20c6b2'

@app.route('/search' , methods=['GET' , 'POST'])
@cross_origin()
def search():
    req_json = request.json
    dict = ast.literal_eval(req_json)
    query = dict['query']
    response = clip_text_image_search.get_images(query)
    response['headers']["Access-Control-Allow-Origin"] = "*"
    return response

@app.route('/similar' , methods=['GET' , 'POST'])
@cross_origin()
def similar():
    req_json = request.json
    dict = ast.literal_eval(req_json)
    query = dict['query']
    response = clip_text_image_search.get_images(query)
    response['headers']["Access-Control-Allow-Origin"] = "*"
    return response

@app.route('/concept' , methods=['GET' , 'POST'])
@cross_origin()
def concept():
    req_json = request.json
    dict = ast.literal_eval(req_json)
    query = dict['query']
    concept = dict['concept']
    response = clip_text_image_search.get_concept(query , concept)
    response['headers']["Access-Control-Allow-Origin"] = "*"
    return response

if __name__ == "__main__":
    app.run(debug=True)




