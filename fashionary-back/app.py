import ast
import json
from flask import request
from flask import Flask , render_template , url_for , redirect , session
from forms import TextSearchForm 
import clip_text_image_search
import image_preprocessing
from flask_cors import CORS , cross_origin

app = Flask(__name__)
CORS(app, supports_credentials=True)
app.config['SECRET_KEY'] = 'a6bd69faba671892e7dd05c20f20c6b2'

posts = [
    {
        'author' : 'abcd',
        'title' : 'why not',
        'content' : 'bla bla bla bla bla blabla',
        'date' : '1.1.1'
    },
    {
        'author' : 'jjjj',
        'title' : 'ohhh teh',
        'content' : 'CHa  CHa CHa CHa CHa',
        'date' : '12.12.12'
    }
]

card = [1,2,3,4,5,6,7,8,9,10 ,11,12,13,14,15]

@app.route("/")
def hello_world():
    return render_template('home.html'  ,posts=posts)

@app.route("/store")
def store():
    # form = TextSearchForm()
    return render_template('store.html' , card=card) 


@app.route('/search' , methods=['GET' , 'POST'])
@cross_origin()
def search():
    print(" request : " ,request.json)
    req_json = request.json
    dict = ast.literal_eval(req_json)
    query = dict['query']
    print("query from search" , query)
    response = clip_text_image_search.get_pothos(query)
    print(" response : " ,response)
    response['headers']["Access-Control-Allow-Origin"] = "*"
    return response

@app.route('/similar' , methods=['GET' , 'POST'])
@cross_origin()
def similar():
    print(" query : " ,request.json)
    req_json = request.json
    dict = ast.literal_eval(req_json)
    query = dict['query']
    print("query :::: " , type(query))
    response = clip_text_image_search.get_pothos(query)
    print(" response : " ,response)
    response['headers']["Access-Control-Allow-Origin"] = "*"
    return response

@app.route('/concept' , methods=['GET' , 'POST'])
@cross_origin()
def concept():
    print(" query : " ,request.json)
    req_json = request.json
    dict = ast.literal_eval(req_json)
    query = dict['query']
    concept = dict['concept']
    print("quuuerrrryyy :::: " , type(query) , query)
    print("concept :::: " , type(concept) , concept)
    response = clip_text_image_search.get_concept(query , concept)
    print(" response : " ,response)
    response['headers']["Access-Control-Allow-Origin"] = "*"
    return response

# @app.route('/search/id/<id>' , methods=['GET' , 'POST'])  # 'GET' is the default method, you don't need to mention it explicitly
# def query(query):
    
#     # image_paths , image_ids = clip_text_image_search.get_pothos(query)
#     # new_image_paths , new_image_ids = clip_text_image_search.get_pothos(query)
#     return query
# @app.route('/search' , methods=['POST'])
# def concept():
#     if request.method == 'POST':
#         new = request.form['search']
#     print("new : " , new)
#     print(" old : " , session["query"])
#     query = session["query"] + str(new)
#     return redirect(url_for('search', query=query))
# @app.route('/drop', methods=['POST'])
# def delete_images():
#     return redirect(url_for('drop'))

if __name__ == "__main__":
    app.run(debug=True)




