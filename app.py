from flask import flask, render_template, jsonify, request
import AIRI_processor
app=flask(__name__)
app.config['SECRETKEY']='milenka'
@app.route('/',methods=["GET","POST"])
def index():
    return render_template('index.html',**locals())

if __name__=='__main__':
    app.run(host='0.0.0.0',port='8888',debug=True)