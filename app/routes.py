from flask import Blueprint, request, render_template, jsonify
from models.model import classify_news

main = Blueprint('main', __name__)

@main.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@main.route('/classify', methods=['POST'])
def classify():
    data = request.get_json()
    article = data['article']
    category = classify_news(article)
    print(f"The news belongs to {category} category!")  # Print to terminal
    return jsonify({'category': category})