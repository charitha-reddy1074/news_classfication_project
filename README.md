# Flask News Categorization

This project is a Flask web application that classifies news articles into various categories such as politics, sports, and business. It utilizes a dataset from Kaggle and employs machine learning techniques, potentially using transformers, to enhance classification accuracy.

## Project Structure

```
flask-news-categorization
├── app
│   ├── __init__.py          # Initializes the Flask application and configuration
│   ├── routes.py            # Defines application routes for processing input and returning results
│   ├── templates
│   │   └── index.html       # HTML template for the front end
│   └── static
│       └── styles.css       # CSS styles for the front end
├── models
│   └── model.py             # Logic for loading the dataset and implementing the classification model
├── kagglehub_download.py     # Code to download the dataset from Kaggle
├── requirements.txt         # Lists project dependencies
├── run.py                   # Entry point for running the Flask application
└── README.md                # Project documentation
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd flask-news-categorization
   ```

2. **Install dependencies:**
   Ensure you have Python installed, then run:
   ```
   pip install -r requirements.txt
   ```

3. **Download the dataset:**
   The dataset will be downloaded automatically when you run the application.

4. **Run the application:**
   Execute the following command:
   ```
   python run.py
   ```
   The application will start on `http://127.0.0.1:5000/`.

## Usage

- Open your web browser and navigate to `http://127.0.0.1:5000/`.
- Enter a news article in the input box and click the "Classify" button.
- The application will process the input and display the classification result along with a graph of the classified data.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.