import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/data_preparation.db')
df = pd.read_sql_table('data_preparation', engine)

# load model
model = joblib.load("../models/NLPclassifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    
    # PLOT 1 (Distribution of Message Genres)
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    
    # PLOT 2 (What is the attribute more frequently in news genre)
    news = df[df['genre']=='news'].iloc[:,4:]
    df_2 = pd.DataFrame(news)

    #create an empty list to save mean by column
    means_2 = []

    # get values with for cycle
    for i in range(len(df_2.columns)):
        column_2 = df_2.columns[i]
        mean_2 = df_2[column_2].mean()
        means_2.append([column_2, mean_2])

    # dataframe with results sorted by major value first
    results_2 = pd.DataFrame(means_2, columns=['column','mean'])
    results_2 = results_2.sort_values(by='mean',ascending=False)
    news_column_2 = results_2['column']
    news_values_2 = results_2['mean'] 
    
    # PLOT 3 (What is the attribute more frequently in direct genre)
    direct = df[df['genre']=='direct'].iloc[:,4:]
    df_3 = pd.DataFrame(direct)

    #create an empty list to save mean by column
    means_3 = []

    # get values with for cycle
    for i in range(len(df_3.columns)):
        column_3 = df_3.columns[i]
        mean_3 = df_3[column_3].mean()
        means_3.append([column_3, mean_3])

    # dataframe with results sorted by major value first
    results_3 = pd.DataFrame(means_3, columns=['column','mean'])
    results_3 = results_3.sort_values(by='mean',ascending=False)
    news_column_3 = results_3['column']
    news_values_3 = results_3['mean'] 
    
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                    
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
                {
            'data': [
                Bar(
                    x=news_column_2,
                    y=news_values_2
                )
            ],

            'layout': {
                'title': 'Feature Probability on "New" Genre',
                'yaxis': {
                    'title': "Probabiity"
                },
                'xaxis': {
                    'title': "Feature"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=news_column_3,
                    y=news_values_3
                )
            ],

            'layout': {
                'title': 'Feature Probability on "Direct" Genre',
                'yaxis': {
                    'title': "Probabiity"
                },
                'xaxis': {
                    'title': "Feature"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()