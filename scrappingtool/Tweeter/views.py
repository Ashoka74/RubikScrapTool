from sentence_transformers import SentenceTransformer, util
from django.shortcuts import render
import numpy as np
import snscrape
import snscrape.modules.reddit as srd
from snscrape import modules
import pandas as pd
import json
import requests
from django.shortcuts import redirect, render
from django.http import HttpResponse  # , request
import requests
from django.views.generic import View
from newscatcherapi import NewsCatcherApiClient
import re
import openai
from openai.embeddings_utils import cosine_similarity
import datetime
from textblob import TextBlob
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from turtle import back
from matplotlib.pyplot import legend
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import skimage as sk
from skimage import data
from PIL import Image
import umap
import numpy as np
import warnings
from matplotlib import pyplot as plt
import plotly.express as px
warnings.filterwarnings('ignore')

openai.api_key = '<API_KEY>'

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model = model.to('cuda')  # or CPU


def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity


def getPolarity(text):
    return TextBlob(text).sentiment.polarity


def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'

# Can be used for automatic media suggestions based on cosine similarity
# cos_df = pd.read_csv(
#     './templates/tweeter/media_similarity.csv')


# Not working yet
# def return_similar_news(request):
#     media = input('Enter media name: ')
#     # cos_df[media].sort_values(ascending=False)[1:11]
#     output = cos_df[media].sort_values(ascending=False)[1:5].to_html()
#     render(request, 'tweeter/news.html', {'output': output})


class home(View):
    def get(self, request):
        if request.method == 'POST':
            return redirect('home')
        else:
            # Don't forget to specify subfolder here to avoid error
            return render(request, 'tweeter/home.html')



def CleanTxt(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text)  # Remove @ Mentions
    text = re.sub(r'#', '', text)  # Remove '#' Hashtags
    text = re.sub(r'https?:\/\/\S+', '', text)  # Remove hyperlinks
    # Remove non-alphanumeric characters
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    return text


def NewsScrap(request):
    if request.method == 'POST':
        search_in = request.POST.get('search_in')
        try:
            sources = request.POST.get('sources')
        except:
            sources = request.POST.getlist('sources')
        print(sources)
        keyword = request.POST.get('keyword')
        start_date = request.POST.get('start_date')
        end_date = request.POST.get('end_date')
        # limit = request.POST.get('limit')

        # # Get value from checkbox from the form
        # topic_list = []

        # business = request.POST.get('Business')
        # culture = request.POST.get('Culture')
        # science = request.POST.get('Science')
        # mainstream = request.POST.get('News')

        # print(business, culture, science, mainstream)

        # topic_list = [business, culture, science, mainstream]

        # mytopic = ' OR '.join(topic_list)
        # print(mytopic)

        # ADD THE AUTOCOMPLETE FUNCTIONALITY
        

        DataFrame = pd.DataFrame()

        if start_date == '':
            start_date = 'one week ago'
        if end_date == '':
            end_date = 'today'

        newscatcherapi = NewsCatcherApiClient(
            x_api_key='YOUR_API_KEY')


        all_articles = newscatcherapi.get_search_all_pages(q=keyword,
                                                           search_in=search_in,
                                                           lang='en',
                                                           countries='us',
                                                           # topic=mytopic,
                                                           sources=sources,
                                                           to_rank=5000,
                                                           from_=start_date,
                                                           to_=end_date,
                                                           page_size=100)

        df = pd.concat([DataFrame, pd.DataFrame(
            all_articles['articles'])], ignore_index=True)

        # if SWOT is ticked
        if request.POST.get('swot') == 'on':
            # split articles per paragraphs
            df['summary'] = df['summary'].astype(str)

            df.split_summary = df['summary'].apply(
                lambda x: re.split(r"(?<=[.!?])\s+", x))

            df.split_summary = df.split_summary.apply(
                lambda x: list(map(lambda y: CleanTxt(y), x)))

            df.polarity = df.split_summary.apply(
                lambda x: list(map(lambda y: getPolarity(y), x)))

            df.subjectivity = df.split_summary.apply(
                lambda x: list(map(lambda y: getSubjectivity(y), x)))

            # compute embeddings for each paragraphs
            df.paragraph_embeddings = df.split_summary.apply(
                lambda x: list(map(lambda y: model.encode(y), x)))

            # create a synthetic SWOT summary based on the keywords
            completion_articles = openai.Completion.create(engine="text-davinci-002", prompt='Write an elaborate and detailed paragraph about the positive aspects of {}'.format(
                keyword), max_tokens=300, temperature=0.4, top_p=1, frequency_penalty=0, presence_penalty=0)

            completion_articles2 = openai.Completion.create(engine="text-davinci-002", prompt='Write an elaborate and detailed paragraph about the negative aspects of {}'.format(
                keyword), max_tokens=300, temperature=0.4, top_p=1, frequency_penalty=0, presence_penalty=0)

            # compute the embedding of the synthetic summary
            embedding_question = model.encode(
                completion_articles.choices[0].text, convert_to_tensor=True).tolist()

            embedding_question2 = model.encode(
                completion_articles2.choices[0].text, convert_to_tensor=True).tolist()

            # compute the cosine similarity between the synthetic summary and each paragraph
            df.cosine_similarity = df.paragraph_embeddings.apply(lambda x: list(
                map(lambda y: util.pytorch_cos_sim(embedding_question, y), x)))

            df.cosine_similarity2 = df.paragraph_embeddings.apply(lambda x: list(
                map(lambda y: util.pytorch_cos_sim(embedding_question2, y), x)))

            # create a new dataframe with the paragraphs and the cosine similarity
            new_df = pd.DataFrame({'paragraph': df.split_summary.explode(), 'cosine_similarity': df.cosine_similarity.explode(
            )}).sort_values(by='cosine_similarity', ascending=False).head(30)

            new_df2 = pd.DataFrame({'paragraph': df.split_summary.explode(), 'cosine_similarity2': df.cosine_similarity2.explode(
            )}).sort_values(by='cosine_similarity2', ascending=False).head(30)

            SWOT_Data = new_df.paragraph.to_list()
            SWOT_Data2 = new_df2.paragraph.to_list()

            SWOT_analysis = openai.Completion.create(engine="text-davinci-002", prompt='Data: Positive aspects:{}\nNegative aspects:{}\nWrite an elaborate SWOT analysis on {} based on the above data:\n\nStrengths:\n -'.format(
                SWOT_Data, SWOT_Data2, keyword), max_tokens=800, temperature=0.3, top_p=1, frequency_penalty=0.3, presence_penalty=0.3)

         
            
            SWOT_analysis = SWOT_analysis.choices[0].text.replace(' -', '\n-')

            return render(request, 'tweeter/news.html', {'df': SWOT_Data, 'df2': SWOT_Data2, 'SWOT_analysis': 'Strengths: {}'.format(SWOT_analysis)})

        if request.POST.get('pestle') == 'on':
            # split articles per paragraphs
            df['summary'] = df['summary'].astype(str)

            df.split_summary = df['summary'].apply(
                lambda x: re.split(r"(?<=[.!?])\s+", x))

            df.split_summary = df.split_summary.apply(
                lambda x: list(map(lambda y: CleanTxt(y), x)))

            df.polarity = df.split_summary.apply(
                lambda x: list(map(lambda y: getPolarity(y), x)))

            df.subjectivity = df.split_summary.apply(
                lambda x: list(map(lambda y: getSubjectivity(y), x)))

            # compute embeddings for each paragraphs
            df.paragraph_embeddings = df.split_summary.apply(
                lambda x: list(map(lambda y: model.encode(y), x)))

            # # create a synthetic SWOT summary based on positive and negative keywords
            # completion_articles = openai.Completion.create(engine="text-davinci-002", prompt='Write an elaborate and detailed paragraph about the positive aspects of {}'.format(
            #     keyword), max_tokens=300, temperature=0.4, top_p=1, frequency_penalty=0, presence_penalty=0)

            # completion_articles2 = openai.Completion.create(engine="text-davinci-002", prompt='Write an elaborate and detailed paragraph about the negative aspects of {}'.format(
            #     keyword), max_tokens=300, temperature=0.4, top_p=1, frequency_penalty=0, presence_penalty=0)

            polit = openai.Completion.create(engine="text-davinci-002", prompt='Write an elaborate and detailed paragraph about the positive and negative effects of {} on politics.\n\n'.format(
                keyword), max_tokens=300, temperature=0.7, top_p=1, frequency_penalty=0, presence_penalty=0)

            econom = openai.Completion.create(engine="text-davinci-002", prompt='Write an elaborate and detailed paragraph about the positive and negative effects of {} on economics.\n\n'.format(
                keyword), max_tokens=300, temperature=0.7, top_p=1, frequency_penalty=0, presence_penalty=0)

            soci = openai.Completion.create(engine="text-davinci-002", prompt='Write an elaborate and detailed paragraph about positive and negative social effects of {}.\n\n'.format(
                keyword), max_tokens=300, temperature=0.7, top_p=1, frequency_penalty=0, presence_penalty=0)

            tech = openai.Completion.create(engine="text-davinci-002", prompt='Write an elaborate and detailed paragraph about positive and negative effects of {} on technology.\n\n'.format(
                keyword), max_tokens=300, temperature=0.7, top_p=1, frequency_penalty=0, presence_penalty=0)

            legal = openai.Completion.create(engine="text-davinci-002", prompt='Write an elaborate and detailed paragraph about positive and negative legal effects of {}.\n\n'.format(
                keyword), max_tokens=300, temperature=0.7, top_p=1, frequency_penalty=0, presence_penalty=0)

            environ = openai.Completion.create(engine="text-davinci-002", prompt='Write an elaborate and detailed paragraph about positive and negative effects of {} on the environment.\n\n'.format(
                keyword), max_tokens=300, temperature=0.7, top_p=1, frequency_penalty=0, presence_penalty=0)

            # compute the embedding of the synthetic summary
            # embedding_question = model.encode(
            #     completion_articles.choices[0].text, convert_to_tensor=True).tolist()

            # embedding_question2 = model.encode(
            #     completion_articles2.choices[0].text, convert_to_tensor=True).tolist()

            polit = model.encode(
                polit.choices[0].text, convert_to_tensor=True).tolist()

            econom = model.encode(
                econom.choices[0].text, convert_to_tensor=True).tolist()

            soci = model.encode(
                soci.choices[0].text, convert_to_tensor=True).tolist()

            tech = model.encode(
                tech.choices[0].text, convert_to_tensor=True).tolist()

            legal = model.encode(
                legal.choices[0].text, convert_to_tensor=True).tolist()

            environ = model.encode(
                environ.choices[0].text, convert_to_tensor=True).tolist()

            # compute the cosine similarity between the synthetic summary and each paragraph
            # df.cosine_similarity = df.paragraph_embeddings.apply(lambda x: list(
            #     map(lambda y: util.pytorch_cos_sim(embedding_question, y), x)))

            # df.cosine_similarity2 = df.paragraph_embeddings.apply(lambda x: list(
            #     map(lambda y: util.pytorch_cos_sim(embedding_question2, y), x)))

            df.polit = df.paragraph_embeddings.apply(lambda x: list(
                map(lambda y: util.pytorch_cos_sim(polit, y), x)))

            df.econom = df.paragraph_embeddings.apply(lambda x: list(
                map(lambda y: util.pytorch_cos_sim(econom, y), x)))

            df.soci = df.paragraph_embeddings.apply(lambda x: list(
                map(lambda y: util.pytorch_cos_sim(soci, y), x)))

            df.tech = df.paragraph_embeddings.apply(lambda x: list(
                map(lambda y: util.pytorch_cos_sim(tech, y), x)))

            df.legal = df.paragraph_embeddings.apply(lambda x: list(
                map(lambda y: util.pytorch_cos_sim(legal, y), x)))

            df.environ = df.paragraph_embeddings.apply(lambda x: list(
                map(lambda y: util.pytorch_cos_sim(environ, y), x)))

            # # create a new dataframe with the paragraphs and the cosine similarity
            # new_df = pd.DataFrame({'paragraph': df.split_summary.explode(), 'cosine_similarity': df.cosine_similarity.explode(
            # )}).sort_values(by='cosine_similarity', ascending=False)
            # # remove double paragraphs
            # new_df = new_df.drop_duplicates(subset=['paragraph']).head(50)

            # new_df2 = pd.DataFrame({'paragraph': df.split_summary.explode(), 'cosine_similarity2': df.cosine_similarity2.explode(
            # )}).sort_values(by='cosine_similarity2', ascending=False)
            # new_df2 = new_df2.drop_duplicates(subset=['paragraph']).head(50)

            new_df_polit = pd.DataFrame({'paragraph': df.split_summary.explode(), 'cosine_similarity_polit': df.polit.explode(
            )}).sort_values(by='cosine_similarity_polit', ascending=False)
            new_df_polit = new_df_polit.drop_duplicates(
                subset=['paragraph']).head(60)

            new_df_econom = pd.DataFrame({'paragraph': df.split_summary.explode(), 'cosine_similarity_econom': df.econom.explode(
            )}).sort_values(by='cosine_similarity_econom', ascending=False)
            new_df_econom = new_df_econom.drop_duplicates(
                subset=['paragraph']).head(60)

            new_df_soci = pd.DataFrame({'paragraph': df.split_summary.explode(), 'cosine_similarity_soci': df.soci.explode(
            )}).sort_values(by='cosine_similarity_soci', ascending=False)
            new_df_soci = new_df_soci.drop_duplicates(
                subset=['paragraph']).head(60)

            new_df_tech = pd.DataFrame({'paragraph': df.split_summary.explode(), 'cosine_similarity_tech': df.tech.explode(
            )}).sort_values(by='cosine_similarity_tech', ascending=False)
            new_df_tech = new_df_tech.drop_duplicates(
                subset=['paragraph']).head(60)

            new_df_legal = pd.DataFrame({'paragraph': df.split_summary.explode(), 'cosine_similarity_legal': df.legal.explode(
            )}).sort_values(by='cosine_similarity_legal', ascending=False)
            new_df_legal = new_df_legal.drop_duplicates(
                subset=['paragraph']).head(60)

            new_df_environ = pd.DataFrame({'paragraph': df.split_summary.explode(), 'cosine_similarity_environ': df.environ.explode(
            )}).sort_values(by='cosine_similarity_environ', ascending=False)
            new_df_environ = new_df_environ.drop_duplicates(
                subset=['paragraph']).head(60)

            # PESTLE_Data = new_df.paragraph.to_list()
            # PESTLE_Data2 = new_df2.paragraph.to_list()

            PESTLE_Data_polit = new_df_polit.paragraph.to_list()
            PESTLE_Data_econom = new_df_econom.paragraph.to_list()
            PESTLE_Data_soci = new_df_soci.paragraph.to_list()
            PESTLE_Data_tech = new_df_tech.paragraph.to_list()
            PESTLE_Data_legal = new_df_legal.paragraph.to_list()
            PESTLE_Data_environ = new_df_environ.paragraph.to_list()

            P = openai.Completion.create(engine="text-davinci-002", prompt='Data: {}\nSummarize the above data into 3-4 most important legal implications of {}:\n\n -'.format(
                PESTLE_Data_polit, keyword), max_tokens=550, temperature=0.3, top_p=1, frequency_penalty=0.3, presence_penalty=0.3).choices[0].text.replace(' -', '\n-')

            e = openai.Completion.create(engine="text-davinci-002", prompt='Data: {}\nSummarize the above data into 3-4 most important financial implications of {}:\n\n -'.format(
                PESTLE_Data_econom, keyword), max_tokens=550, temperature=0.3, top_p=1, frequency_penalty=0.3, presence_penalty=0.3).choices[0].text.replace(' -', '\n-')

            S = openai.Completion.create(engine="text-davinci-002", prompt='Data: {}\nSummarize the above data into 3-4 most important social implications of {}:\n\n -'.format(
                PESTLE_Data_soci, keyword), max_tokens=550, temperature=0.3, top_p=1, frequency_penalty=0.3, presence_penalty=0.3).choices[0].text.replace(' -', '\n-')

            T = openai.Completion.create(engine="text-davinci-002", prompt='Data: {}\nSummarize the above data into 3-4 most important technological implications of {}:\n\n -'.format(
                PESTLE_Data_tech, keyword), max_tokens=550, temperature=0.3, top_p=1, frequency_penalty=0.3, presence_penalty=0.3).choices[0].text.replace(' -', '\n-')

            L = openai.Completion.create(engine="text-davinci-002", prompt='Data: {}\nSummarize the above data into 3-4 most important legal implications of {}:\n\n -'.format(
                PESTLE_Data_legal, keyword), max_tokens=550, temperature=0.3, top_p=1, frequency_penalty=0.3, presence_penalty=0.3).choices[0].text.replace(' -', '\n-')

            E = openai.Completion.create(engine="text-davinci-002", prompt='Data:{}\nSummarize the above data into 3-4 most important the environmental implications of {}:\n\n -'.format(
                PESTLE_Data_environ, keyword), max_tokens=550, temperature=0.3, top_p=1, frequency_penalty=0.3, presence_penalty=0.3).choices[0].text.replace(' -', '\n-')

            PESTLE_analysis = (
                'Political:\n{}\n\nEconomic:\n{}\n\nSocial:\n{}\n\nTechnological:\n{}\n\nLegal:\n{}\n\nEnvironmental:\n{}'.format(P, e, S, T, L, E))

            PESTLE_Data = 'Political:\n{}\n\nEconomic:\n{}\n\nSocial:\n{}\n\nTechnological:\n{}\n\nLegal:\n{}\n\nEnvironmental:\n{}'.format(
                PESTLE_Data_polit, PESTLE_Data_econom, PESTLE_Data_soci, PESTLE_Data_tech, PESTLE_Data_legal, PESTLE_Data_environ)

            return render(request, 'tweeter/news.html', {'df': PESTLE_Data, 'PESTLE_analysis': PESTLE_analysis})

        else:
            return HttpResponse(df.to_html())

    else:
        return render(request, 'tweeter/news.html')


def RedditScrap(request):
    if request.method == 'POST':
        subreddit = request.POST.get('subreddit')
        limit = request.POST.get('limit')

        df = pd.DataFrame(columns=[
                          'type', 'author', 'body', 'created', 'id', 'parentId', 'subreddit', 'url'])

        j = 0

        for i in snscrape.modules.reddit.RedditSubredditScraper(subreddit).get_items():
            try:
                i.url
            except:
                continue

            if j < int(limit):
                i = i.json()
                i = json.loads(i)
                _type = i['_type']
                author = i['author']
                try:
                    body = i['body']
                except:
                    body = ''
                # embed = model.encode(body)  # try astype(str)
                created = i['created']
                _id = i['id']
                try:
                    parentId = i['parentId']
                except:
                    parentId = ''
                subreddit = i['subreddit']
                _url = i['url']

                row1 = [_type, author, body, created,
                        _id, parentId, subreddit, _url]  # embed
                df.loc[len(df)] = row1
                j += 1
            else:
                break

        # convert the dataframe to json
        # df_json = df.to_json(orient='records')

        df_table = df.to_html()

        # if SWOT is ticked
        if request.POST.get('swot') == 'on':
            # split articles per paragraphs
            df.paragraph_embeddings = df.body.apply(
                lambda x: model.encode(x, convert_to_tensor=True).tolist())

            # create a synthetic SWOT summary based on the keywords
            positive_gpt = openai.Completion.create(engine="text-davinci-002", prompt='Write an elaborate and detailed paragraph about the positive aspects of {}.'.format(
                subreddit), max_tokens=300, temperature=0.4, top_p=1, frequency_penalty=0, presence_penalty=0)

            negative_gpt = openai.Completion.create(engine="text-davinci-002", prompt='Write an elaborate and detailed paragraph about the negative aspects of {}.'.format(
                subreddit), max_tokens=300, temperature=0.4, top_p=1, frequency_penalty=0, presence_penalty=0)

            print(positive_gpt.choices[0].text)
            print(negative_gpt.choices[0].text)

            # compute the embedding of the synthetic summary
            positive_embeddings = model.encode(
                positive_gpt.choices[0].text, convert_to_tensor=True).tolist()

            negative_embeddings = model.encode(
                negative_gpt.choices[0].text, convert_to_tensor=True).tolist()

            # compute the cosine similarity between the synthetic summary and each paragraph
            df['positive_similarity'] = df.paragraph_embeddings.apply(lambda x: util.pytorch_cos_sim(
                positive_embeddings, x))

            df['negative_similarity'] = df.paragraph_embeddings.apply(lambda x: util.pytorch_cos_sim(
                negative_embeddings, x))

            # create a new dataframe with the paragraphs and the cosine similarity
            new_df = df.sort_values(
                by='positive_similarity', ascending=False).head(30)

            new_df2 = df.sort_values(
                by='negative_similarity', ascending=False).head(30)

            SWOT_Data = new_df.body.to_list()
            SWOT_Data2 = new_df2.body.to_list()

            SWOT_analysis = openai.Completion.create(engine="text-davinci-002", prompt='Data: Positive aspects:{}\nNegative aspects:{}\nWrite an elaborate SWOT analysis on {} based on the above data:\n\nStrengths:\n -'.format(
                SWOT_Data, SWOT_Data2, subreddit), max_tokens=800, temperature=0.3, top_p=1, frequency_penalty=0.3, presence_penalty=0.3)

            SWOT_analysis = SWOT_analysis.choices[0].text.replace(' -', '\n-')

            print(SWOT_analysis)

            return render(request, 'tweeter/reddit.html', {'df': SWOT_Data, 'df2': SWOT_Data2, 'SWOT_analysis': 'Strengths: {}'.format(SWOT_analysis), 'subreddit': subreddit}, content_type='text/html')

        elif request.POST.get('emotion') == 'on':

            # define emotions
            emotions_csv = pd.read_csv(
                './Tweeter/templates/tweeter/emotions.csv')

            def get_key(element):
                return element['label']

            tokenizer = AutoTokenizer.from_pretrained(
                "joeddav/distilbert-base-uncased-go-emotions-student", truncation=True)
            model2 = AutoModelForSequenceClassification.from_pretrained(
                "joeddav/distilbert-base-uncased-go-emotions-student")
            nlp = pipeline('sentiment-analysis', model=model2,
                           tokenizer=tokenizer, top_k=27, truncation=True, device=0)

            # some cleaning for reddit datasets
            df['text'] = df.body.astype(str)
            df = df[df['text'] != 'nan']
            df = df[df['text'] != '[deleted]']
            df = df[df['text'] != '[removed]']

            dataset_umap = pd.DataFrame(
                columns=['text', 'emotion', 'score', 'top_emotion'])

            # apply emotion model on data and get the labels and scores
            for i in range(len(df)):
                label = []
                score = []
                jsonfile = (nlp(df['text'].iloc[i]))
                jsonfile[0].sort(key=get_key)
                for j in range(0, 27):
                    jsonfile2 = np.array(jsonfile)
                    label.append(jsonfile2[0][j]['label'])
                    score.append(jsonfile2[0][j]['score'])

                dataset_umap.loc[len(dataset_umap)] = [df['text'].iloc[i], label, score, label[score.index(
                    max(score))]]

            # df_table = dataset_umap[dataset_umap['top_emotion'] == 'caring'].to_html()

            # return render(request, 'tweeter/reddit.html', {'df': df_table, 'subreddit': subreddit}, content_type='text/html')

            dataset_umap['top_emotion_number'] = dataset_umap['top_emotion'].astype(
                'category').cat.codes

            dataset_umap['trimmed_text'] = dataset_umap['text'].str[:175]

            dataset_umap = dataset_umap.sort_values(by=['top_emotion'])

            if len(dataset_umap) > 1000:
                dataset_umap = dataset_umap[dataset_umap['top_emotion'].isin(
                    dataset_umap['top_emotion'].value_counts().head(10).index)]

                # Uncomment to save file
                # dataset_umap.to_hdf('df_emotion.h5', key='df', mode='w')

                reduce_dim = umap.UMAP(
                    n_components=3, n_neighbors=8, min_dist=0.95)
                embedding = dataset_umap['score'].tolist()
                embedding = np.array(embedding)
                umap_embeddings = reduce_dim.fit_transform(
                    embedding, y=dataset_umap['top_emotion_number'])

                dataset_umap['x'] = umap_embeddings[:, 0]
                dataset_umap['y'] = umap_embeddings[:, 1]
                dataset_umap['z'] = umap_embeddings[:, 2]
                # assign colors to the top 10 emotions
                colors = emotions_csv[emotions_csv['Emotions'].isin(
                    dataset_umap['top_emotion'].value_counts().head(10).index)].iloc[:, 1].values.tolist()
                if len(dataset_umap) > 10000:
                    dot_size = 1
                else:
                    dot_size = 1.5

            elif len(dataset_umap) <= 1000:
                reduce_dim = umap.UMAP(
                    n_components=3, n_neighbors=8, min_dist=0.95)
                embedding = dataset_umap['score'].tolist()
                embedding = np.array(embedding)
                umap_embeddings = reduce_dim.fit_transform(
                    embedding, y=dataset_umap['top_emotion_number'])

                dataset_umap['x'] = umap_embeddings[:, 0]
                dataset_umap['y'] = umap_embeddings[:, 1]
                dataset_umap['z'] = umap_embeddings[:, 2]
                # assign colors to the top 10 emotions
                colors = emotions_csv.iloc[:, 1].values.tolist()
                dot_size = 2

            pyLogo = Image.open("./Tweeter/templates/tweeter/Favicon_B.png")
            pyLogo2 = Image.open("./Tweeter/templates/tweeter/Favicon_B.png")

            fig = px.scatter_3d(dataset_umap, x='x', y='y', z='z', color='top_emotion', hover_name='trimmed_text', hover_data={
                                'x': False, 'y': False, 'z': False, 'top_emotion': False, 'text': False}, color_discrete_sequence=colors, opacity=1, template='plotly_dark')

            fig.update_traces(marker=dict(size=dot_size))

            fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(
                size=0.1, color='black'), showlegend=True, name=' ', hoverinfo='none'))

            # legend on the right side
            fig.update_layout(legend=dict(
                bgcolor='rgba(17,17,17,0)', xanchor='right'))

            fig.update_layout(scene=dict(
                xaxis=dict(
                    title=' ',
                    nticks=0,
                    # backgroundcolor="rgb(0, 0, 0, 1)",
                    gridcolor="rgba(17,17,17, 0)",
                    showbackground=True,
                    zerolinecolor="rgba(17,17,17, 0)",
                    zeroline=False,
                    showgrid=False,
                    showticklabels=False,
                    showspikes=False
                ),
                # hide ticks


                yaxis=dict(
                    # name
                    title=' ',
                    nticks=0,
                    # backgroundcolor="rgb(0, 0, 0, 1)",
                    gridcolor="rgba(17,17,17, 0)",
                    showbackground=True,
                    zerolinecolor="rgba(17,17,17, 0)",
                    zeroline=False,
                    showgrid=False,
                    showticklabels=False,
                    showspikes=False
                ),



                zaxis=dict(
                    # name
                    title=' ',
                    nticks=0,
                    # backgroundcolor="rgba(0, 0, 0, 1)",
                    gridcolor="rgba(17,17,17, 0)",
                    showbackground=True,
                    zerolinecolor="rgba(17,17,17, 0)",
                    zeroline=False,
                    showgrid=False,
                    showticklabels=False,
                    showspikes=False),)
                # tickvals=[],),
            )

            fig.update_layout(coloraxis_showscale=False, legend=dict(x=0.1, y=0.5, traceorder='normal', font=dict(
                family='Noto Serif', size=14, color='white'), bgcolor='rgba(17,17,17,0)', bordercolor='rgba(17,17,17,0)', borderwidth=0))

            fig.add_layout_image(
                dict(
                    source=pyLogo,
                    xref="x",
                    yref="y",
                    x=-1,
                    y=3.8,
                    # xanchor = "left",
                    # yanchor = "top",
                    sizex=.4,
                    sizey=.4,
                    opacity=1,
                    layer="above",
                )

            )  # , margin=dict(l=0, r=0, t=0, b=0, pad=0))

            fig.update_layout(legend={'itemsizing': 'constant'}, legend_title_text=' ', legend_title_font_color='white', legend_title_font_family='Noto Serif',
                              legend_font_color='white', legend_font_size=14, legend_font_family='Noto Serif', legend_bgcolor='rgba(17,17,17,0)', legend_bordercolor='rgba(17,17,17,0)', legend_borderwidth=2)

            # , title_font_size=30, title_font_family='Noto Serif', title_font_color='white', title_x=0.06, title_y=0.95, title_xanchor='left', title_yanchor='top', title_text='Cluster of Emotions for {}\n                                        n = {}'.format(subreddit, len(dataset_umap)), margin=dict(l=0, r=0, b=0, t=0, pad=0))
            fig.update_layout(scene_camera_eye=dict(x=0.87, y=-0.88, z=0.84), scene_camera_center=dict(
                x=0, y=0, z=0), template='plotly_dark', hovermode='x unified', margin=dict(l=0, r=0, b=0, t=0, pad=2))

            fig.update_layout(coloraxis_showscale=False)
            fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False,
                             showline=False, automargin=False, showspikes=False)
            fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False,
                             showline=False, automargin=False, showspikes=False)

            return render(request, 'tweeter/reddit.html', {'fig': fig.to_html(full_html=False, include_plotlyjs='cdn', default_height='750px', default_width='1500px')})

        else:
            return HttpResponse(df_table)

    else:
        return render(request, 'tweeter/reddit.html')


def TweetScrap(request):
    if request.method == 'POST':
        limit = request.POST.get('limit')
        username = request.POST.get('username')
        hashtags = request.POST.get('hashtag')
        since = request.POST.get('start_date')
        until = request.POST.get('end_date')

        # check if username or hastag is True
        if username != '':
            if username[0] == '@':
                username = username[1:]

            tweet_list = []

            if until is None:
                until = datetime.datetime.strftime(
                    datetime.date.today(), '%Y-%m-%d')
            if since is None:
                since = datetime.datetime.strftime(
                    datetime.datetime.strptime(until, '%Y-%m-%d') - datetime.timedelta(days=7), '%Y-%m-%d')

            search_criteria = f"from:{username} since:{since} until:{until} exclude:retweets exclude:replies"

            for i, tweet in enumerate(snscrape.modules.twitter.TwitterSearchScraper(search_criteria).get_items()):
                if i < int(limit):
                    tweet_list.append(
                        [tweet.date, tweet.id, tweet.content, tweet.user.username])
                else:
                    break

            df = pd.DataFrame(tweet_list, columns=[
                'Datetime', 'Tweet Id', 'Text', 'Username'])

            query = username

            df = df.sort_values(by='Datetime', ascending=True)

            df_table2 = df.to_html()

        if hashtags != '':
            hashtags = hashtags.split(',')
            # remove spaces
            hashtags = [i.strip() for i in hashtags]
            for i in range(len(hashtags)):
                if hashtags[i][0] != '#':
                    hashtags[i] = '#{}'.format(hashtags[i])

            hashtags = " OR ".join(hashtags)
            print(hashtags)

            tweet_list2 = []

            if until is None:
                until = datetime.datetime.strftime(
                    datetime.date.today(), '%Y-%m-%d')
            if since is None:
                since = datetime.datetime.strftime(
                    datetime.datetime.strptime(until, '%Y-%m-%d') - datetime.timedelta(days=7), '%Y-%m-%d')

            search_criteria = f"{hashtags} since:{since} until:{until} exclude:retweets exclude:replies"

            if until and since is None:
                search_criteria = f"{hashtags} exclude:retweets exclude:replies"

            print(search_criteria)

            for i, tweet in enumerate(snscrape.modules.twitter.TwitterSearchScraper(search_criteria).get_items()):
                if i < int(limit):
                    tweet_list2.append(
                        [tweet.date, tweet.id, tweet.content, tweet.user.username])  # maybe use astype(str)
                else:
                    break

            df = pd.DataFrame(tweet_list2, columns=[
                'Datetime', 'Tweet Id', 'Text', 'Username'])

            df = df.sort_values(by='Datetime', ascending=True)

            query = hashtags

            # convert the dataframe to json
            df_table2 = df.to_html()

        if request.POST.get('swot') == 'on':

            df['clean_text'] = df.Text.apply(lambda x: CleanTxt(x))

            # split articles per paragraphs
            df.paragraph_embeddings = df['clean_text'].apply(
                lambda x: model.encode(x, convert_to_tensor=True).tolist())

            # create a synthetic SWOT summary based on the keywords
            positive_gpt = openai.Completion.create(engine="text-davinci-002", prompt='Write an elaborate and detailed paragraph about the positive aspects of {}.'.format(
                query), max_tokens=300, temperature=0.4, top_p=1, frequency_penalty=0, presence_penalty=0)

            negative_gpt = openai.Completion.create(engine="text-davinci-002", prompt='Write an elaborate and detailed paragraph about the negative aspects of {}.'.format(
                query), max_tokens=300, temperature=0.4, top_p=1, frequency_penalty=0, presence_penalty=0)

            print(positive_gpt.choices[0].text)
            print(negative_gpt.choices[0].text)

            # compute the embedding of the synthetic summary
            positive_embeddings = model.encode(
                positive_gpt.choices[0].text, convert_to_tensor=True).tolist()

            negative_embeddings = model.encode(
                negative_gpt.choices[0].text, convert_to_tensor=True).tolist()

            # compute the cosine similarity between the synthetic summary and each paragraph
            df['positive_similarity'] = df.paragraph_embeddings.apply(lambda x: util.pytorch_cos_sim(
                positive_embeddings, x))

            df['negative_similarity'] = df.paragraph_embeddings.apply(lambda x: util.pytorch_cos_sim(
                negative_embeddings, x))

            # create a new dataframe with the paragraphs and the cosine similarity
            new_df = df.sort_values(
                by='positive_similarity', ascending=False).head(20)

            new_df2 = df.sort_values(
                by='negative_similarity', ascending=False).head(20)

            SWOT_Data = new_df['clean_text'].to_list()
            SWOT_Data2 = new_df2['clean_text'].to_list()

            SWOT_analysis = openai.Completion.create(engine="text-davinci-002", prompt='Data: Positive aspects:{}\nNegative aspects:{}\nWrite an elaborate SWOT analysis on {} based on the above data:\n\nStrengths:\n -'.format(
                SWOT_Data, SWOT_Data2, query), max_tokens=800, temperature=0.3, top_p=1, frequency_penalty=0.3, presence_penalty=0.3)

            # SWOT_all = SWOT_Data + SWOT_Data2

            # create a \n before each - in SWOT analysis
            SWOT_analysis = SWOT_analysis.choices[0].text.replace(' -', '\n-')

            print(SWOT_analysis)

            return render(request, 'tweeter/twitter.html', {'df': SWOT_Data, 'df2': SWOT_Data2, 'SWOT_analysis': 'Strengths: {}'.format(SWOT_analysis), 'query': query}, content_type='text/html')

        else:
            return HttpResponse(df_table2)

    else:
        return render(request, 'tweeter/twitter.html')


# class TwitterData(View):

#     def get_data(self, request):
#         if request.method == 'POST':
#             username = request.POST.get('username')
#             limit = request.POST.get('limit')

#             # print hello world
#             print('hello world')
#             return render(request, 'tweeter/twitter.html', {'username': username, 'limit': limit})

#             # data_list = []
#             # j = 0
#             # for i in modules.twitter.TwitterSearchScraper(self.username).get_items():
#             #     if j < self.limit:
#             #         data_list.append(i.content)
#             #         j += 1
#             #     else:
#             #         break

#             # df = pd.DataFrame(data_list)
#             # # return the dataframe
#             # df_json = df.to_json(orient='records')

#            # return render(request, 'home.html', {'df_json': df_json})
#         else:
#             return render(request, 'tweeter/twitter.html')
#         # return render(request, 'reddit.html')


# # test the class
# if __name__ == '__main__':

#     social_media = input(
#         'Enter the social media you want to scrape (Twitter/Reddit): ').lower()
#     if social_media == 'reddit':
#         subreddit = input('Enter the subreddit you want to scrape: ')
#         limit = int(input('Enter the limit: '))
#         reddit = RedditData(subreddit, limit)
#         df = reddit.get_data()
#         print(df.columns)
#     elif social_media == 'twitter':

#         type_of_search = input(
#             'Enter the type of search you want to do (User/Hashtag): ').lower()
#         if type_of_search == 'user':
#             username = 'from:{}'.format(
#                 input('Enter the username you want to scrape: ')).lower()
#             limit = int(input('Enter the limit: '))
#             twitter = TwitterData(username, limit)
#             df = twitter.get_data()
#         elif type_of_search == 'hashtag':
#             hashtag = '#{}'.format(
#                 input('Enter the hashtag you want to scrape: ')).lower()
#             print(df.columns)

#         twitter = TwitterData(username, limit)
#         df = twitter.get_data()
#         print(df)
