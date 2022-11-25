
from django.urls import path
from .views import home, TweetScrap, RedditScrap, NewsScrap
from django.conf.urls import url, include

urlpatterns = [
    path('', home.as_view(), name='home'),
    path('reddit/', RedditScrap, name='reddit'),
    path('twitter/', TweetScrap, name='twitter'),
    path('news/', NewsScrap, name='news')
]
