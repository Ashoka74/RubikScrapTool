from django.core.management.base import BaseCommand, CommandError
import csv
import pandas as pd

cos_df = pd.read_csv(
    'C:/Users/sinan/OneDrive/Desktop/projects/RubikLab/Visuals/media_similarity.csv')


