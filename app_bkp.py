import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import base64

parent_folder_path  = r"C:\Users\miyan\OneDrive\Documents\my docs\Courses TAMU\Data Viz\project\years"
spotify_csv = os.path.join(parent_folder_path, "spotify_combined_data.csv")
spotify_data = pd.read_csv(spotify_csv, parse_dates=['album_release_date'])
st.set_page_config(page_icon=None, layout="centered")
st.title("Harmonizing Decades: A Musical Journey Through Spotify (1994-2023)")

def image_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
image_local('background.jpg')

# st.set_page_config(layout="wide")
#--------------------------------------------------------------------------------------
# Plot 1
#analying the mean populatity of rock, pop, rap, and r&b music
df = spotify_data[['song_id', 'song_popularity', 'artist_genres', 'album_release_date']]
genres_to_check = ['rock', 'rap', 'r&b', 'country', 'k-pop']
df.loc[:, genres_to_check] = 0
substring_to_check = 'rock'
df.loc[:, 'rock'] = df['artist_genres'].str.contains(substring_to_check, case=False)
substring_to_check = 'rap'
df.loc[:, 'rap'] = df['artist_genres'].str.contains(substring_to_check, case=False)
substring_to_check = 'r&b'
df.loc[:, 'r&b'] = df['artist_genres'].str.contains(substring_to_check, case=False)
substring_to_check = 'country'
df.loc[:, 'country'] = df['artist_genres'].str.contains(substring_to_check, case=False)
##############################################################################
df_rap = df[['song_popularity', 'rap', 'album_release_date']]
df_rap = df_rap[df_rap['rap']==True]
df_rap = df_rap[['song_popularity', 'album_release_date']]
df_rap_filtered = df_rap[(df_rap['album_release_date'].dt.year >= 1994) & (df_rap['album_release_date'].dt.year <= 2023)]
rap_count_by_year = df_rap_filtered.groupby(df_rap_filtered['album_release_date'].dt.year).count()
rap_count_by_year.columns = ['year', 'rap']
rap_count_by_year['year'] = rap_count_by_year.index

################################################################################
df_rock = df[['song_popularity', 'rock', 'album_release_date']]
df_rock = df_rock[df_rock['rock']==True]
df_rock = df_rock[['song_popularity', 'album_release_date']]
# Filter rows within the range from 1994 to 2023
df_rock_filtered = df_rock[(df_rock['album_release_date'].dt.year >= 1994) & (df_rock['album_release_date'].dt.year <= 2023)]
rock_count_by_year = df_rock_filtered.groupby(df_rock_filtered['album_release_date'].dt.year).count()
rock_count_by_year.columns = ['year', 'rock']
rock_count_by_year['year'] = rock_count_by_year.index

###############################################################################
df_rb = df[['song_popularity', 'r&b', 'album_release_date']]
df_rb = df_rb[df_rb['r&b']==True]
df_rb = df_rb[['song_popularity', 'album_release_date']]
# Filter rows within the range from 1994 to 2023
df_rb_filtered = df_rb[(df_rb['album_release_date'].dt.year >= 1994) & (df_rb['album_release_date'].dt.year <= 2023)]
rb_count_by_year = df_rb_filtered.groupby(df_rb_filtered['album_release_date'].dt.year).count()
rb_count_by_year.columns = ['year', 'r&b']
rb_count_by_year['year'] = rb_count_by_year.index
#---------------------------------------------------------------------------------------------------------------

df_country = df[['song_popularity', 'country', 'album_release_date']]
df_country = df_country[df_country['country']==True]
df_country = df_country[['song_popularity', 'album_release_date']]
# Filter rows within the range from 1994 to 2023
df_country_filtered = df_country[(df_country['album_release_date'].dt.year >= 1994) & (df_country['album_release_date'].dt.year <= 2023)]
country_count_by_year = df_country_filtered.groupby(df_country_filtered['album_release_date'].dt.year).count()
country_count_by_year.columns = ['year', 'country']
country_count_by_year['year'] = country_count_by_year.index

df_count_by_genres = pd.merge(rap_count_by_year, rock_count_by_year, on='year')
df_count_by_genres = pd.merge(df_count_by_genres, rb_count_by_year, on='year')
df_count_by_genres = pd.merge(df_count_by_genres, country_count_by_year, on='year')

# Melt the DataFrame to convert it to long format for Seaborn
df_count_by_genres_melted = df_count_by_genres.melt(id_vars='year', var_name='genre', value_name='count')

# Streamlit app
st.title('Count of Songs by Genre Over Years')

# Add sliders for zooming and scrolling
start_year, end_year = st.slider('Select Years Range', min_value=min(df_count_by_genres_melted['year']),
                                 max_value=max(df_count_by_genres_melted['year']), value=(2010, 2019))

genres_to_display = st.multiselect('Select Genres to Display', df_count_by_genres_melted['genre'].unique(),
                                   default=df_count_by_genres_melted['genre'].unique())

# Filter the DataFrame based on selected years and genres
filtered_df = df_count_by_genres_melted[(df_count_by_genres_melted['year'] >= start_year) &
                                         (df_count_by_genres_melted['year'] <= end_year) &
                                         (df_count_by_genres_melted['genre'].isin(genres_to_display))]

# Plot using Seaborn
sns.set(style="whitegrid", palette="bright")
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(x='year', y='count', hue='genre', data=filtered_df, marker='o', ax=ax)
plt.xlabel('Year')
plt.ylabel('Count')
plt.title('Count of Songs by Genre Over Years')

# Display the plot in Streamlit
st.pyplot(fig)

#------------------------------------------------------------------------------------------------------------------------
#Plot 2
def filter_df_with_genres(genres_to_check, df):
    df.loc[:, genres_to_check] = 0
    for genre in genres_to_check:
        df.loc[:, genre] = df['artist_genres'].str.contains(genre, case=False)
    filtered_df = df.loc[~(df[genres_to_check] == False).all(axis=1)]
    filtered_df = filtered_df.loc[~(filtered_df[genres_to_check] == True).all(axis=1)]
    return filtered_df

df = spotify_data[['song_id', 'song_popularity', 'artist_genres', 'album_release_date',  'tempo', 'loudness', 'liveness', 'valence', 'danceability', 'energy', 'acousticness', 'instrumentalness']]
genres_to_check = ['grunge', 'rock']
genre_df = filter_df_with_genres(genres_to_check, df)
genre_df['genre'] = np.where(genre_df['rock'], 'rock', 'grunge')

columns_to_compare = ['tempo', 'loudness', 'liveness', 'danceability', 'energy',  'acousticness']

st.title('Boxplot Explorer')
pick_feature = st.selectbox('Select a feature', columns_to_compare)
boxplot_color = st.color_picker(f'Select color for the boxplot of {pick_feature}', '#1f78b4')
# Create a boxplot
fig, ax = plt.subplots(figsize=(15, 5))
sns.boxplot(x=genre_df["genre"], y=genre_df[pick_feature], ax=ax, color=boxplot_color, linewidth=2)
ax.set_title(f'Boxplot for {pick_feature}')
# Display the plot in Streamlit
st.pyplot(fig)

#Plot 2
def filter_df_with_genres(genres_to_check, df):
    df.loc[:, genres_to_check] = 0
    for genre in genres_to_check:
        df.loc[:, genre] = df['artist_genres'].str.contains(genre, case=False)
    filtered_df = df.loc[~(df[genres_to_check] == False).all(axis=1)]
    filtered_df = filtered_df.loc[~(filtered_df[genres_to_check] == True).all(axis=1)]
    return filtered_df
###########################################################################################################
df = spotify_data[['song_id', 'song_popularity', 'artist_genres', 'album_release_date',  'tempo', 'loudness', 'liveness', 'valence', 'danceability', 'energy', 'acousticness', 'instrumentalness']]
genres_to_check = ['rap', 'r&b']
genre_df = filter_df_with_genres(genres_to_check, df)
genre_df['genre'] = np.where(genre_df['rap'], 'rap', 'r&b')

columns_to_compare1 = ['tempo', 'loudness', 'liveness', 'danceability', 'energy', 'acousticness']

st.title('Rap R&B Boxplot Explorer')
pick_feature1 = st.selectbox('Select a feature', columns_to_compare1, key="pick_feature1")
boxplot_color1 = st.color_picker(f'Select color for the boxplot of {pick_feature1}', '#1f78b4', key="boxplot_color1")

# Create a boxplot
fig1, ax1 = plt.subplots(figsize=(15, 5))
sns.set_palette("bright")
# sns.boxplot(x=genre_df["genre"], y=genre_df[pick_feature1], ax=ax1, color=boxplot_color1)
sns.boxplot(x=genre_df["genre"], y=genre_df[pick_feature1], ax=ax1, color=boxplot_color1, linewidth=2)
ax1.set_title(f'Boxplot for {pick_feature1}')

# Display the plot in Streamlit
st.pyplot(fig1)

#--------------------------------------------------------------------------------------------------------------------------------------
#plot 3
df = spotify_data[['song_id', 'song_popularity', 'artist_genres', 'album_release_date', 'Year']]
genres_to_check = ['rap', 'rock', 'country', 'r&b', 'hip hop']
genre_df = filter_df_with_genres(genres_to_check, df)
df_merged = df[['Year']].groupby('Year').size().reset_index(name='count')
for genre in genres_to_check:
    df_genre_count = genre_df[genre_df[genre]][['Year', genre]]
    df_genre_count = df_genre_count.groupby('Year').size().reset_index(name=genre)
    df_merged = pd.merge(df_merged,df_genre_count, on='Year', how='outer' )
df_merged.fillna(0, inplace=True)
df_merged = df_merged.astype(int)
df_merged.drop(columns=['count'], inplace=True)

df_merged.set_index('Year', inplace=True)

st.title('Heatmap of Genre Counts Over Years')
fig, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(df_merged, cmap="viridis", annot=True, fmt="d", ax=ax)  # cmap is the color map
plt.xlabel('Genres')
plt.ylabel('Year')
plt.title('Heatmap of Feature Counts Over Years')

# Display the plot in Streamlit
st.pyplot(fig)
