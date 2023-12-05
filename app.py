import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import base64

cwd = os.getcwd()
parent_folder_path  = rf"{cwd}\years"
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
image_local('background2.jpg')

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

#################### KOMAL ##################################
#plot 1----------------------------------------------------------------------------------------------
file_path = f'{parent_folder_path}\combined_artists.csv'
df = pd.read_csv(file_path)
df = df.dropna(subset=['artist_genres'])
df['artist_genres'] = df['artist_genres'].str.split(', ').explode('artist_genres')

# Identify top ten common genres across all years
genres_by_year = df.groupby('year')['artist_genres'].apply(set)
common_genres = set.intersection(*genres_by_year)
top_common_genres = df[df['artist_genres'].isin(common_genres)]['artist_genres'].value_counts().nlargest(15).index

# Streamlit app
st.title('Genre Popularity Over Time')

# Selector for each line
selected_genres = st.multiselect('Select Genres', top_common_genres)

# Aggregate the data
df_filtered = df[df['artist_genres'].isin(top_common_genres)]
df_grouped = df_filtered.groupby(['year', 'artist_genres'])['artist_popularity'].mean().reset_index()

# Visualize the trends for all selected genres
fig, ax = plt.subplots(figsize=(12, 8))

# Plot all genres
for genre in top_common_genres:
    genre_data = df_grouped[df_grouped['artist_genres'] == genre]
    ax.plot(genre_data['year'], genre_data['artist_popularity'], label=genre, alpha=0.15)

# Highlight selected genres
for selected_genre in selected_genres:
    selected_genre_data = df_grouped[df_grouped['artist_genres'] == selected_genre]
    ax.plot(selected_genre_data['year'], selected_genre_data['artist_popularity'], label=f'{selected_genre} (Selected)', linewidth=2.5)

# Set labels and title
ax.set_xlabel('Year')
ax.set_ylabel('Average Popularity')
ax.set_title('Trends of Genre Popularity Over Time')
ax.legend()

# Display the plot in Streamlit
st.pyplot(fig)

## plot 2 -------------------------------------------------------------------------
st.title('Audio-feaure Pairplot')
file_path = f'{parent_folder_path}\combined_audio_features.csv'
df = pd.read_csv(file_path)
features = ['tempo', 'loudness', 'danceability', 'energy', 'valence', 'acousticness', 'instrumentalness']
df_subset = df[features]
sns.set(style="whitegrid")
# plt.figure(figsize=(10, 10))
pair_plot = sns.pairplot(df_subset, diag_kind='kde')
pair_plot.fig.suptitle('Matrix Scatter Plot of Audio Features', y=1.02)
# plt.show()
st.pyplot(pair_plot)

# plot 3----------------------------------------------------------------------------------------------

# Add a year_group column
bins = [1994, 1999, 2004, 2009, 2014, 2019, 2024]
labels = ['1994-1998', '1999-2003', '2004-2008', '2009-2013', '2014-2018', '2019-2023']
df['year_group'] = pd.cut(df['year'], bins=bins, labels=labels)

# Selecting relevant features for the violin plots
features = ['tempo', 'loudness', 'danceability', 'energy', 'valence', 'acousticness', 'instrumentalness']

# Streamlit app
st.title("Violin Plots for Audio Features Over Time")

# Allow user to select a feature
selected_feature = st.selectbox("Select a feature", features)

# Create a figure to hold the subplot
fig, ax = plt.subplots(figsize=(15, 8))

# Use a pastel palette for the violin plots
palette = sns.color_palette("pastel")

# Create violin plots with a fixed width
sns.violinplot(x='year_group', y=selected_feature, data=df, scale='width',
               inner=None, color="#ADADF7")
# Overlay a box plot
sns.boxplot(x='year_group', y=selected_feature, data=df,
            whis=[25, 75], width=0.1, fliersize=0, color='k', boxprops=dict(alpha=.3))
medians = df.groupby('year_group')[selected_feature].median()
median_positions = np.arange(len(labels))
ax.scatter(median_positions, medians, color='white', s=30, zorder=3)
ax.plot(median_positions, medians, color='magenta', linestyle='-', linewidth=2, zorder=2)

# Set labels and title
ax.set_title(f"{selected_feature.capitalize()} Over Time")
ax.set_xlabel('Year Group')
ax.set_ylabel(selected_feature.capitalize())

# Rotate x-axis labels
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

# Display the plot in Streamlit
st.pyplot(fig)

# plot 4 ---------------------------------------------------------------------------------------------
import plotly.graph_objects as go

# Load your combined artist data
df = pd.read_csv(f'{parent_folder_path}/combined_artists.csv')

# Preprocessing the data for Sankey diagram
df = df.dropna(subset=['artist_genres'])
df['artist_genres'] = df['artist_genres'].str.split(', ').explode('artist_genres')
bins = list(range(1994, 2025, 5))
labels = [f'{i}-{i+4}' for i in bins[:-1]]
df['year_group'] = pd.cut(df['year'], bins=bins, labels=labels, right=False)
top_genres = df['artist_genres'].value_counts().nlargest(15).index
df_filtered = df[df['artist_genres'].isin(top_genres)]
# df_grouped = df_filtered.groupby(['year_group', 'artist_genres'])['artist_popularity'].mean().reset_index()
df_grouped = df_filtered.groupby(['year_group', 'artist_genres'], observed=True)['artist_popularity'].mean().reset_index()


# Creating the labels for the Sankey diagram
year_labels = df_grouped['year_group'].unique().tolist()
genre_labels = df_grouped['artist_genres'].unique().tolist()
all_labels = list(dict.fromkeys(year_labels + genre_labels))

# Create source-target pairs for Year Group -> Genre, ensuring the indexing is correct
source = [year_labels.index(year_group) for year_group in df_grouped['year_group']]
target = [len(year_labels) + genre_labels.index(genre) for genre in df_grouped['artist_genres']]
value = df_grouped['artist_popularity'].tolist()

# Plot the Sankey diagram
sankey_fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=all_labels
    ),
    link=dict(
        source=source,
        target=target,
        value=value
    ))])
# sankey_fig.update_layout(title_text="Sankey Diagram of Top 15 Genre Popularity Across 5-Year Intervals", font_size=10)
st.title('Sankey Diagram of Top 15 Genre Popularity Across 5-Year Intervals')
st.plotly_chart(sankey_fig)

#plot 5-------------------------------------------------------------------------------------------------------------------
df_tracks = pd.read_csv(f'{parent_folder_path}/combined_tracks.csv')
# Convert song_duration from milliseconds to minutes
df_tracks = pd.read_csv(f'{parent_folder_path}/combined_tracks.csv')

# Convert song_duration from milliseconds to minutes
df_tracks['song_duration_min'] = df_tracks['song_duration'] / 60000
# Convert 'year' to datetime and extract the year for grouping
df_tracks['year_dt'] = pd.to_datetime(df_tracks['year'], format='%Y')
df_tracks['year'] = pd.to_datetime(df_tracks['year'], format='%Y').dt.year
# Calculate the median song duration for each year
duration_by_year = df_tracks.groupby('year')['song_duration_min'].median()

# Streamlit app
st.title('Song Duration Analysis Over Years')

# Line chart for song duration trend over the years
st.subheader('Trend of Song Duration Over Years')
fig_line = plt.figure(figsize=(10, 6))
plt.plot(duration_by_year.index, duration_by_year, marker='o', color='teal', linestyle='-')
plt.title('Trend of Song Duration Over Years')
plt.xlabel('Year')
plt.ylabel('Median Song Duration (minutes)')
plt.grid(True)
st.pyplot(fig_line)

# Box plots for song duration by year
st.subheader('Song Duration Trend Over Years (Box Plots)')
fig_box = plt.figure(figsize=(10, 6))
boxplot = sns.boxplot(x='year', y='song_duration_min', data=df_tracks, color='purple', showfliers=False)
plt.title('Song Duration Trend Over Years')
plt.xlabel('Year')
plt.ylabel('Song Duration (minutes)')
plt.ylim(1, 8)
plt.xticks(rotation=45)
st.pyplot(fig_box)