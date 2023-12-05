import pandas as pd
import os

parent_folder_path  = r"C:\Users\miyan\OneDrive\Documents\my docs\Courses TAMU\Data Viz\project\years"

data_albums_df = pd.DataFrame()
data_artists_df = pd.DataFrame()
audio_features_df = pd.DataFrame()
data_tracks_df = pd.DataFrame()


# Iterate through subdirectories
for year_folder in os.listdir(parent_folder_path):
    year_folder_path = os.path.join(parent_folder_path, year_folder)
    # Check if the item is a directory
    if os.path.isdir(year_folder_path):
        # Get a list of CSV files in the subdirectory
        csv_files = [f for f in os.listdir(year_folder_path) if f.endswith('.csv')]
        # Iterate through each CSV file in the subdirectory
        for csv_file in csv_files:
            # Check if the CSV file starts with "data_albums"
            if csv_file.startswith("data_albums"):
                # Construct the full path to the CSV file
                file_path = os.path.join(year_folder_path, csv_file)
                try:
                    # Read the CSV file using pandas
                    df = pd.read_csv(file_path)
                    # Add a column indicating the year to differentiate data
                    df['Year'] = int(year_folder)
                    # Append the DataFrame to the data_albums_df
                    data_albums_df = pd.concat([data_albums_df, df], ignore_index=True)
                except pd.errors.EmptyDataError:
                    print(f"Warning: {csv_file} in {year_folder} is empty and will be skipped.")
                except pd.errors.ParserError:
                    print(f"Warning: Error parsing {csv_file} in {year_folder}. Check if the file has a valid CSV format.")
# Display the merged data_albums DataFrame
print("Merged data_albums DataFrame:")
print(data_albums_df.head())


# Iterate through subdirectories
for year_folder in os.listdir(parent_folder_path):
    year_folder_path = os.path.join(parent_folder_path, year_folder)
    # Check if the item is a directory
    if os.path.isdir(year_folder_path):
        # Get a list of CSV files in the subdirectory
        csv_files = [f for f in os.listdir(year_folder_path) if f.endswith('.csv')]
        # Iterate through each CSV file in the subdirectory
        for csv_file in csv_files:
            # Check if the CSV file starts with "data_albums"
            if csv_file.startswith("data_artists"):
                # Construct the full path to the CSV file
                file_path = os.path.join(year_folder_path, csv_file)
                try:
                    # Read the CSV file using pandas
                    df = pd.read_csv(file_path)
                    # Append the DataFrame to the data_albums_df
                    data_artists_df = pd.concat([data_artists_df, df], ignore_index=True)
                except pd.errors.EmptyDataError:
                    print(f"Warning: {csv_file} in {year_folder} is empty and will be skipped.")
                except pd.errors.ParserError:
                    print(f"Warning: Error parsing {csv_file} in {year_folder}. Check if the file has a valid CSV format.")
# Display the merged data_albums DataFrame
print("Merged data_artists DataFrame:")
print(data_artists_df.head())


# Iterate through subdirectories
for year_folder in os.listdir(parent_folder_path):
    year_folder_path = os.path.join(parent_folder_path, year_folder)
    # Check if the item is a directory
    if os.path.isdir(year_folder_path):
        # Get a list of CSV files in the subdirectory
        csv_files = [f for f in os.listdir(year_folder_path) if f.endswith('.csv')]
        # Iterate through each CSV file in the subdirectory
        for csv_file in csv_files:
            # Check if the CSV file starts with "data_albums"
            if csv_file.startswith("data_audio_features"):
                # Construct the full path to the CSV file
                file_path = os.path.join(year_folder_path, csv_file)
                try:
                    # Read the CSV file using pandas
                    df = pd.read_csv(file_path)
                    # Append the DataFrame to the data_albums_df
                    audio_features_df = pd.concat([audio_features_df, df], ignore_index=True)
                except pd.errors.EmptyDataError:
                    print(f"Warning: {csv_file} in {year_folder} is empty and will be skipped.")
                except pd.errors.ParserError:
                    print(f"Warning: Error parsing {csv_file} in {year_folder}. Check if the file has a valid CSV format.")
# Display the merged data_albums DataFrame
print("Merged audio_features DataFrame:")
print(audio_features_df.head())


# Iterate through subdirectories
for year_folder in os.listdir(parent_folder_path):
    year_folder_path = os.path.join(parent_folder_path, year_folder)
    # Check if the item is a directory
    if os.path.isdir(year_folder_path):
        # Get a list of CSV files in the subdirectory
        csv_files = [f for f in os.listdir(year_folder_path) if f.endswith('.csv')]
        # Iterate through each CSV file in the subdirectory
        for csv_file in csv_files:
            # Check if the CSV file starts with "data_albums"
            if csv_file.startswith("data_tracks"):
                # Construct the full path to the CSV file
                file_path = os.path.join(year_folder_path, csv_file)
                try:
                    # Read the CSV file using pandas
                    df = pd.read_csv(file_path)
                    # Append the DataFrame to the data_albums_df
                    data_tracks_df = pd.concat([data_tracks_df, df], ignore_index=True)
                except pd.errors.EmptyDataError:
                    print(f"Warning: {csv_file} in {year_folder} is empty and will be skipped.")
                except pd.errors.ParserError:
                    print(f"Warning: Error parsing {csv_file} in {year_folder}. Check if the file has a valid CSV format.")
# Display the merged data_albums DataFrame
print("Merged data_tracks DataFrame:")
print(data_tracks_df.head())

merged_df = pd.merge(data_tracks_df, audio_features_df, on="song_id", how="inner")
merged_df = pd.merge(merged_df, data_albums_df, on="album_id", how="inner")
merged_df = pd.merge(merged_df, data_artists_df, on="artist_id", how="inner")

df_final = merged_df.drop_duplicates()

df_cleaned = df_final.dropna(subset=['album_release_date'])
df_cleaned.drop(columns="album_genres", inplace=True)
df_cleaned['album_release_date'] = pd.to_datetime(df_cleaned['album_release_date'], format='%Y-%m-%d', errors='coerce')
df_cleaned = df_cleaned.dropna()
df_cleaned.drop_duplicates(subset=['song_id'], inplace=True)
# df_cleaned['artist_genres'] = df_cleaned['artist_genres'].str.split(', ')

output_file_path = os.path.join(parent_folder_path, 'spotify_combined_data.csv')
df_cleaned.to_csv(output_file_path, index=False)