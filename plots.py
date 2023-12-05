import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

parent_folder_path  = r"C:\Users\miyan\OneDrive\Documents\my docs\Courses TAMU\Data Viz\project\years"
file_path = f'{parent_folder_path}\combined_audio_features.csv'

# Load the data
df = pd.read_csv(file_path)
features = ['tempo', 'loudness', 'danceability', 'energy', 'valence', 'acousticness', 'instrumentalness']

# Filter the DataFrame to include only the selected features
df_subset = df[features]

# Setting the color palette globally
sns.set_palette("viridis")

# Setting the aesthetic style of the plots
sns.set(style="whitegrid")

# Creating a pairplot in pastel colors
plt.figure(figsize=(10, 10))
sns.pairplot(df_subset, diag_kind='kde', palette="viridis")
plt.suptitle('Matrix Scatter Plot of Audio Features', y=1.02)
plt.show()