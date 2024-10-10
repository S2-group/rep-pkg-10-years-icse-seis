# Import necessary libraries
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# Step 1: Load the BERT model for embedding generation
print("Loading the BERT model...")
model = SentenceTransformer('all-MiniLM-L6-v2')  # A lightweight BERT model for fast processing

# Step 2: Load the Excel file with ID and keywords
file_path = 'data.xlsx'  # Replace with your file path
sheet_name = 'keyword_id_mapping'  # Specify the sheet name
df = pd.read_excel(file_path, sheet_name=sheet_name)

# Ensure the 'ID' and 'KEYWORDS' columns exist
if 'ID' not in df.columns or 'KEYWORDS' not in df.columns:
    raise ValueError("The 'ID' or 'KEYWORDS' columns are missing from the Excel sheet.")

# Step 3: Split the 'KEYWORDS' column by commas and clean up spaces
df['keywords_split'] = df['KEYWORDS'].str.split(',').apply(lambda x: [keyword.strip() for keyword in x])

# Step 4: Create a new DataFrame for IDs and their associated keywords
id_keywords = df.explode('keywords_split')[['ID', 'keywords_split']]

# Step 5: Generate embeddings for the keywords using BERT
print("Generating embeddings for keywords...")
keyword_embeddings = model.encode(id_keywords['keywords_split'].dropna().tolist())

# Step 6: Perform KMeans clustering on the keyword embeddings
print("Clustering the IDs using KMeans...")
num_clusters = 380  # You can adjust the number of clusters as needed
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(keyword_embeddings)

# Step 7: Assign each ID to a cluster based on the common keywords
id_keywords['Cluster'] = kmeans.labels_  # Assign each keyword a cluster label

# Step 8: Create the output DataFrame by grouping by cluster
clustered_ids = id_keywords.groupby('Cluster').agg({
    'ID': lambda x: ', '.join(x.drop_duplicates()),  # Join unique IDs
    'keywords_split': lambda x: ', '.join(x.drop_duplicates()),  # Join keywords
}).reset_index()

# Step 9: Count the number of unique IDs and the number of keywords in each cluster
clustered_ids['Unique ID Count'] = clustered_ids['ID'].apply(lambda x: len(set(x.split(', '))))
clustered_ids['Keyword Count'] = clustered_ids['keywords_split'].apply(lambda x: len(set(x.split(', '))))

# Sort the DataFrame by the Keyword Count in descending order
clustered_ids = clustered_ids.sort_values(by='Keyword Count', ascending=False)

# Rename columns for clarity
clustered_ids = clustered_ids.rename(columns={
    'keywords_split': 'Keywords',
    'ID': 'Unique IDs'
})

# Step 10: Save the results into a new Excel file with two sheets
output_file = 'clusters.xlsx'  # Define output file
with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
    df[['ID', 'KEYWORDS']].to_excel(writer, sheet_name='ID_to_Keyword_Mapping', index=False)
    clustered_ids.to_excel(writer, sheet_name='IDClusters', index=False)

print(f"Clustering results have been saved to '{output_file}'.")
