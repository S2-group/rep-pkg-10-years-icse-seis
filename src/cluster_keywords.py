# Import necessary libraries
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

print("Loading the BERT model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

file_path = 'data.xlsx'
sheet_name = 'keyword_id_mapping'
df = pd.read_excel(file_path, sheet_name=sheet_name)

if 'ID' not in df.columns or 'KEYWORDS' not in df.columns:
    raise ValueError("The 'ID' or 'KEYWORDS' columns are missing from the Excel sheet.")

df['keywords_split'] = df['KEYWORDS'].str.split(',').apply(lambda x: [keyword.strip() for keyword in x])

id_keywords = df.explode('keywords_split')[['ID', 'keywords_split']]

print("Generating embeddings for keywords...")
keyword_embeddings = model.encode(id_keywords['keywords_split'].dropna().tolist())

print("Clustering the IDs using KMeans...")
num_clusters = 380
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(keyword_embeddings)

id_keywords['Cluster'] = kmeans.labels_

clustered_ids = id_keywords.groupby('Cluster').agg({
    'ID': lambda x: ', '.join(x.drop_duplicates()),
    'keywords_split': lambda x: ', '.join(x.drop_duplicates()),
}).reset_index()

clustered_ids['Unique ID Count'] = clustered_ids['ID'].apply(lambda x: len(set(x.split(', '))))
clustered_ids['Keyword Count'] = clustered_ids['keywords_split'].apply(lambda x: len(set(x.split(', '))))

clustered_ids = clustered_ids.sort_values(by='Keyword Count', ascending=False)

clustered_ids = clustered_ids.rename(columns={
    'keywords_split': 'Keywords',
    'ID': 'Unique IDs'
})

output_file = 'clusters.xlsx'  
with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
    df[['ID', 'KEYWORDS']].to_excel(writer, sheet_name='ID_to_Keyword_Mapping', index=False)
    clustered_ids.to_excel(writer, sheet_name='IDClusters', index=False)

print(f"Clustering results have been saved to '{output_file}'.")
