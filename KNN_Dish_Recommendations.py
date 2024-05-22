# Assumes yelp_academic_dataset_business.json in same directory. Github wont let me upload the dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from surprise import Dataset, Reader, KNNWithMeans
from surprise.model_selection import train_test_split as surprise_train_test_split
from surprise import accuracy

# Número de filas a cargar para cada dataset
sample_size = 10000

# Carga una muestra del dataset de negocios
business_data_path = 'yelp_academic_dataset_business.json'
business_df = pd.read_json(business_data_path, lines=True, nrows=sample_size)

# Filtrar solo los registros que corresponden a restaurantes
business_df = business_df[business_df['categories'].str.contains('Restaurants', na=False)]

# Carga una muestra del dataset de reseñas
review_data_path = 'review.json'
review_df = pd.read_json(review_data_path, lines=True, nrows=sample_size)

# Fusionar los datasets de negocios y reseñas en función del business_id
merged_df = pd.merge(review_df, business_df, on='business_id', how='inner')

# Procesamiento de datos
merged_df['is_vegetarian'] = merged_df['categories'].str.contains('Vegetarian', case=False)
merged_df['is_vegan'] = merged_df['categories'].str.contains('Vegan', case=False)
merged_df['is_gluten_free'] = merged_df['categories'].str.contains('Gluten-Free', case=False)
merged_df['is_halal'] = merged_df['categories'].str.contains('Halal', case=False)
merged_df['is_kosher'] = merged_df['categories'].str.contains('Kosher', case=False)
merged_df['is_organic'] = merged_df['attributes'].str.contains('Organic', case=False)
merged_df['is_paleo'] = merged_df['attributes'].str.contains('Paleo', case=False)
merged_df['is_keto'] = merged_df['attributes'].str.contains('Keto', case=False)

# Dividir el DataFrame en conjuntos de entrenamiento y prueba
train_df, test_df = train_test_split(merged_df, test_size=0.2, random_state=42)

# Utilizar el modelo de predicción KNN
reader = Reader(line_format='user item rating', sep=',')
data = Dataset.load_from_df(train_df[['user_id', 'business_id', 'stars']], reader)
trainset, testset = surprise_train_test_split(data, test_size=0.2, random_state=42)

# Entrenar el modelo KNN
algo = KNNWithMeans()
algo.fit(trainset)

# Hacer predicciones en el conjunto de prueba
predictions = algo.test(testset)

# Calcular RMSE
rmse = accuracy.rmse(predictions)
print(f"RMSE: {rmse}")
