import os

import numpy as np
import pandas as pd

import ast
import joblib
from tqdm import tqdm
from collections import Counter

from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

from gensim.models import Word2Vec

outputs_data_path = os.path.join("data", "outputs")



def get_cluster_id(recipes):
    """
    Train KMeans Clustering to determine hidden clustering in recipe metadata, and use resulting cluster id in training of WD and FFM model
    """
    print("Train KMeans Clustering to identify clustering in food recipes:")
    tqdm.pandas()

    # Train KMeans Clustering, using cluster_vector as training data, and target number of cluster at 20
    recipes["cluster_vector"] = recipes.progress_apply(build_recipe_vector, axis=1)
    X = np.stack(recipes["cluster_vector"].values)

    kmeans = KMeans(n_clusters=20, random_state=42, n_init='auto')
    recipes["cluster"] = kmeans.fit_predict(X)

    # Count cluster sizes
    cluster_counts = recipes["cluster"].value_counts().sort_index()

    # Get the list of small clusters (< 100 counts), which will be grouped as one cluster
    threshold = 100
    small_cluster_ids = cluster_counts[cluster_counts < threshold].index.tolist()

    # Temporarily reassign small clusters to a new ID (999)
    recipes["cluster"] = recipes["cluster"].apply(
        lambda c: 999 if c in small_cluster_ids else c
    )

    # Re-map all cluster IDs to new consecutive integers
    unique_clusters = sorted(recipes["cluster"].unique())
    cluster_id_mapping = {old: new for new, old in enumerate(unique_clusters)}
    recipes["cluster"] = recipes["cluster"].map(cluster_id_mapping)

    # Print counts for each cluster id
    cluster_counts = recipes["cluster"].value_counts().sort_index()
    print(cluster_counts)

    return recipes



def build_recipe_vector(row):
    """
    Build recipe vector comprising of nutrition data, ingredient_embeddings, and (cooking) technique vector
    """
    nutrition_vector = np.array([
                                    row["kcal"], row["fat"], row["sugar"], row["sodium"],
                                    row["protein"], row["sat_fat"], row["carb"], row["minutes"]
                                ], dtype=np.float32)
    ingredient_embeddings_vector = np.array(row['ingredient_embeddings'], dtype=np.float32)
    techniques_vector = np.array(row['techniques'], dtype=np.float32)

    return np.concatenate([nutrition_vector, ingredient_embeddings_vector, techniques_vector])



def generate_final_tag_embeddings(recipes):
    """
    Generate final tag embeddings using Word2Vec
    """

    final_tag_corpus = recipes["final_tags"].dropna().tolist()

    # Use dimension = 50 for final tags embeddings
    embedding_dim = 50

    # Define tag2vec model
    tag2vec_model = Word2Vec(
        sentences=final_tag_corpus,
        vector_size=embedding_dim,
        window=5,
        min_count=1,
        sg=1,           # Skip-gram
        workers=16,
        seed=42,
        epochs=100
    )

    # Save tag2vec
    tag2vec_path = os.path.join(outputs_data_path, "tag2vec.model")
    tag2vec_model.save(tag2vec_path)

    # Generate embeddings
    recipes["final_tag_embeddings"] = recipes["final_tags"].apply(
        lambda tags: get_tag_embedding(tags, tag2vec_model, embedding_dim)
    )

    return recipes, final_tag_corpus



def get_tag_embedding(tags, model, dim):
    """
    Generate average embeddings of final tags using Word2Vec
    """
    vectors = [model.wv[tag] for tag in tags if tag in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(dim)



def get_fallback_tags(recipes, food2vec_model_vectors, frequent_ingredients):
    """
    Get fallback tags, from list of ingredients that cannot be found in food2vec model vectors
    """
    # Get list of ingredients that cannot be matched in food2vec vectors, and put them in fallback_tags
    recipes["fallback_tags"] = recipes["ingredients"].apply(
        lambda ings: process_ingredients_with_fallback(ings, food2vec_model_vectors, frequent_ingredients)
    )

    all_fallback_tags = [ing.strip().lower() for sublist in recipes["fallback_tags"] for ing in sublist]
    fallback_tags_counter = Counter(all_fallback_tags)

    # Determine the number of unique fallback tags
    num_unique_fallback_tags = len(fallback_tags_counter)
    print(f"Total unique fallback tags: {num_unique_fallback_tags}")

    return recipes, all_fallback_tags



def process_ingredients_with_fallback(ingredient_list, embedding_vocab, fallback_set):
    """
    Get list of ingredients that cannot be matched in food2vec vectors, and put them in fallback_tags
    """
    final_tags = []
    for ing in ingredient_list:
        ing_lower = ing.strip().lower()
        if ing_lower in embedding_vocab:
            continue  # already used in embedding
        elif ing_lower in fallback_set:
            final_tags.append(ing_lower)  # treat as tag
    return final_tags



def process_all_tags(recipes):
    """
    Process search_terms, tags, and final_tags, and update them in recipes
    """

    # Process search_terms
    recipes['search_terms'] = recipes['search_terms'].apply(lambda x: list(x))
    all_search_tags = [ing.strip().lower() for sublist in recipes["search_terms"] for ing in sublist]
    search_tags_counter = Counter(all_search_tags)

    num_unique_search_tags = len(search_tags_counter)
    print(f"Total unique search terms: {num_unique_search_tags}")

    # Process tags
    recipes["tags"] = recipes["tags"].apply(
        lambda x: [i.lower().strip() for i in ast.literal_eval(x)] if isinstance(x, str) else []
    )
    all_tags = [ing.strip().lower() for sublist in recipes["tags"] for ing in sublist]
    tags_counter = Counter(all_tags)

    num_unique_tags = len(tags_counter)
    print(f"Total unique tags: {num_unique_tags}")

    # Process final_tags
    recipes['final_tags'] = recipes.apply(
        lambda row: row['search_terms'] + row['fallback_tags'] + row['tags'],
        axis=1
    )
    all_final_tags = [ing.strip().lower() for sublist in recipes["final_tags"] for ing in sublist]
    final_tags_counter = Counter(all_final_tags)

    num_unique_final_tags = len(final_tags_counter)
    print(f"Total unique final tags: {num_unique_final_tags}")

    return recipes



def generate_ingredient_embeddings(recipes):
    """
    Generate ingredient embeddings using food2vec
    """
    # Load food2vec model (model.bin / binary Word2Vec format)
    food2vec_path = os.path.join(outputs_data_path, "food2vec_model.bin")
    food2vec_model_vectors = Word2Vec.load(food2vec_path).wv

    # Print general stats of food2vec
    num_food2vec_embeddings = len(food2vec_model_vectors.key_to_index)
    food2vec_embeddings_dim = food2vec_model_vectors.vector_size
    print(f"Loaded {num_food2vec_embeddings} ingredient embeddings from food2vec model, with dimension {food2vec_embeddings_dim}")

    # Calculate average of ingredient embedding for each ingredient
    def get_mean_vector(ingredient_list, embedding_vocab=food2vec_model_vectors, embedding_dim=100):
        vectors = [embedding_vocab[word] for word in ingredient_list if word in embedding_vocab]
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(embedding_dim)  # or use np.nan if you want to filter later

    # Compute mean embedding per recipe
    recipes["ingredient_embeddings"] = recipes["ingredients"].apply(get_mean_vector)

    return recipes, food2vec_model_vectors



def generate_nutrition_features(recipes):
    """
    Generate nutrition features (numeric, per serving, as separate columns), based on a list of nutrition in original dataset
    """
    # Parse nutrition data from a list of nutrition, and concatenate with recipes, each as a separate numeric column
    recipes = pd.concat([recipes, recipes["nutrition"].apply(parse_nutrition).apply(pd.Series)], axis=1)

    nutr_cols = ["kcal", "fat", "sugar", "sodium", "protein", "sat_fat", "carb"]

    # Avoid divide-by-zero due to null/zero servings
    recipes["servings"] = recipes["servings"].replace(0, 1)

    # Calculate nutrition per serving
    for col in nutr_cols:
        recipes[col] = recipes[col] / recipes["servings"]

    # Drop rows with NA
    recipes = recipes.dropna(subset=["minutes", "kcal", "fat", "sugar", "sodium", "protein", "carb", "techniques"])

    return recipes



def parse_nutrition(nutrition_str):
    """
    Parse nutrition data from a list of nutrition
    """
    try:
        vals = eval(nutrition_str)
        return {"kcal": vals[0], "fat": vals[1], "sugar": vals[2],
                "sodium": vals[3], "protein": vals[4], "sat_fat": vals[5], "carb": vals[6]}
    except:
        return None
    


def user_item_encoder(interactions, recipes):
    """
    Encode user id and recipe id
    """
    # Initialize encoder for user id and recipe id
    user_enc = LabelEncoder()
    item_enc = LabelEncoder()

    # Encode user id and recipe id
    interactions["user_id_enc"] = user_enc.fit_transform(interactions["user_id"])
    interactions["item_id_enc"] = item_enc.fit_transform(interactions["recipe_id"])
    recipes["item_id_enc"] = item_enc.transform(recipes["id"])

    # Save the encoders
    user_enc_path = os.path.join(outputs_data_path, "user_encoder.pkl")
    item_enc_path = os.path.join(outputs_data_path, "item_encoder.pkl")
    joblib.dump(user_enc, user_enc_path)
    joblib.dump(item_enc, item_enc_path)

    return interactions, recipes, user_enc, item_enc



def expand_ffm_features(df):
    """
    Expand ingredient_embeddings, final_tag_embeddings, and techniques, from vectors into separate columns, which will be used in training of both WD and FFM model
    """
    df = df.reset_index(drop=True)
    
    # Use np.array for flattening
    ing = np.stack(df["ingredient_embeddings"].values)
    tag = np.stack(df["final_tag_embeddings"].values)
    tech = np.stack(df["techniques"].values)
    
    # Convert directly to dataFrame
    ing_df = pd.DataFrame(ing, columns=[f"ing_{i}" for i in range(ing.shape[1])])
    tag_df = pd.DataFrame(tag, columns=[f"tag_{i}" for i in range(tag.shape[1])])
    tech_df = pd.DataFrame(tech, columns=[f"tech_{i}" for i in range(tech.shape[1])])
    
    # Concatenate without copying original (unless needed)
    ffm_df = pd.concat([df.drop(columns=["ingredient_embeddings", "final_tag_embeddings", "techniques"]),
                        ing_df, tag_df, tech_df], axis=1)
    
    # Ensure float columns are float64, int columns are int64
    for col in ffm_df.columns:
        if ffm_df[col].dtype == "float32":
            ffm_df[col] = ffm_df[col].astype("float64")
        elif ffm_df[col].dtype == "int32":
            ffm_df[col] = ffm_df[col].astype("int64")
    
    return ffm_df