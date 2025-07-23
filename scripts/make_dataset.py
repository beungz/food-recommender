import os

import numpy as np
import pandas as pd

import ast
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm
from collections import Counter

from sklearn.preprocessing import StandardScaler

from pytorch_widedeep.preprocessing import TabPreprocessor, WidePreprocessor

from scripts.build_features import get_cluster_id, generate_final_tag_embeddings, get_fallback_tags, process_all_tags, generate_ingredient_embeddings, generate_nutrition_features, user_item_encoder, expand_ffm_features

recipes_data_path = os.path.join("data", "raw", "recipe_interaction", "RAW_recipes.csv")
interactions_data_path = os.path.join("data", "raw", "recipe_interaction", "RAW_interactions.csv")
recipes_serving_data_path = os.path.join("data", "raw", "recipe_interaction", "recipes_w_search_terms.csv")
recipes_techniques_path = os.path.join("data", "raw", "recipe_interaction", "PP_recipes.csv")

processed_data_path = os.path.join("data", "processed")
outputs_data_path = os.path.join("data", "outputs")

wd_model_path = os.path.join("models", "deep_learning")



def prepare_data():
    """
    Prepare data and generate features
    """

    # Load dataset from Kaggle/Food.com
    recipes, recipes_serving, recipes_techniques, interactions = load_fooddotcom_csv_data()

    # Filter only interactins with rating >= 4
    interactions = interactions[interactions["rating"] >= 4].copy()

    # Downsample for resource-efficient training (10% sampling)
    print(f"Before downsampling: {len(interactions)} interactions from {interactions['user_id'].nunique()} users.")
    interactions = downsample_interactions(interactions, frac=0.10, seed=42)
    print(f"After downsampling: {len(interactions)} interactions from {interactions['user_id'].nunique()} users.")

    # Filter only recipes data that has corresponding interactions
    valid_recipe_ids = set(interactions["recipe_id"])
    recipes = recipes[recipes["id"].isin(valid_recipe_ids)].copy()

    # Encode user id and recipe id
    interactions, recipes, user_enc, item_enc= user_item_encoder(interactions, recipes)

    # Save interactions
    interactions_encoded_path = os.path.join(processed_data_path, "interactions_encoded.pkl")
    interactions.to_pickle(interactions_encoded_path)

    # Merge techniques (cooking techniques) data from recipes_techniques to recipes
    recipes = recipes.merge(recipes_techniques[["id", "techniques"]], on="id", how="left")
    recipes['techniques'] = recipes['techniques'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    # Convert servings into integer
    recipes_serving["servings"] = pd.to_numeric(recipes_serving["servings"], errors="coerce").fillna(1).astype(int)

    # Convert search_terms stored as string-encoded lists into real Python lists
    recipes_serving["search_terms"] = recipes_serving["search_terms"].fillna("[]").apply(ast.literal_eval)

    # Merge servings and search_terms from recipes_serving into recipes
    recipes = recipes.merge(recipes_serving[["id", "servings", "search_terms"]], on="id", how="left")

    # Generate nutrition features by converting from list of nutrition amount into seperate columns of numbers
    recipes = generate_nutrition_features(recipes)

    recipes["ingredients"] = recipes["ingredients"].apply(
        lambda x: [i.lower().strip() for i in ast.literal_eval(x)] if isinstance(x, str) else []
    )

    # Generate ingredient_embeddings from list of ingredients, using food2vec model
    recipes, food2vec_model_vectors = generate_ingredient_embeddings(recipes)

    # Get relevant ingredient stats
    all_ingredients, frequent_ingredients = get_ingredient_stat(recipes)

    # For ingredients that are not found in food2vec embeddings, add them back as fallback tags if they have counts exceed frequent_ingredients threshold
    recipes, all_fallback_tags = get_fallback_tags(recipes, food2vec_model_vectors, frequent_ingredients)

    # Process on all tags-like features, and create a combination of tags (tags, search_terms, fallback tags)
    recipes, all_final_tags, all_tags_search_terms = process_all_tags(recipes)

    # Scale numeric features
    num_cols = ["minutes", "kcal", "fat", "sugar", "sodium", "protein", "carb"]
    recipes_unscaled = recipes.copy()

    scaler = StandardScaler()
    recipes[num_cols] = scaler.fit_transform(recipes[num_cols])

    scaler_path = os.path.join(outputs_data_path, "scaler.pkl")
    joblib.dump(scaler, scaler_path)

    # Get final_tag_embeddings by using word2vec on a combination of tags (tags, search_terms, fallback tags)
    recipes, final_tag_corpus = generate_final_tag_embeddings(recipes)

    # Train KMeans Clustering to determine a hidden clusterring on the recipes metadata, and the cluster id will be used as training data
    recipes = get_cluster_id(recipes)

    # Add new features to recipes_unscaled
    recipes_unscaled = recipes_unscaled.merge(
        recipes[['id', 'final_tag_embeddings', 'cluster_vector', 'cluster']],
        on='id',
        how='left'
    )

    # Save recipes, recipes_unscaled
    recipes_path = os.path.join(processed_data_path, "recipes.pkl")
    recipes_unscaled_path = os.path.join(processed_data_path, "recipes_unscaled.pkl")
    recipes.to_pickle(recipes_path)
    recipes_unscaled.to_pickle(recipes_unscaled_path)

    print("\nData preparation done")

    return recipes, recipes_unscaled, interactions, user_enc, item_enc, scaler



def load_fooddotcom_csv_data():
    """
    Load data (Kaggle/Food.com) from CSV into pandas dataframe
    """

    recipes = pd.read_csv(recipes_data_path)
    recipes_serving = pd.read_csv(recipes_serving_data_path)
    recipes_techniques = pd.read_csv(recipes_techniques_path)
    interactions = pd.read_csv(interactions_data_path)

    return recipes, recipes_serving, recipes_techniques, interactions



def downsample_interactions(interactions_df, frac=0.1, seed=42):
    """
    Downsamples the interaction DataFrame by selecting a fraction of the users and keeping all of their interactions.
    """
    np.random.seed(seed)
    unique_users = np.sort(interactions_df["user_id"].unique())
    sampled_users = np.random.choice(unique_users, size=int(len(unique_users) * frac), replace=False)

    return interactions_df[interactions_df["user_id"].isin(sampled_users)].copy()



def get_ingredient_stat(recipes):
    """
    Get ingredient stat, and list of frequent_ingredients, to be used as a cut-off threshold for ingredients to be incorporated as tags (for those not in food2vec embeddings)
    """

    # Count occurrences of each ingredient
    all_ingredients = [ing.strip().lower() for sublist in recipes["ingredients"] for ing in sublist]
    ingredient_counter = Counter(all_ingredients)

    # Set a threshold — ingredients that occur >= 100 times are kept as tags
    threshold = 100
    num_unique_ingredients = len(ingredient_counter)
    frequent_ingredients = {ing for ing, count in ingredient_counter.items() if count >= threshold}
    print(f"Total Unique ingredients: {num_unique_ingredients}, Frequent ingredients (>= {threshold} occurrences): {len(frequent_ingredients)}")

    # Make all_ingredients unique
    all_ingredients_unique = sorted(list(set(all_ingredients)))
    frequent_ingredients_unique = sorted(list(set(frequent_ingredients)))

    # Save all_ingredients
    all_ingredients_path = os.path.join(processed_data_path, "all_ingredients.pkl")
    joblib.dump(all_ingredients_unique, all_ingredients_path)

    frequent_ingredients_path = os.path.join(processed_data_path, "frequent_ingredients.pkl")
    joblib.dump(frequent_ingredients_unique, frequent_ingredients_path)

    return all_ingredients_unique, frequent_ingredients_unique



def stratified_user_split(interactions, recipes, valid_frac=0.15, test_frac=0.15, random_state=42):
    """
    Stratified split the data based on user, so that interaction of User A will be included in train, validation, and test set, based on the specified ratio
    If User A has less than 3 interactions, then incorporate all interactions in only training set.
    """
    np.random.seed(random_state)

    train_rows = []
    valid_rows = []
    test_rows = []

    # Split interaction data for each user to train, validation, and test set
    for uid, group in interactions[['user_id', 'user_id_enc', 'recipe_id', 'item_id_enc', 'rating']].groupby("user_id_enc"):
        group = group.sample(frac=1, random_state=random_state)  # shuffle interactions per user

        n_total = len(group)
        n_test = int(n_total * test_frac)
        n_valid = int(n_total * valid_frac)

        if n_total < 3:
            # Not enough interactions: put all in train
            train_rows.append(group)
        else:
            test_split = group.iloc[:n_test]
            valid_split = group.iloc[n_test:n_test + n_valid]
            train_split = group.iloc[n_test + n_valid:]

            # Ensure train is not empty
            if len(train_split) == 0:
                train_split = valid_split
                valid_split = pd.DataFrame()
            train_rows.append(train_split)
            valid_rows.append(valid_split)
            test_rows.append(test_split)

    train_df = pd.concat(train_rows).reset_index(drop=True)
    valid_df = pd.concat(valid_rows).reset_index(drop=True)
    test_df = pd.concat(test_rows).reset_index(drop=True)

    # Save train_df, valid_df, test_df
    train_df_path = os.path.join(processed_data_path, "train_df.pkl")
    valid_df_path = os.path.join(processed_data_path, "valid_df.pkl")
    test_df_path = os.path.join(processed_data_path, "test_df.pkl")
    train_df.to_pickle(train_df_path)
    valid_df.to_pickle(valid_df_path)
    test_df.to_pickle(test_df_path)

    print(f"Train users: {train_df['user_id_enc'].nunique()}")
    print(f"Valid users: {valid_df['user_id_enc'].nunique()}")
    print(f"Test users: {test_df['user_id_enc'].nunique()}")
    print("Data size before augmentation with negative samples:")
    print(f"Train size: {len(train_df)}, Valid size: {len(valid_df)}, Test size: {len(test_df)}")

    print("\nGenerating negative samples for the training set with n_neg = 5:")
    train_df = generate_test_with_negatives_fast(train_df, interactions, recipes, n_neg=5, n_jobs=-1)
    train_df_with_neg_path = os.path.join(processed_data_path, "train_df_with_neg.pkl")
    train_df.to_pickle(train_df_with_neg_path)

    print("\nGenerating negative samples for the validation set with n_neg = 10:")
    valid_df = generate_test_with_negatives_fast(valid_df, interactions, recipes, n_neg=10, n_jobs=-1)
    valid_df_with_neg_path = os.path.join(processed_data_path, "valid_df_with_neg.pkl")
    valid_df.to_pickle(valid_df_with_neg_path)

    print("\nGenerating negative samples for the test set with n_neg = 20:")
    test_df = generate_test_with_negatives_fast(test_df, interactions, recipes, n_neg=20, n_jobs=-1)
    test_df_with_neg_path = os.path.join(processed_data_path, "test_df_with_neg.pkl")
    test_df.to_pickle(test_df_with_neg_path)

    print("\nStratified user split into train, validation, and test set done")

    return train_df, valid_df, test_df



def generate_test_with_negatives_fast(test_df, interactions_df, all_recipes_df, n_neg=100, n_jobs=-1):
    """
    Generates test set with negative samples. Ensures (user, item) pairs are unique across all rows.
    """

    # Get seen items and all items
    user_item_seen = interactions_df.groupby("user_id_enc")["item_id_enc"].apply(set).to_dict()
    all_items = np.array(all_recipes_df["item_id_enc"].unique())

    # Global set to track duplicates
    seen_pairs = set()

    # Generate negative samples for each row of positive interactions in test_df
    def generate_unique_row_with_negatives(row):
        uid = row["user_id_enc"]
        iid = row["item_id_enc"]
        seen = user_item_seen.get(uid, set())
        candidates = np.setdiff1d(all_items, list(seen), assume_unique=True)

        # Add positive pair
        results = []
        if (uid, iid) not in seen_pairs:
            results.append({"user_id_enc": uid, "item_id_enc": iid, "label": 1})
            seen_pairs.add((uid, iid))

        # Add negatives
        if len(candidates) == 0:
            return results
        elif len(candidates) < n_neg:
            sampled_neg = candidates
        else:
            sampled_neg = np.random.choice(candidates, n_neg, replace=False)

        for neg_iid in sampled_neg:
            if (uid, neg_iid) not in seen_pairs:
                results.append({"user_id_enc": uid, "item_id_enc": neg_iid, "label": 0})
                seen_pairs.add((uid, neg_iid))

        return results

    # Run the function in parallel to reduce computing time
    results_nested = Parallel(n_jobs=n_jobs, backend="loky", verbose=4)(
        delayed(generate_unique_row_with_negatives)(row)
        for row in tqdm(test_df.to_dict(orient="records"))
    )

    # Flatten the results
    results_flat = [item for sublist in results_nested for item in sublist]
    return pd.DataFrame(results_flat)



def merge_recipes_interactions(recipes, train_df, valid_df, test_df):
    """
    Merge train, valid, test data with interactions data, with the relevant recipes metadata
    """

    # Create recipes_small to store all relevant features to be used in training
    num_cols = ["minutes", "kcal", "fat", "sugar", "sodium", "protein", "carb"]
    recipes_small = recipes[["item_id_enc"] + num_cols + ["techniques", "final_tag_embeddings", "ingredient_embeddings", "cluster"]]
    recipes_small.columns = ["item_id_enc"] + num_cols + ["techniques", "final_tag_embeddings", "ingredient_embeddings", "cluster"]
    valid_items = set(recipes_small["item_id_enc"])

    # Merge train, valid, test set of interactions, with recipe_small
    train_merged = train_df.copy()
    train_merged = train_merged[train_merged["item_id_enc"].isin(valid_items)]
    train_merged = train_merged.merge(recipes_small, on="item_id_enc", how="left")

    valid_merged = valid_df.copy()
    valid_merged = valid_merged[valid_merged["item_id_enc"].isin(valid_items)]
    valid_merged = valid_merged.merge(recipes_small, on="item_id_enc", how="left")

    test_merged = test_df.copy()
    test_merged = test_merged[test_merged["item_id_enc"].isin(valid_items)]
    test_merged = test_merged.merge(recipes_small, on="item_id_enc", how="left")

    # Drop columns that will not be used in training
    columns_to_drop = ["user_id", "recipe_id"]
    train_merged = train_merged.drop(columns=[col for col in columns_to_drop if col in train_merged.columns])
    valid_merged = valid_merged.drop(columns=[col for col in columns_to_drop if col in valid_merged.columns])
    test_merged = test_merged.drop(columns=[col for col in columns_to_drop if col in test_merged.columns])

    # Save train_merged, valid_merged, test_merged
    train_merged_path = os.path.join(processed_data_path, "train_merged.pkl")
    valid_merged_path = os.path.join(processed_data_path, "valid_merged.pkl")
    test_merged_path = os.path.join(processed_data_path, "test_merged.pkl")

    train_merged.to_pickle(train_merged_path)
    valid_merged.to_pickle(valid_merged_path)
    test_merged.to_pickle(test_merged_path)

    print("\nRecipe-interaction merge done")

    return train_merged, valid_merged, test_merged



def write_ffm_file_from_expanded_df(ffm_df, output_path, label_col="label", progress_every=0.20):
    """
    Write FFM files, which will be used to train FFM model, based on FFM definition
    """
    field_map = {col: i for i, col in enumerate(ffm_df.columns) if col != label_col}
    total_rows = len(ffm_df)
    step = max(1, int(total_rows * progress_every))

    with open(output_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(ffm_df.itertuples(index=False)):
            label = getattr(row, label_col)
            fields = []

            for col, field_id in field_map.items():
                val = getattr(row, col)
                if isinstance(val, (int, float)) and val != 0:
                    fields.append(f"{field_id}:{field_id}:{val:.6f}")

            f.write(f"{label} {' '.join(fields)}\n")

            # Log progress
            if i % step == 0 or i == total_rows - 1:
                percent = (i + 1) / total_rows * 100
                print(f"Progress: {percent:.1f}% ({i + 1:,}/{total_rows:,} rows)")

    print("FFM file write complete.")

    return



def prepare_ffm_data(train_merged, valid_merged, test_merged):
    """
    Prepare data for FFM model training
    """
    # Paths to FFM files
    ffm_train_path = os.path.join(outputs_data_path, "ffm_train.txt")
    ffm_valid_path = os.path.join(outputs_data_path, "ffm_valid.txt")
    ffm_test_path = os.path.join(outputs_data_path, "ffm_test.txt")

    # Paths to raw data that is used to convert into FFM files
    train_ffm_df_path = os.path.join(outputs_data_path, "train_ffm_df.pkl")
    valid_ffm_df_path = os.path.join(outputs_data_path, "valid_ffm_df.pkl")
    test_ffm_df_path = os.path.join(outputs_data_path, "test_ffm_df.pkl")

    # Expand feature vector into separate columns, save it, convert into FFM format, then write FFM files
    print("\nWriting ffm_train.txt")
    train_ffm_df = expand_ffm_features(train_merged)
    train_ffm_df.to_pickle(train_ffm_df_path)
    write_ffm_file_from_expanded_df(train_ffm_df, ffm_train_path, label_col="label")

    print("\nWriting ffm_valid.txt")
    valid_ffm_df = expand_ffm_features(valid_merged)
    valid_ffm_df.to_pickle(valid_ffm_df_path)
    write_ffm_file_from_expanded_df(valid_ffm_df, ffm_valid_path, label_col="label")

    print("\nWriting ffm_test.txt")
    test_ffm_df = expand_ffm_features(test_merged)
    test_ffm_df.to_pickle(test_ffm_df_path)
    write_ffm_file_from_expanded_df(test_ffm_df, ffm_test_path, label_col="label")

    print("\nFFM data preparation done")

    return train_ffm_df, valid_ffm_df, test_ffm_df


def prepare_wd_data(train_ffm_df, valid_ffm_df, test_ffm_df):
    """
    Prepare data for WD model training
    """
    # Identify sparse and dense columns
    sparse_cols = ["user_id_enc", "item_id_enc"]
    dense_cols = [col for col in train_ffm_df.columns if col not in ["label"] + sparse_cols]

    # Set tab and wide preprocessor
    tab_preprocessor = TabPreprocessor(
        embed_cols=sparse_cols,
        continuous_cols=dense_cols
    )

    wide_preprocessor = WidePreprocessor(wide_cols=["user_id_enc", "item_id_enc"])

    # Convert data into tab/wide format
    X_tab_train = tab_preprocessor.fit_transform(train_ffm_df)
    X_wide_train = wide_preprocessor.fit_transform(train_ffm_df)
    y_train = train_ffm_df["label"].values

    X_tab_valid = tab_preprocessor.transform(valid_ffm_df)
    X_wide_valid = wide_preprocessor.transform(valid_ffm_df)
    y_valid = valid_ffm_df["label"].values

    X_tab_test = tab_preprocessor.transform(test_ffm_df)
    X_wide_test = wide_preprocessor.transform(test_ffm_df)
    y_test = test_ffm_df["label"].values

    # Save the preprocessors
    tab_preprocessor_path = os.path.join(wd_model_path, "tab_preprocessor.pkl")
    wide_preprocessor_path = os.path.join(wd_model_path, "wide_preprocessor.pkl")
    joblib.dump(tab_preprocessor, tab_preprocessor_path)
    joblib.dump(wide_preprocessor, wide_preprocessor_path)
    
    print("\nWD data preparation done")

    return tab_preprocessor, wide_preprocessor, X_tab_train, X_wide_train, y_train, X_tab_valid, X_wide_valid, y_valid, X_tab_test, X_wide_test, y_test