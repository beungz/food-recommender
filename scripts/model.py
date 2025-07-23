import os

import numpy as np
import pandas as pd

import shutil 
import gc

import torch
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F

from pytorch_widedeep.models import Wide, TabMlp, WideDeep
from pytorch_widedeep.training import Trainer

import xlearn as xl

from recommenders.utils.timer import Timer
from recommenders.evaluation.python_evaluation import recall_at_k, map_at_k, ndcg_at_k
from recommenders.tuning.parameter_sweep import generate_param_grid

from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

from scripts.build_features import expand_ffm_features
from scripts.make_dataset import write_ffm_file_from_expanded_df

ffm_model_path = os.path.join("models", "classical_machine_learning")
wd_model_path = os.path.join("models", "deep_learning")
outputs_data_path = os.path.join("data", "outputs")
processed_data_path = os.path.join("data", "processed")



def ffm_train_model(test_merged):
    """
    Trains an FFM model and evaluates metrics at various k values.
    """

    # Set file paths for FFM training, validation, and test datasets
    ffm_train_path = os.path.join(outputs_data_path, "ffm_train.txt")
    ffm_valid_path = os.path.join(outputs_data_path, "ffm_valid.txt")
    ffm_test_path = os.path.join(outputs_data_path, "ffm_test.txt")

    ffm_modelout_path = os.path.join(ffm_model_path, "model.out")
    ffm_output_path = os.path.join(ffm_model_path, "output.txt")

    # Initialize FFM Model
    ffm_model = xl.create_ffm()
    ffm_model.setTrain(ffm_train_path)
    ffm_model.setValidate(ffm_valid_path)

    # Set model parameters
    LEARNING_RATE = 0.01        # learning rate
    LAMBDA = 0.0001             # regular lambda
    EPOCH = 15                  # number of epochs
    OPT_METHOD = "adagrad"      # optimization method: "sgd", "adagrad" and "ftrl"
    METRIC = "auc"              # evaluation metric: "acc", "prec", "f1" and "auc"
    K = 8                       # Latent factor dimension

    param = {
        "task": "binary",
        "lr": LEARNING_RATE,
        "lambda": LAMBDA,
        "metric": METRIC,
        "epoch": EPOCH,
        "opt": OPT_METHOD,
        "k": K
    }

    # Train FFM model
    with Timer() as time_train:
        print("Training the model...")
        ffm_model.fit(param, ffm_modelout_path)
    print(f"Training time: {time_train}")

    # Make prediction on test set
    ffm_model.setTest(ffm_test_path)
    ffm_model.setSigmoid()  # Convert output to probabilities in [0, 1]
    ffm_model.predict(ffm_modelout_path, ffm_output_path)

    # Retrieve prediction scores from output file
    with open(ffm_output_path, encoding='utf-8') as f:
        test_predictions = [float(x.strip()) for x in f.readlines()]
    pred_df = test_merged.copy()
    pred_df["score"] = test_predictions

    # Evaluate metrics on test set for various k
    eval_results = evaluate_topk(pred_df, k_list=[5, 10, 20])

    return ffm_model, test_predictions, eval_results



def ffm_train_model_hyperparam_tuning(test_merged, param_dict):
    """
    Performs hyperparameter tuning for the FFM model using the provided parameter grid.
    """

    # Set file paths for FFM training, validation, and test datasets
    ffm_train_path = os.path.join(outputs_data_path, "ffm_train.txt")
    ffm_valid_path = os.path.join(outputs_data_path, "ffm_valid.txt")
    ffm_test_path = os.path.join(outputs_data_path, "ffm_test.txt")

    ffm_modelout_path = os.path.join(ffm_model_path, "model.out")
    ffm_output_path = os.path.join(ffm_model_path, "output.txt")

    best_ffm_modelout_path = os.path.join(ffm_model_path, "best_model.out")
    best_ffm_output_path = os.path.join(ffm_model_path, "best_output.txt")

    test_merged_path = os.path.join(processed_data_path, "test_merged.pkl")

    # Generate parameter grid from dictionary
    param_grid = generate_param_grid(param_dict)

    # Create a DataFrame to store tuning results
    tuned_results = pd.DataFrame(columns=["task", "lr", "lambda", "k", "opt", "epoch", "recall_k_200", "recall_k_100", "recall_k_50", "recall_k_20", "recall_k_10"])

    # Initialize recall
    best_map_k_10 = 0.0
    best_ffm_model = None
    best_test_predictions = None
    best_eval_result = None

    # Clean up memory
    clear_memory_freq = 3
    count = 0
    gc.collect()

    # Loop through all set of parameters
    for param in param_grid:
        with Timer() as time_train:
            print(f"\nParameter Set {count + 1}/{len(param_grid)}")
            print(f"Training FFM with: {param}")

            # Initialize FFM Model
            ffm_model = xl.create_ffm()
            ffm_model.setTrain(ffm_train_path)
            ffm_model.setValidate(ffm_valid_path)

            # Train FFM model
            ffm_model.fit(param, ffm_modelout_path)
            print(f"Model trained successfully")

            # Make prediction on test set
            ffm_model.setTest(ffm_test_path)
            ffm_model.setSigmoid()  # Convert output to probabilities
            ffm_model.predict(ffm_modelout_path, ffm_output_path)

            # Retrieve prediction scores from output file
            with open(ffm_output_path, encoding='utf-8') as f:
                test_predictions = [float(x.strip()) for x in f.readlines()]

            pred_df = test_merged.copy()
            pred_df["score"] = test_predictions

            gc.collect()

            # Evaluate metrics on test set for various k
            eval_results = evaluate_topk(pred_df, k_list=[5, 10, 20])

            map_k_10 = eval_results[4]

            # Keep the best model, recall, and predictions
            if map_k_10 > best_map_k_10:
                best_ffm_model = ffm_model
                best_test_predictions = test_predictions
                best_eval_result = eval_results
                best_map_k_10 = map_k_10
  
                print(f"New best model found with MAP@10: {best_map_k_10:.6f}")
                shutil.copy(ffm_modelout_path, best_ffm_modelout_path)
                shutil.copy(ffm_output_path, best_ffm_output_path)

            # Append results to tuned_results dataFrame
            tuned_results = pd.concat([
                tuned_results,
                pd.DataFrame([{
                    "task": param["task"],
                    "lr": param["lr"],
                    "lambda": param["lambda"],
                    "k": param["k"],
                    "opt": param["opt"],
                    "epoch": param["epoch"],
                    "auc": best_eval_result[0],
                    "map_5": best_eval_result[1],
                    "recall_5": best_eval_result[2],
                    "ndcg_5": best_eval_result[3],
                    "map_10": best_eval_result[4],
                    "recall_10": best_eval_result[5],
                    "ndcg_10": best_eval_result[6],
                    "map_20": best_eval_result[7],
                    "recall_20": best_eval_result[8],
                    "ndcg_20": best_eval_result[9]
                }])
            ], ignore_index=True)

        print(f"Training time: {time_train}")

        # Clean up memory
        gc.collect()

        # Increment count and clear memory periodically
        count += 1
        if count % clear_memory_freq == 0:
            print(f"\nClearing memory after {count} iterations...")
            test_merged = None
            test_merged = pd.read_pickle(test_merged_path)
            print("Reloaded test_merged from disk to free memory.")

    # Save tuning results
    tuned_results_path = os.path.join(ffm_model_path, "tuned_results.csv")
    tuned_results.to_csv(tuned_results_path, index=False)

    return best_ffm_model, best_test_predictions, best_eval_result, tuned_results



def wd_train_model(tab_preprocessor, wide_preprocessor, X_tab_train, X_wide_train, y_train, X_tab_valid, X_wide_valid, y_valid, X_tab_test, X_wide_test, y_test, test_merged):
    """
    Train WD Model with the given preprocessed training/validation set
    """
    
    # Set the wide and deep part of WD model
    wide = Wide(input_dim=wide_preprocessor.wide_dim + 1, pred_dim=1)

    deep = TabMlp(
        column_idx=tab_preprocessor.column_idx,
        continuous_cols=tab_preprocessor.continuous_cols,
        mlp_hidden_dims=[256, 128, 64, 32], 
        mlp_dropout=0.1,
        mlp_batchnorm=False
    )

    # Initialize WD Model
    wd_model = WideDeep(wide=wide, deeptabular=deep)

    # Set device, and sent the model to device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    wd_model.to(device)

    # Set the trainer
    trainer = Trainer(
        wd_model,
        objective="binary",
        optimizers=Adam(wd_model.parameters(), lr=1e-3),
        custom_loss_function=BCEWithLogitsLoss(pos_weight=torch.tensor([5.0])),
        device=device,
    )

    # Provide prelim information of WD model via print
    print("Model structure:")
    print(wd_model)

    print("Training device:", device)
    print("Train size:", X_tab_train.shape)
    print("Validation size:", X_tab_valid.shape)

    # Train WD Model
    trainer.fit(
        X_train={
            "X_wide": torch.tensor(X_wide_train).float(),
            "X_tab": torch.tensor(X_tab_train).float(),
            "target": torch.tensor(y_train).float()
        },
        X_val={
            "X_wide": torch.tensor(X_wide_valid).float(),
            "X_tab": torch.tensor(X_tab_valid).float(),
            "target": torch.tensor(y_valid).float()
        },
        n_epochs=5
    )

    # Save WD Model
    deepwide_model_path = os.path.join(wd_model_path, "deepwide_model.pt")
    torch.save(wd_model.state_dict(), deepwide_model_path)

    # Set model to eval mode
    wd_model.eval()

    # Get logit/raw prediction from the model and then apply sigmoid to get probability score
    with torch.no_grad():
        logits = wd_model({
            "wide": torch.tensor(X_wide_test).long().to(device),
            "deeptabular": torch.tensor(X_tab_test).float().to(device)
        })
        prob_preds = F.sigmoid(logits)
    test_predictions = prob_preds.squeeze().cpu().numpy()

    pred_df = test_merged.copy()
    pred_df["score"] = test_predictions

    # Evaluate metrics on test set for various k
    eval_results = evaluate_topk(pred_df, k_list=[5, 10, 20])

    return wd_model, test_predictions, eval_results



def evaluate_topk(pred_df, k_list=[5, 10, 20]):
    """
    Evaluate AUC and Recall@K, NDCG@K for multiple values of k.
    Requires pred_df with columns ["user_id_enc", "item_id_enc", "score", "label"].
    """

    # AUC across the whole dataset
    try:
        auc_score = roc_auc_score(pred_df["label"], pred_df["score"])
    except ValueError:
        auc_score = float("nan")
        print("Warning: Unable to calculate AUC due to only one class in labels.")

    print(f"AUC: {auc_score:.4f}")

    # Calculation for Recall@k and NDCG@k
    pred_df_sorted = pred_df.sort_values(by=["user_id_enc", "score"], ascending=[True, False])

    rating_true = pred_df[pred_df["label"] == 1][["user_id_enc", "item_id_enc"]].copy()
    rating_true.columns = ["userID", "itemID"]
    rating_true["rating"] = 1  # Required by MAP@K, NDCG@K

    # Save AUC to results
    results = [auc_score]

    # Loop through a list of k to calculate MAP, Recall, NDCG
    for k in k_list:
        pred_df_topk = pred_df_sorted.groupby("user_id_enc").head(k)
        rating_pred = pred_df_topk[["user_id_enc", "item_id_enc", "score"]].copy()
        rating_pred.columns = ["userID", "itemID", "prediction"]

        map_score = map_at_k(rating_true, rating_pred, col_user="userID", col_item="itemID",
                                col_prediction="prediction", k=k)
        recall = recall_at_k(rating_true, rating_pred, col_user="userID", col_item="itemID",
                                col_prediction="prediction", k=k)
        ndcg = ndcg_at_k(rating_true, rating_pred, col_user="userID", col_item="itemID",
                                col_prediction="prediction", col_rating="rating", k=k)
        
        print(f"MAP@{k}:   {map_score:.4f}")
        print(f"Recall@{k}: {recall:.4f}")
        print(f"NDCG@{k}:   {ndcg:.4f}")

        # Save MAP, Recall, NDCG to results
        results.extend([map_score, recall, ndcg])

    return results



def get_top_k_recommendations_existing_user(
    user_id_enc,
    user_enc,
    item_enc,
    interactions,
    recipes,
    predict_fn,
    model=None,
    preprocessor_tab=None,
    preprocessor_wide=None,
    scaler=None,
    k=200
):
    """
    For a specified existing user_id_enc, generate top-k recommendations using recipe metadata and a specified prediction function.
    Automatically unscales numerical recipe features using StandardScaler.
    """

    NUMERIC_FEATURES = ["minutes", "kcal", "fat", "sugar", "sodium", "protein", "carb"]

    user_id = user_enc.inverse_transform([user_id_enc])[0]

    # Get seen items for this user
    seen_items = interactions[interactions["user_id_enc"] == user_id_enc]["item_id_enc"].unique()

    recipes_small = recipes[["item_id_enc"] + NUMERIC_FEATURES + ["techniques", "final_tag_embeddings", "ingredient_embeddings", "cluster"]]
    recipes_small.columns = ["item_id_enc"] + NUMERIC_FEATURES + ["techniques", "final_tag_embeddings", "ingredient_embeddings", "cluster"]

    # Build test set for unseen items
    test_df = recipes_small[~recipes_small["item_id_enc"].isin(seen_items)].copy()
    # Add user_id_enc and dummy label to the test set
    test_df["user_id_enc"] = user_id_enc
    test_df["label"] = 0


    # Predict scores using either WD or FFM
    if predict_fn.__name__ == "wd_predict":
        test_predictions = predict_fn(model, preprocessor_tab, preprocessor_wide, test_df)
    elif predict_fn.__name__ == "ffm_predict":
        test_predictions = predict_fn(model, test_df)
    else:
        raise ValueError("Unsupported predict_fn. Use wd_predict or ffm_predict.")

    test_df["score"] = test_predictions

    # Rank and select top-k
    top_k = test_df.sort_values("score", ascending=False).drop_duplicates("item_id_enc").head(k).copy()
    top_k["recipe_id"] = item_enc.inverse_transform(top_k["item_id_enc"])

    # Unscale numerical columns
    if scaler is not None:
        scaled_numeric = top_k[NUMERIC_FEATURES].values
        unscaled_numeric = scaler.inverse_transform(scaled_numeric)
        # Replace in top_k
        for i, col in enumerate(NUMERIC_FEATURES):
            top_k[col] = unscaled_numeric[:, i]

    # Add user_id to top_k
    top_k["user_id"] = user_id

    # Merge with metadata for display (name, cluster_vector)
    top_k = top_k.merge(
        recipes[["item_id_enc", "id", "name", "cluster_vector"]],
        how="left",
        on="item_id_enc"
    )

    return top_k[[
        "user_id", "user_id_enc", "recipe_id", "item_id_enc", "name", "score",
        "minutes", "kcal", "fat", "sugar", "sodium", "protein", "carb",
        "cluster", "techniques", "final_tag_embeddings", "ingredient_embeddings", "cluster_vector"
    ]].reset_index(drop=True)



def naive_evaluation(interactions, test_merged, top_n=10):
    '''
    Naive prediction for users, by predicting the most popular recipes, based on number of interactions, and evaluate MAP, NDCG, Recall, AUC
    '''

    # Compute item popularity (frequency of interaction)
    item_counts = interactions["item_id_enc"].value_counts()
    
    # Keep only top_n most popular items
    top_n_items = item_counts.head(top_n)

    # Normalize popularity scores to [0, 1]
    max_count = top_n_items.max()
    item_scores = (top_n_items / max_count).to_dict()

    # Create prediction scores for test_merged
    pred_df = test_merged.copy()
    pred_df["score"] = pred_df["item_id_enc"].map(item_scores).fillna(0.0)

    # Evaluate metrics on test set for various k
    eval_results = evaluate_topk(pred_df, k_list=[5, 10, 20])

    test_predictions = pred_df["score"]

    return test_predictions, eval_results



def wd_predict(wd_model, tab_preprocessor, wide_preprocessor, test_merged):
    '''
    Get prediction using WD model
    '''

    # Expand feature vectors into separated columns
    test_ffm_df = expand_ffm_features(test_merged)

    # Set device and move model to device
    device = torch.device("cpu")
    wd_model.to(device)
    wd_model.eval() 

    # Tab preprocess X
    X_tab_test = tab_preprocessor.transform(test_ffm_df)
    X_wide_test = wide_preprocessor.transform(test_ffm_df)

    # Get the logit/raw prediction and convert it to probability score using Sigmoid
    with torch.no_grad():
        logits = wd_model({
            "wide": torch.tensor(X_wide_test).long().to(device),
            "deeptabular": torch.tensor(X_tab_test).float().to(device)
        })
    prob_preds = F.sigmoid(logits)

    test_predictions = prob_preds.squeeze().cpu().numpy()

    return test_predictions



def ffm_predict(ffm_model, test_merged):
    '''
    Get prediction using FFM model
    '''

    # Path to model and output
    ffm_modelout_path = os.path.join(ffm_model_path, "model.out")
    ffm_test_predict_path = os.path.join(ffm_model_path, "ffm_test_predict.txt")
    ffm_output_predict_path = os.path.join(ffm_model_path, "output_predict.txt")

    # Expand feature vector into separated columns, and write FFM file
    test_ffm_df = expand_ffm_features(test_merged)
    write_ffm_file_from_expanded_df(test_ffm_df, ffm_test_predict_path, label_col="label")

    # Make predictions
    ffm_model.setTest(ffm_test_predict_path)
    ffm_model.setSigmoid()  # Convert output to 0-1
    ffm_model.predict(ffm_modelout_path, ffm_output_predict_path)

    # Retrieve prediction from output file
    with open(ffm_output_predict_path, encoding='utf-8') as f:
        test_predictions = [float(x.strip()) for x in f.readlines()]

    return test_predictions



def get_top_k_for_new_user(
    recipes,
    liked_ingredients=None,
    disliked_ingredients=None,
    keywords=None,
    max_cooking_time=None,
    cook_or_buy="buy",
    nutrition_goals=None,
    top_n=100,
    popularity_col="popularity_score",
    scaler=None,
    interactions=None,
    item_enc=None
):
    '''
    For a new user, generate top-k recommendations using rule based scoring from cooking times, list of ingredients, keywords, and nutrition goals.
    '''
    df = recipes.copy()

    # Unscale minutes and compare it with max_cooking_time to filter list of recipes
    if scaler:
        try:
            df_unscaled = df[["minutes", "kcal", "fat", "sugar", "sodium", "protein", "carb"]].copy()
            df_unscaled[["minutes", "kcal", "fat", "sugar", "sodium", "protein", "carb"]] = scaler.inverse_transform(
                df_unscaled[["minutes", "kcal", "fat", "sugar", "sodium", "protein", "carb"]]
            )
            df["minutes_unscaled"] = df_unscaled["minutes"]
        except:
            df["minutes_unscaled"] = df["minutes"]  # fallback if not scaled
    else:
        df["minutes_unscaled"] = df["minutes"]

    # Filter by cooking time
    if max_cooking_time is not None:
        df = df[df["minutes_unscaled"] <= max_cooking_time]

    # Ingredient Matching
    def ingredient_match_score(ingredients, liked, disliked):
        score = 0
        if liked:
            score += sum(1 for ing in liked if ing.lower() in [i.lower() for i in ingredients])
        if disliked:
            score -= sum(1 for ing in disliked if ing.lower() in [i.lower() for i in ingredients])
        return score

    if liked_ingredients or disliked_ingredients:
        df["liked_score"] = df["ingredients"].apply(
            lambda x: ingredient_match_score(x, liked_ingredients, disliked_ingredients)
        )
    else:
        df["liked_score"] = 0

    # Keyword Matching (search_terms + tags)
    if keywords:
        keywords = [k.lower() for k in keywords]

        def keyword_score_fn(row):
            terms = row.get("search_terms", []) + row.get("tags", [])
            terms_joined = " ".join(terms).lower() if isinstance(terms, list) else str(terms).lower()
            return sum(k in terms_joined for k in keywords)

        df["keyword_score"] = df.apply(keyword_score_fn, axis=1)
    else:
        df["keyword_score"] = 0

    # Nutrition Goals Matching
    nutrition_columns = ["minutes", "kcal", "fat", "sugar", "sodium", "protein", "carb"]

    if nutrition_goals and scaler:
        # Create goal vector with all 7 features (include minutes)
        user_goal_array = np.array([nutrition_goals.get(col, 0) for col in nutrition_columns]).reshape(1, -1)
        user_goal_scaled = scaler.transform(user_goal_array)[0]

        def nutrition_distance(row):
            recipe_vec = np.array([row.get(col, 0) for col in nutrition_columns])
            return np.linalg.norm(recipe_vec - user_goal_scaled)

        df["nutrition_distance"] = df.apply(nutrition_distance, axis=1)
        max_dist = df["nutrition_distance"].max()
        df["nutrition_score"] = 1 - df["nutrition_distance"] / max_dist if max_dist > 0 else 0
    else:
        df["nutrition_score"] = 0

    # Compute Popularity Score
    if popularity_col not in df.columns:
        if interactions is not None and item_enc is not None:
            df["item_id_enc"] = item_enc.transform(df["id"])
            pop_df = compute_popularity_score(interactions)
            df = df.merge(pop_df, on="item_id_enc", how="left")
            df["popularity_score"] = df["popularity_score"].fillna(df["popularity_score"].mean())
        else:
            df["popularity_score"] = 0

    # Final Weighted Score
    weights = {
        "liked_score": 0.4,
        "keyword_score": 0.2,
        "nutrition_score": 0.2,
        "popularity_score": 0.2
    }

    df["final_score"] = sum(df[col] * w for col, w in weights.items())
    df = df.sort_values("final_score", ascending=False).head(top_n)

    # Unscale nutrition columns for display
    if scaler:
        scaled_cols = ["minutes", "kcal", "fat", "sugar", "sodium", "protein", "carb"]
        df[scaled_cols] = scaler.inverse_transform(df[scaled_cols])

    return df.reset_index(drop=True)



def compute_popularity_score(interactions):
    '''
    Compute Bayesian popularity score from interactions dataframe.
    '''
    recipe_stats = interactions.groupby("item_id_enc").agg(
        rating_mean=("rating", "mean"),
        rating_count=("rating", "count")
    ).reset_index()

    C = recipe_stats["rating_mean"].mean()
    m = recipe_stats["rating_count"].quantile(0.60)

    recipe_stats["popularity_score"] = (
        (recipe_stats["rating_count"] / (recipe_stats["rating_count"] + m)) * recipe_stats["rating_mean"] +
        (m / (recipe_stats["rating_count"] + m)) * C
    )

    return recipe_stats[["item_id_enc", "popularity_score"]]



def get_virtual_user_vector(recipes_df, recipe_ids):
    '''
    Create virtual vector for new user, which will be used to compare with vectors of existing users
    '''
    selected = recipes_df[recipes_df['id'].isin(recipe_ids)]
    
    return {
        "ingredient_embedding": np.stack(selected["ingredient_embeddings"]).mean(axis=0),
        "tag_embedding": np.stack(selected["final_tag_embeddings"]).mean(axis=0),
        "nutrition": selected[["kcal", "protein", "carb", "fat"]].mean(axis=0).to_numpy()
    }



def get_existing_user_vectors(interactions_df, recipes_df, threshold=4.0):
    '''
    Get vectors of existing users, which will be used to find similar existing users (compared to the new user)
    '''

    # Filter interactions by ratings >= threshold
    high_rated = interactions_df[interactions_df["rating"] >= threshold]
    merged = high_rated.merge(recipes_df, left_on="item_id_enc", right_on="id")

    # Get vectors of existing users
    user_vectors = {}
    for uid, group in merged.groupby("user_id_enc"):
        user_vectors[uid] = {
            "ingredient_embedding": np.stack(group["ingredient_embeddings"]).mean(axis=0),
            "tag_embedding": np.stack(group["final_tag_embeddings"]).mean(axis=0),
            "nutrition": group[["kcal", "protein", "carb", "fat"]].mean(axis=0).to_numpy()
        }

    return user_vectors



def find_most_similar_user(virtual_user_vector, user_vectors):
    '''
    From the list of existing users, find the most similar user (compared to the new user), using cosine similarity
    '''
    
    max_score = -1
    best_user_id = None
    for uid, vec in user_vectors.items():
        ing_sim = cosine_similarity(
            virtual_user_vector["ingredient_embedding"].reshape(1, -1),
            vec["ingredient_embedding"].reshape(1, -1)
        )[0, 0]
        tag_sim = cosine_similarity(
            virtual_user_vector["tag_embedding"].reshape(1, -1),
            vec["tag_embedding"].reshape(1, -1)
        )[0, 0]
        nutri_sim = cosine_similarity(
            virtual_user_vector["nutrition"].reshape(1, -1),
            vec["nutrition"].reshape(1, -1)
        )[0, 0]
        total_sim = (0.4 * ing_sim) + (0.4 * tag_sim) + (0.2 * nutri_sim)
        if total_sim > max_score:
            max_score = total_sim
            best_user_id = uid
    return best_user_id



def combine_recommendations(rule_based, model_based, k=50):
    '''
    Alternating rule-based and model-based recommendations, while preserving uniqueness and limiting to k items.
    '''
    seen = set()
    combined = []
    
    # Fill in the list from rule based and model based, in an alternating pattern
    for r, m in zip(rule_based, model_based):
        for item in (r, m):
            if item not in seen:
                combined.append(item)
                seen.add(item)
                if len(combined) == k:
                    return combined

    # If still not enough, fill from remaining items
    for remaining in rule_based[len(combined)//2:] + model_based[len(combined)//2:]:
        if remaining not in seen:
            combined.append(remaining)
            seen.add(remaining)
            if len(combined) == k:
                break

    return combined
