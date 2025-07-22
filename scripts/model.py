import os

import numpy as np
import pandas as pd

import shutil 
import gc
import itertools

import torch
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F

from pytorch_widedeep.models import Wide, TabMlp, WideDeep
from pytorch_widedeep.training import Trainer

import xlearn as xl

from recommenders.utils.timer import Timer
from recommenders.evaluation.python_evaluation import precision_at_k, recall_at_k, map_at_k, ndcg_at_k
from recommenders.tuning.parameter_sweep import generate_param_grid

from sklearn.metrics import roc_auc_score

from scripts.build_features import expand_ffm_features
from scripts.make_dataset import write_ffm_file_from_expanded_df

ffm_model_path = os.path.join("models", "classical_machine_learning")
wd_model_path = os.path.join("models", "deep_learning")
outputs_data_path = os.path.join("data", "outputs")
processed_data_path = os.path.join("data", "processed")



def ffm_train_model(test_merged):
    """
    Trains an FFM model and evaluates recall at various k values.
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
    
    Parameters:
        test_merged : The merged test dataset containing features and labels.
        param_dict : Dictionary of hyperparameters to sweep over for tuning.
    
    Returns:
        best_ffm_model: The best trained FFM model based on Recall@200.
        best_test_predictions : Predictions from the best model.
        best_recall_k_x : Best Recall@x score.
        tuned_results (pd.DataFrame): DataFrame containing results for all hyperparameter combinations.
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



def naive_evaluation(interactions, test_merged):
    """
    Naive prediction for users, by predicting the most popular recipes, based on number of interactions, and evaluate recall@k
    """

    # Compute item popularity (frequency of interaction)
    item_counts = interactions["item_id_enc"].value_counts()
    
    # Normalize popularity scores to [0, 1]
    max_count = item_counts.max()
    item_scores = (item_counts / max_count).to_dict()

    # Create prediction scores for test_merged
    pred_df = test_merged.copy()
    pred_df["score"] = pred_df["item_id_enc"].map(item_scores).fillna(0.0)

    # Evaluate metrics on test set for various k
    eval_results = evaluate_topk(pred_df, k_list=[5, 10, 20])

    test_predictions = pred_df["score"]

    return test_predictions, eval_results



def wd_predict(wd_model, tab_preprocessor, wide_preprocessor, test_merged):
    """
    Get prediction using WD model
    """

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
    """
    Get prediction using FFM model
    """

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



# TO BE REVISED
def get_top_k_recommendations_new_user(user_id_enc, interactions, test_merged, test_predictions, recipes_unscaled, user_enc, item_enc, k=200):
    """
    Return top-K recommended recipes for a new user.
    """
    user_id = user_enc.inverse_transform([user_id_enc])[0]

    seen_items = interactions[interactions["user_id_enc"] == user_id_enc]["item_id_enc"].unique()

    # Add scores to test set
    test_merged_df = test_merged.copy()
    test_merged_df["score"] = test_predictions

    # Get user-specific candidates
    user_candidates = test_merged_df[test_merged_df["user_id_enc"] == user_id_enc]
    user_candidates = user_candidates[~user_candidates["item_id_enc"].isin(seen_items)]

    # If not enough candidates, sample from other scored items (same user)
    if len(user_candidates) < k:
        needed = k - len(user_candidates)
        print(f"Warning: only {len(user_candidates)} user-specific candidates available, padding with top from other users")
        extra_candidates = test_merged_df[
            ~test_merged_df["item_id_enc"].isin(seen_items)
        ].sort_values("score", ascending=False).drop_duplicates("item_id_enc").head(needed)

        # Ensure same user_id_enc for consistency
        extra_candidates["user_id_enc"] = user_id_enc
        user_candidates = pd.concat([user_candidates, extra_candidates], ignore_index=True)

    # Get top-k
    top_k = user_candidates.sort_values("score", ascending=False).drop_duplicates("item_id_enc").head(k).copy()
    top_k["recipe_id"] = item_enc.inverse_transform(top_k["item_id_enc"])
    top_k["user_id"] = user_id

    top_k = top_k.drop(columns = ['minutes', 'kcal', 'fat', 'sugar', 'sodium', 'protein', 'carb'])

    # Merge with recipe metadata
    top_k = top_k.merge(
        recipes_unscaled[['id', 'minutes', 'kcal', 'fat', 'sugar', 'sodium', 'protein', 'carb', 'name','cluster_vector']],
        how="left",
        left_on="recipe_id",
        right_on="id"
    ).drop(columns=["id"])

    return top_k[[
        "user_id", "user_id_enc", "recipe_id", "item_id_enc", "name", "score", "label",
        "minutes", "kcal", "fat", "sugar", "sodium", "protein", "carb", "cluster", "techniques", "final_tag_embeddings", "ingredient_embeddings", "cluster_vector"
    ]].reset_index(drop=True)