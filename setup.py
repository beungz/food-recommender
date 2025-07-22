from scripts.make_dataset import prepare_data, stratified_user_split, merge_recipes_interactions, prepare_ffm_data, prepare_wd_data
from scripts.model import ffm_train_model, ffm_train_model_hyperparam_tuning, wd_train_model, naive_evaluation



def main():
    '''Main function to run the entire pipeline for food recommender system.'''


    # A. Get dataset
    print("\nA. Get dataset")

    # Step A1. Prepare data and generate features
    print("\nStep A1. Prepare data and generate features")
    recipes, recipes_unscaled, interactions, user_enc, item_enc, scaler = prepare_data()

    # Step A2. Stratified user split into train, validation, and test set
    print("\nStep A2. Stratified user split into train, validation, and test set")
    train_df, valid_df, test_df = stratified_user_split(interactions, recipes, valid_frac=0.15, test_frac=0.15)

    train_merged, valid_merged, test_merged = merge_recipes_interactions(recipes, train_df, valid_df, test_df)

    # Step A3. FFM: preprocess train data
    print("\nStep A3. FFM: preprocess train data")
    train_ffm_df, valid_ffm_df, test_ffm_df = prepare_ffm_data(train_merged, valid_merged, test_merged)

    # Step A4. WD: preprocess train data
    print("\nStep A4. WD: preprocess train data")
    tab_preprocessor, wide_preprocessor, X_tab_train, X_wide_train, y_train, X_tab_valid, X_wide_valid, y_valid, X_tab_test, X_wide_test, y_test = prepare_wd_data(train_ffm_df, valid_ffm_df, test_ffm_df)


    # B. Train FFM and WD models
    print("\nB. Train FFM and WD models")

    # Step B1: Train FFM model
    print("\nStep B1: Train FFM model")
    ffm_model, test_predictions, eval_results = ffm_train_model(test_merged)

    # Step B1: Train FFM model
    print("\nStep B1: FFM model hyperparameter finetuning")

    lr_values = [0.01, 0.05, 0.1]
    lambda_values = [0.0001, 0.0005, 0.001]
    k_values = [8] # 8, 16, 32
    opt_values = ["adagrad"] # "adagrad", "sgd", "ftrl"
    epoch_values = [15] # 15, 30

    param_dict = {"task": ["binary"], "metric": ["auc"], "lr": lr_values, "lambda": lambda_values, "k": k_values, "opt": opt_values, "epoch": epoch_values}

    best_ffm_model, best_test_predictions, best_eval_result, tuned_results = ffm_train_model_hyperparam_tuning(test_merged, param_dict)


    # Step B2: Train WD model
    print("\nStep B2: Train WD model")
    wd_model, test_predictions, eval_results = wd_train_model(
        tab_preprocessor, wide_preprocessor, 
        X_tab_train, X_wide_train, y_train, X_tab_valid, X_wide_valid, y_valid, X_tab_test, X_wide_test, y_test, 
        test_merged
    )

    # Step B3: Evaluate naive model
    print("\nStep B3: Evaluate naive model")
    test_predictions, eval_results = naive_evaluation(interactions, test_merged)


if __name__ == "__main__":
    main()