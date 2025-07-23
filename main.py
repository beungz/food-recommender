import os

import numpy as np
import pandas as pd

import joblib

import torch
from pytorch_widedeep.models import Wide, TabMlp, WideDeep

import streamlit as st

from scripts.model import wd_predict, get_top_k_recommendations_existing_user, get_top_k_for_new_user, get_virtual_user_vector, get_existing_user_vectors, find_most_similar_user, combine_recommendations

processed_data_path = os.path.join("data", "processed")
outputs_data_path = os.path.join("data", "outputs")
wd_model_path = os.path.join("models", "deep_learning")



@st.cache_resource
def load_wd_model():
    '''
    Load the WD model trained for food recommender system
    '''
    # Load user and item encoder
    user_enc_path = os.path.join(outputs_data_path, "user_encoder.pkl")
    item_enc_path = os.path.join(outputs_data_path, "item_encoder.pkl")
    user_enc = joblib.load(user_enc_path)
    item_enc = joblib.load(item_enc_path)

    # Load scaler
    scaler_path = os.path.join(outputs_data_path, "scaler.pkl")
    scaler = joblib.load(scaler_path)

    # Load recipes and interactions dataframe
    recipes_path = os.path.join(processed_data_path, "recipes.pkl")
    interactions_encoded_path = os.path.join(processed_data_path, "interactions_encoded.pkl")
    recipes = pd.read_pickle(recipes_path)
    interactions = pd.read_pickle(interactions_encoded_path)

    # Load list of ingredients and tags, to be used as options in multiselect
    all_ingredients_path = os.path.join(processed_data_path, "all_ingredients.pkl")
    all_ingredients = joblib.load(all_ingredients_path)
    frequent_ingredients_path = os.path.join(processed_data_path, "frequent_ingredients.pkl")
    frequent_ingredients = joblib.load(frequent_ingredients_path)
    all_final_tags_path = os.path.join(processed_data_path, "all_final_tags.pkl")
    all_final_tags = joblib.load(all_final_tags_path)
    all_tags_search_terms_path = os.path.join(processed_data_path, "all_tags_search_terms.pkl")
    all_tags_search_terms = joblib.load(all_tags_search_terms_path)

    # Load preprocessors
    tab_preprocessor_path = os.path.join(wd_model_path, "tab_preprocessor.pkl")
    wide_preprocessor_path = os.path.join(wd_model_path, "wide_preprocessor.pkl")
    tab_preprocessor = joblib.load(tab_preprocessor_path)
    wide_preprocessor = joblib.load(wide_preprocessor_path)

    # Set the wide and deep part of WD model
    wide = Wide(input_dim=wide_preprocessor.wide_dim + 1, pred_dim=1)

    deep = TabMlp(
        column_idx=tab_preprocessor.column_idx,
        continuous_cols=tab_preprocessor.continuous_cols,
        mlp_hidden_dims=[256, 128, 64, 32], 
        mlp_dropout=0.1,
        mlp_batchnorm=False
    )

    # Initialize WD Model and load its state dict
    wd_model = WideDeep(wide=wide, deeptabular=deep)

    deepwide_model_path = os.path.join(wd_model_path, "deepwide_model.pt")
    wd_model.load_state_dict(torch.load(deepwide_model_path))

    device = torch.device("cpu")
    wd_model.to(device)

    return user_enc, item_enc, scaler, recipes, interactions, all_ingredients, frequent_ingredients, all_final_tags, all_tags_search_terms, tab_preprocessor, wide_preprocessor, wd_model



def run():
    '''
    Run the Streamlit app for food recommendation.
    '''
    # Load WD model
    user_enc, item_enc, scaler, recipes, interactions, all_ingredients, frequent_ingredients, all_final_tags, all_tags_search_terms, tab_preprocessor, wide_preprocessor, wd_model = load_wd_model()

    # Streamlit UI
    st.title("Food Recommender System")
    st.markdown("Personalized food recipe suggestions for new users based on preferences & similarity.")

    # User inputs
    cook_or_buy = st.radio("Do you want to cook or buy food?", ["cook", "buy"])
    max_cooking_time = st.slider("Maximum cooking time (minutes):", 5, 120, 30) if cook_or_buy == "cook" else None

    liked_ingredients = st.multiselect(
        "Select or type favorite ingredients:",
        options=all_ingredients,
        help="Start typing to search or choose from the list."
    )

    disliked_ingredients = st.multiselect(
        "Select or type disliked/allergy ingredients:",
        options=all_ingredients,
        help="Start typing to search or choose from the list."
    )

    keywords = st.multiselect(
        "Keywords:",
        options=all_tags_search_terms,
        help="Start typing to search or choose from the list."
    )

    with st.expander("Set your meal nutrition goals"):
        kcal = st.number_input("Max kcal per meal", 0, 10000, 500)
        protein = st.number_input("Max protein (g)", 0, 300, 20)
        carb = st.number_input("Max carbs (g)", 0, 500, 60)
        fat = st.number_input("Max fat (g)", 0, 200, 20)
    
    # Get Recommendations button
    if st.button("Get Recommendations"):
        # Process all user inputs
        nutrition_goals = dict(kcal=kcal, protein=protein, carb=carb, fat=fat)
        liked_ingredients = [x.strip() for x in liked_ingredients if x.strip()]
        disliked_ingredients = [x.strip() for x in disliked_ingredients if x.strip()]
        keywords_list = [x.strip() for x in keywords if x.strip()]

        # Rule-based recommender - recommend food recipes based on inputs that new user gave on Streamlit interface, using ruled based scoring
        rule_based_recs_df = get_top_k_for_new_user(
            recipes,
            liked_ingredients=liked_ingredients,
            disliked_ingredients=disliked_ingredients,
            keywords=keywords_list,
            max_cooking_time=max_cooking_time,
            cook_or_buy=cook_or_buy,
            nutrition_goals=nutrition_goals,
            top_n=50,
            scaler=scaler,
            interactions=interactions,
            item_enc=item_enc
        )
        rule_based_ids = rule_based_recs_df["id"].tolist()

        # Virtual user vector - use virtual user vector to compare with vectors of other existing users with cosine similarity, and return the user (id) with the most similar in preference
        virtual_user_vec = get_virtual_user_vector(recipes, rule_based_ids[:10])
        user_vectors = get_existing_user_vectors(interactions, recipes)
        most_similar_user = find_most_similar_user(virtual_user_vec, user_vectors)

        # Wide & Deep model recommender - recommend food recipes based on similar existing user
        model_recs_ids_df = get_top_k_recommendations_existing_user(
            most_similar_user,
            user_enc,
            item_enc,
            interactions,
            recipes,
            predict_fn = wd_predict,
            model = wd_model,
            preprocessor_tab = tab_preprocessor,
            preprocessor_wide = wide_preprocessor,
            scaler=scaler,
            k=50
        )
        model_recs_ids = model_recs_ids_df["recipe_id"].tolist()


        # Combine results from rule based and WD model based method
        combined = combine_recommendations(rule_based_ids, model_recs_ids, k=50)
        final_df = recipes[recipes["id"].isin(combined)].copy()
        final_df = final_df.set_index("id").loc[combined].reset_index() 

        # Unscale the nutrition and minutes data
        nutrition_cols = ["minutes", "kcal", "fat", "sugar", "sodium", "protein", "carb"]
        final_df[nutrition_cols] = scaler.inverse_transform(final_df[nutrition_cols])
        final_df[nutrition_cols] = final_df[nutrition_cols].round(2)

        # Display resulting list of recommended food recipes
        st.markdown("### Top Food Recommendations")

        # Select and rename columns for display
        display_df = final_df[["name", "minutes", "kcal", "protein", "carb", "fat", "sodium", "ingredients"]].copy()
        display_df.columns = ["Name", "Minutes", "kcal", "Protein (g)", "Carbs (g)", "Fat (g)", "Sodium (mg)", "Ingredients"]

        # Convert ingredient lists to comma-separated strings
        display_df["Ingredients"] = display_df["Ingredients"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))

        # Set index to start from 1
        display_df.index = np.arange(1, len(display_df) + 1)

        # Define formatting rules
        format_dict = {
            "Minutes": "{:.0f}",
            "kcal": "{:.1f}",
            "Protein (g)": "{:.1f}",
            "Carbs (g)": "{:.1f}",
            "Fat (g)": "{:.1f}",
            "Sodium (mg)": "{:.1f}",
            # No formatting for Ingredients or Name
        }

        # Style the DataFrame: format + row color + alignment
        styled_df = (
            display_df.style
            .format(format_dict)
            .set_properties(**{'text-align': 'left'})
            .apply(lambda x: ['background-color: #f9f9f9' if i % 2 else '' for i in range(len(x))], axis=0)
        )

        # Show styled dataframe in Streamlit
        st.dataframe(styled_df, use_container_width=True)


if __name__ == "__main__":
    run()