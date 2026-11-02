lgb_params = dict(objective="binary", learning_rate=0.015, num_leaves=15, min_child_samples=100,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.4, reg_lambda=5.0,
                  verbose=-1, random_state=7)

cat_params = dict(learning_rate=0.012, depth=7, l2_leaf_reg=10, rsm=0.5, eval_metric="AUC", verbose=0,
                  random_seed=8, od_type="Iter", od_wait=300)

# a bigger down payment should never make the result worse
always_up = ["ltv", "ltv_vs_model", "ltv_vs_area", "loan_amount", "loan_vs_model", "loan_to_income"]
always_down = ["down_payment"]

# fields never used for the decision, whatever the search says
policy_drop = ["Gender"]

# what the applicant is actually asked, and which file columns each answer fills
questions = {
    "vehicle": ["Model_Description", "Make_Code", "Model_Variant", "Product_Code"],
    "on_road_price": ["LTV"],
    # without a down payment there is no loan amount and no ltv either
    "down_payment": ["Loan_Amount", "LTV"],
    "pincode": ["Pincode", "Final_Tier"],
    "age": ["Age"],
    "employment": ["Employment_Type"],
    "income": ["Net_salary"],
    "active_loans": ["PAST_LOANS_ACTIVE"],
    "education": ["Qualifications"],
    "residence": ["Resident_Type"],
}
