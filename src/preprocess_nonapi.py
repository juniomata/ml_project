import pandas as pd
import numpy as np
import util as util
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def clean_headers(val):
    if isinstance(val, str):
        # remove special chars (but skip emtpy spaces and all)
        val = "".join(char for char in val if char.isalnum() or char in (" ", "_"))
        # convert to snake case
        val = val.strip().lower().replace(" ", "_")
        return val
    else:
        return val


def load_dataset(config_data: dict) -> pd.DataFrame:
    # Load every set of data
    x_train = util.pickle_load(config_data["train_set_path"][0])
    y_train = util.pickle_load(config_data["train_set_path"][1])
    x_valid = util.pickle_load(config_data["valid_set_path"][0])
    y_valid = util.pickle_load(config_data["valid_set_path"][1])
    x_test = util.pickle_load(config_data["test_set_path"][0])
    y_test = util.pickle_load(config_data["test_set_path"][1])
    # Concatenate x and y each set
    train_set = pd.concat([x_train, y_train], axis=1)
    valid_set = pd.concat([x_valid, y_valid], axis=1)
    test_set = pd.concat([x_test, y_test], axis=1)
    # Return 3 set of data
    return train_set, valid_set, test_set


def nan_detector(set_data: pd.DataFrame) -> pd.DataFrame:
    # Create copy of set data
    set_data = set_data.copy()
    # Replace -999 with NaN
    set_data.replace(-999, np.nan, inplace=True)
    # Return replaced set data
    return set_data


def remove_outliers(set_data):
    num_col = train_set.select_dtypes(
        include=['int64', 'float64']).columns.to_list()
    set_data = train_set.copy()
    list_of_set_data = list()

    for col_name in num_col:
        q1 = set_data[col_name].quantile(0.25)
        q3 = set_data[col_name].quantile(0.75)
        iqr = q3 - q1
        set_data_cleaned = set_data[~((set_data[col_name] < (q1 - 1.5 * iqr)) | (set_data[col_name] > (q3 + 1.5 * iqr)))].copy()
        list_of_set_data.append(set_data_cleaned.copy())
    
    set_data_cleaned = pd.concat(list_of_set_data)
    count_duplicated_index = set_data_cleaned.index.value_counts()
    used_index_data = count_duplicated_index[count_duplicated_index == (set_data.shape[1]-1)].index
    set_data_cleaned = set_data_cleaned.loc[used_index_data].drop_duplicates()

    return set_data_cleaned


def ohe_fit(data_tobe_fitted: dict, ohe_path: str) -> OneHotEncoder:
    # Create ohe object
    ohe_cat = OneHotEncoder(sparse_output=False)
    # Fit ohe
    ohe_cat.fit(np.array(data_tobe_fitted).reshape(-1, 1))
    # Save ohe object
    util.pickle_dump(ohe_cat, ohe_path)
    # Return trained ohe
    return ohe_cat


def ohe_transform(set_data: pd.DataFrame, transformed_column: str, ohe_path: str) -> pd.DataFrame:
    # Create copy of set data
    set_data = set_data.copy()
    # Load ohe customer_statuscal var
    ohe_cat = util.pickle_load(ohe_path)
    # Transform variable categorical var of set data, resulting array
    cat_features = ohe_cat.transform(np.array(set_data[transformed_column].to_list()).reshape(-1, 1))
    # Convert to dataframe
    cat_features = pd.DataFrame(cat_features.tolist(), columns=list(ohe_cat.categories_[0]))
    # Set index by original set data index
    cat_features.set_index(set_data.index, inplace=True)
    # Rename columns
    cat_features = cat_features.add_prefix(transformed_column + "_")
    cat_features = cat_features.rename(columns=clean_headers)
    # Concatenate new features with original set data
    set_data = pd.concat([cat_features, set_data], axis=1)
    # Drop transformed categorical column
    set_data.drop(columns=transformed_column, inplace=True)
    # Convert columns type to string
    new_col = [str(col_name) for col_name in set_data.columns.to_list()]
    set_data.columns = new_col
    # Return new feature engineered set data
    return set_data


def rus_fit_resample(set_data: pd.DataFrame) -> pd.DataFrame:
    # Create copy of set data
    set_data = set_data.copy()
    # Create sampling object
    rus = RandomUnderSampler(random_state=26)
    # Balancing set data
    x_rus, y_rus = rus.fit_resample(set_data.drop(
        "customer_status", axis=1), set_data.customer_status)
    # Concatenate balanced data
    set_data_rus = pd.concat([x_rus, y_rus], axis=1)
    # Return balanced data
    return set_data_rus


def sm_fit_resample(set_data: pd.DataFrame) -> pd.DataFrame:
    # Create copy of set data
    set_data = set_data.copy()
    # Create sampling object
    sm = SMOTE(random_state=112)
    # Balancing set data
    x_sm, y_sm = sm.fit_resample(set_data.drop(
        "customer_status", axis=1), set_data.customer_status)
    # Concatenate balanced data
    set_data_sm = pd.concat([x_sm, y_sm], axis=1)
    # Return balanced data
    return set_data_sm


def le_fit(data_tobe_fitted: dict, le_path: str) -> LabelEncoder:
    # Create le object
    le_encoder = LabelEncoder()
    # Fit le
    le_encoder.fit(data_tobe_fitted)
    # Save le object
    util.pickle_dump(le_encoder, le_path)
    # Return trained le
    return le_encoder


def le_transform(label_data: pd.Series, config_data: dict) -> pd.Series:
    # Create copy of label_data
    label_data = label_data.copy()
    # Load le encoder
    le_encoder = util.pickle_load(config_data["le_encoder_path"])
    # If customer_statuses both label data and trained le matched
    if len(set(label_data.unique()) - set(le_encoder.classes_) | set(le_encoder.classes_) - set(label_data.unique())) == 0:
        # Transform label data
        label_data = le_encoder.transform(label_data)
    else:
        raise RuntimeError("Check category in label data and label encoder.")
    # Return transformed label data
    return label_data

if __name__ == "__main__":
    # 1. Load configuration file
    config_data = util.load_config()

    # 2. Load dataset
    train_set, valid_set, test_set = load_dataset(config_data)

    # 3. Converting -999 to NaN
    train_set = nan_detector(train_set)
    valid_set = nan_detector(valid_set)
    test_set = nan_detector(test_set)

    # 4. Handling `avg_monthly_long_distance_charges` and `avg_monthly_gb_download` 
    # 4.1. Handle null train set
    impute_avg_monthly_long_distance_charges = int(train_set.avg_monthly_long_distance_charges.mean())
    impute_avg_monthly_gb_download = int(train_set.avg_monthly_gb_download.median())
    impute_values = {"avg_monthly_long_distance_charges": impute_avg_monthly_long_distance_charges, "avg_monthly_gb_download": impute_avg_monthly_gb_download}
    train_set.fillna(value=impute_values, inplace=True)
    # 4.2. Handle null validation set
    valid_set.fillna(value=impute_values, inplace=True)
    # 4.3. Handle null test set
    test_set.fillna(value=impute_values, inplace=True)

    # 5. Drop features
    train_set = train_set.drop(columns=["number_of_dependents", "total_refunds", "total_extra_data_charges"])
    valid_set = valid_set.drop(columns=["number_of_dependents", "total_refunds", "total_extra_data_charges"])
    test_set = test_set.drop(columns=["number_of_dependents", "total_refunds", "total_extra_data_charges"])

    # # 6. Remove outliers
    # train_set = remove_outliers(train_set)
    # valid_set = remove_outliers(valid_set)
    # test_set = remove_outliers(test_set)

    # 7. Fit ohe with predefined categorical data
    list_cat_var_ohe = ["ohe_" + i for i in config_data["object_columns"]]
    list_cat_var_range = ["range_" + i for i in config_data["object_columns"]]
    list_cat_var_path = ["ohe_" + i  + "_path" for i in config_data["object_columns"]]

    for idx in range(len(list_cat_var_ohe)):
        list_cat_var_ohe[idx] = ohe_fit(config_data[list_cat_var_range[idx]], config_data[list_cat_var_path[idx]])


    # 8. Transform stasiun on train, valid, and test set
    for i in range(len(list_cat_var_ohe)):
        train_set = ohe_transform(train_set, 
                                config_data["object_columns"][i], 
                                config_data[list_cat_var_path[i]]
                                )

    for i in range(len(list_cat_var_ohe)):
        valid_set = ohe_transform(valid_set,
                                config_data["object_columns"][i],
                                config_data[list_cat_var_path[i]]
                                )

    for i in range(len(list_cat_var_ohe)):
        test_set = ohe_transform(test_set,
                                config_data["object_columns"][i],
                                config_data[list_cat_var_path[i]]
                                )

    # 9. Undersampling dataset
    train_set_rus = rus_fit_resample(train_set)

    # 10. SMOTE dataset
    train_set_sm = sm_fit_resample(train_set)

    # 11. Fit label encoder
    le_encoder = le_fit(config_data["label_categories"], 
                        config_data["le_encoder_path"])

    # 12. Label encoding undersampling set
    train_set_rus.customer_status = le_transform(train_set_rus.customer_status, config_data)

    # 13. Label encoding smote set
    train_set_sm.customer_status = le_transform(train_set_sm.customer_status, config_data)

    # 14. Label encoding validation set
    valid_set.customer_status = le_transform(valid_set.customer_status, config_data)


    # 15. Label encoding test set
    test_set.customer_status = le_transform(test_set.customer_status, config_data)

    # 16. Dumping dataset
    x_train = {
        "Undersampling": train_set_rus.drop(columns="customer_status"),
        "SMOTE": train_set_sm.drop(columns="customer_status")
    }

    y_train = {
        "Undersampling": train_set_rus.customer_status,
        "SMOTE": train_set_sm.customer_status
    }

    util.pickle_dump(x_train, "data/processed/x_train_feng.pkl")
    util.pickle_dump(y_train, "data/processed/y_train_feng.pkl")

    util.pickle_dump(valid_set.drop(columns="customer_status"), "data/processed/x_valid_feng.pkl")
    util.pickle_dump(valid_set.customer_status, "data/processed/y_valid_feng.pkl")

    util.pickle_dump(test_set.drop(columns="customer_status"), "data/processed/x_test_feng.pkl")
    util.pickle_dump(test_set.customer_status, "data/processed/y_test_feng.pkl")