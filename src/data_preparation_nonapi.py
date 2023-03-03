import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import copy
import util as util

def clean_headers(val):
    if isinstance(val, str):
        # Remove special chars (but skip emtpy spaces and all)
        val = "".join(char for char in val if char.isalnum()
                      or char in (" ", "_"))
        # Convert to snake case
        val = val.strip().lower().replace(" ", "_")
        return val
    else:
        return val


def read_raw_data(config: dict) -> pd.DataFrame:
    # Create variable to store raw dataset
    raw_dataset = pd.DataFrame()
    # Raw dataset dir
    raw_dataset_dir = config["raw_dataset_dir"]
    # Look and load add CSV files
    raw_dataset = pd.read_csv(raw_dataset_dir)
    # Rename columns
    raw_dataset = raw_dataset.rename(columns=clean_headers)
    # Return raw dataset
    return raw_dataset


def check_data(input_data, params, api=False):
    input_data = copy.deepcopy(input_data)
    params = copy.deepcopy(params)

    if not api:
        # Check data types
        assert input_data.select_dtypes("object").columns.to_list() == \
            params["object_columns_with_target"], "an error occurs in object column(s)."
        assert input_data.select_dtypes("int").columns.to_list() == \
            params["int32_columns"], "an error occurs in int32 column(s)."
        assert input_data.select_dtypes("float").columns.to_list() == \
            params["float_columns"], "an error occurs in float column(s)."

    else:
        # In case checking data from api
        # Include only valid object variable for modeling
        object_columns = params["object_columns"]
        # del object_columns[1:]

        # Include only valid integer variable for modeling
        int_columns = params["int32_columns"]
        # del int_columns[-1]

        # Include only valid integer variable for modeling
        float_columns = params["float_columns"]

        # Check data types
        list_of_objects = input_data.select_dtypes("object").columns.to_list()
        assert input_data.select_dtypes("object").columns.to_list() == \
            object_columns, f"an error occurs in object column(s)! {list_of_objects} vs {object_columns}."
        assert input_data.select_dtypes("int").columns.to_list() == \
            int_columns, "an error occurs in int32 column(s)."
        assert input_data.select_dtypes("float").columns.to_list() == \
            float_columns, "an error occurs in float column(s)."

    assert input_data.age.between(params["range_age"][0], params["range_age"][1]).sum() == len(input_data), "an error occurs in age range." 
    assert input_data.number_of_dependents.between(params["range_number_of_dependents"][0], params["range_number_of_dependents"][1]).sum() == len(input_data), "an error occurs in number_of_dependents range." 
    assert input_data.number_of_referrals.between(params["range_number_of_referrals"][0], params["range_number_of_referrals"][1]).sum() == len(input_data), "an error occurs in number_of_referrals range." 
    assert input_data.tenure_in_months.between(params["range_tenure_in_months"][0], params["range_tenure_in_months"][1]).sum() == len(input_data), "an error occurs in tenure_in_months range." 
    assert input_data.total_extra_data_charges.between(params["range_total_extra_data_charges"][0], params["range_total_extra_data_charges"][1]).sum() == len(input_data), "an error occurs in total_extra_data_charges range." 
    
    assert set(input_data.gender).issubset(set(params["range_gender"])), "an error occurs in gender range." 
    assert set(input_data.married).issubset(set(params["range_married"])), "an error occurs in married range." 
    assert set(input_data.offer).issubset(set(params["range_offer"])), "an error occurs in offer range." 
    assert set(input_data.phone_service).issubset(set(params["range_phone_service"])), "an error occurs in phone_service range." 
    assert set(input_data.multiple_lines).issubset(set(params["range_multiple_lines"])), "an error occurs in multiple_lines range." 
    assert set(input_data.internet_service).issubset(set(params["range_internet_service"])), "an error occurs in internet_service range." 
    assert set(input_data.internet_type).issubset(set(params["range_internet_type"])), "an error occurs in internet_type range." 
    assert set(input_data.online_security).issubset(set(params["range_online_security"])), "an error occurs in online_security range." 
    assert set(input_data.online_backup).issubset(set(params["range_online_backup"])), "an error occurs in online_backup range." 
    assert set(input_data.device_protection_plan).issubset(set(params["range_device_protection_plan"])), "an error occurs in device_protection_plan range." 
    assert set(input_data.premium_tech_support).issubset(set(params["range_premium_tech_support"])), "an error occurs in premium_tech_support range." 
    assert set(input_data.streaming_tv).issubset(set(params["range_streaming_tv"])), "an error occurs in streaming_tv range." 
    assert set(input_data.streaming_movies).issubset(set(params["range_streaming_movies"])), "an error occurs in streaming_movies range." 
    assert set(input_data.streaming_music).issubset(set(params["range_streaming_music"])), "an error occurs in streaming_music range." 
    assert set(input_data.unlimited_data).issubset(set(params["range_unlimited_data"])), "an error occurs in unlimited_data range." 
    assert set(input_data.contract).issubset(set(params["range_contract"])), "an error occurs in contract range." 
    assert set(input_data.paperless_billing).issubset(set(params["range_paperless_billing"])), "an error occurs in paperless_billing range." 
    assert set(input_data.payment_method).issubset(set(params["range_payment_method"])), "an error occurs in payment_method range." 
    # assert set(input_data.customer_status).issubset(set(params["range_customer_status"])), "an error occurs in customer_status range." 


if __name__ == "__main__":
    # 1. Load configuration file
    config_data = util.load_config()

    # 2. Read all raw dataset
    raw_dataset = read_raw_data(config_data)

    # 3. Reset index
    raw_dataset.reset_index(
        inplace=True,
        drop=True
    )

    # 4. Save raw dataset
    util.pickle_dump(
        raw_dataset,
        config_data["raw_dataset_path"]
    )

    # 5. Remove unecessary features
    list_columns_to_drop = ['churn_reason', 'churn_category', 'customer_id', 'city', 'zip_code', 'latitude', 'longitude']
    raw_dataset = raw_dataset.drop(columns=list_columns_to_drop)

    # 6. Exclude Customer Status = 'Joined'
    # raw_dataset = raw_dataset[raw_dataset['customer_status'] != 'Joined'].reset_index(drop=True)

    # 7. Change values ("Yes" to "Y", "No" to "N")
    raw_dataset.replace("Yes", "Y", inplace=True)
    raw_dataset.replace("No", "N", inplace=True)

    # 8. handling missing value for categorical variables (for splitting needs)
    categorical_features = raw_dataset.select_dtypes('object').columns
    raw_dataset[categorical_features] = raw_dataset.select_dtypes('object').fillna('Unknown')

    # 8. Check data definition
    check_data(raw_dataset, config_data)

    # 10. Handing missing value for numerical variables (for splitting needs)
    raw_dataset.avg_monthly_long_distance_charges.fillna(-999, inplace=True)
    raw_dataset.avg_monthly_gb_download.fillna(-999, inplace=True)

    # 11. Splitting input output
    x = raw_dataset[config_data["predictors"]].copy()
    y = raw_dataset.customer_status.copy()

    # 12. Splitting train test
    x_train, x_test, \
        y_train, y_test = train_test_split(
            x, y,
            test_size=0.3,
            random_state=42,
            stratify=y
        )

    # 13. Splitting test valid
    x_valid, x_test, \
        y_valid, y_test = train_test_split(
            x_test, y_test,
            test_size=0.5,
            random_state=42,
            stratify=y_test
        )

    # 14. Save train, valid and test set
    util.pickle_dump(x_train, config_data["train_set_path"][0])
    util.pickle_dump(y_train, config_data["train_set_path"][1])

    util.pickle_dump(x_valid, config_data["valid_set_path"][0])
    util.pickle_dump(y_valid, config_data["valid_set_path"][1])

    util.pickle_dump(x_test, config_data["test_set_path"][0])
    util.pickle_dump(y_test, config_data["test_set_path"][1])