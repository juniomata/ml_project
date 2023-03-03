import streamlit as st
import requests
from PIL import Image

# Load and set images in the first place
header_images = Image.open('assets/header.jpg')
st.image(header_images)

# Add some information about the service
st.title("Blockchurn: Churn Predicition")
st.subheader("Just enter variables below then click Predict button")

# Create form of input
with st.form(key = "air_data_form"):

    # Create box for number input
    age = st.number_input(
        label = "Enter Age Value:",
        min_value = 19,
        max_value = 80,
        help = "Value range from 19 to 80"
    )

    number_of_referrals = st.number_input(
        label = "Enter Number of Referrals Value:",
        min_value = 0,
        max_value = 11,
        help = "Value range from 0 to 11"
    )

    tenure_in_months = st.number_input(
        label = "Enter Tenure in Months Value:",
        min_value = 1,
        max_value = 72,
        help = "Value range from 1 to 72"
    )

    avg_monthly_long_distance_charges = st.number_input(
        label = "Enter Avg Monthly Long Distance Charges Value:",
        min_value = 1.01,
        max_value = 49.99,
        help = "Value range from 1.01 to 49.99"
    )

    avg_monthly_gb_download = st.number_input(
        label = "Enter Avg Monthly GB Download Value:",
        min_value = 2.0,
        max_value = 85.0,
        help = "Value range from 2.0 to 85.0"
    )
    
    monthly_charge = st.number_input(
        label = "Enter Monthly Charge Value:",
        min_value = -10.0,
        max_value = 118.75,
        help = "Value range from -10.0 to 118.75"
    )

    total_charges = st.number_input(
        label = "Enter Total Charges Value:",
        min_value = 18.85,
        max_value = 8684.8,
        help = "Value range from 18.85 to 8684.8"
    )

    total_long_distance_charges = st.number_input(
        label = "Enter Total Long Distance Charges Value:",
        min_value =  0.0,
        max_value = 3564.72,
        help = "Value range from 0.0 to 3564.72"
    )

    total_revenue = st.number_input(
        label = "Enter Total revenue Value:",
        min_value = 21.61,
        max_value = 11979.34,
        help = "Value range from 21.61 to 11979.34"
    )

    # Create select box input
    gender = st.selectbox(
        label = "Enter Gender",
        options = (
            "Female",
            "Male"
        )
    )

    married = st.selectbox(
        label = "Enter Marriage Status",
        options = (
            "Y",
            "N"
        )
    )

    offer = st.selectbox(
        label = "Enter Offer Type",
        options = (
            "None",
            "Offer A",
            "Offer B",
            "Offer C",
            "Offer D",
            "Offer E",
        )
    )

    phone_service = st.selectbox(
        label = "Phone Service",
        options = (
            "Y",
            "N"
        )
    )

    multiple_lines = st.selectbox(
        label = "Multiple Lines",
        options = (
            "Y",
            "N",
            "Unknown"
        )
    )
    
    internet_service = st.selectbox(
        label = "Internet Service",
        options = (
            "Y",
            "N"
        )
    )

    internet_type = st.selectbox(
        label = "Internet Type",
        options = (
            "Cable",
            "Fiber Optic",
            "DSL",
            "Unknown"
        )
    )

    online_security = st.selectbox(
        label = "Online Security",
        options = (
            "Y",
            "N",
            "Unknown"
        )
    )

    online_backup = st.selectbox(
        label = "Online Backup",
        options = (
            "Y",
            "N",
            "Unknown"
        )
    )

    device_protection_plan = st.selectbox(
        label = "Device Protection Plan",
        options = (
            "Y",
            "N",
            "Unknown"
        )
    )

    premium_tech_support = st.selectbox(
        label = "Premium Tech Support",
        options = (
            "Y",
            "N",
            "Unknown"
        )
    )

    streaming_tv = st.selectbox(
        label = "Streaming TV",
        options = (
            "Y",
            "N",
            "Unknown"
        )
    )

    streaming_movies = st.selectbox(
        label = "Streaming Movies",
        options = (
            "Y",
            "N",
            "Unknown"
        )
    )

    streaming_music = st.selectbox(
        label = "Streaming Music",
        options = (
            "Y",
            "N",
            "Unknown"
        )
    )

    unlimited_data = st.selectbox(
        label = "Unlimited Data",
        options = (
            "Y",
            "N",
            "Unknown"
        )
    )

    contract = st.selectbox(
        label = "Contract",
        options = (
            "One Year",
            "Month-to-Month",
            "Two Year"
        )
    )
    
    paperless_billing = st.selectbox(
        label = "Paperless Billing",
        options = (
            "Y",
            "N"
        )
    )

    payment_method = st.selectbox(
        label = "Payment Method",
        options = (
            "Credit Card",
            "Bank Withdrawal",
            "Mailed Check"
        )
    )
    
    # Create button to submit the form
    submitted = st.form_submit_button("Predict")

    # Condition when form submitted
    if submitted:
        # Create dict of all data in the form
        raw_data = {
            "gender" : gender,
            "age" : age,
            "married" : married,
            "number_of_referrals" : number_of_referrals,
            "tenure_in_months" : tenure_in_months,
            "offer" : offer,
            "phone_service" : phone_service,
            "avg_monthly_long_distance_charges" : avg_monthly_long_distance_charges,
            "multiple_lines" : multiple_lines,
            "internet_service" : internet_service,
            "internet_type" : internet_type,
            "avg_monthly_gb_download" : avg_monthly_gb_download,
            "online_security" : online_security,
            "online_backup" : online_backup,
            "device_protection_plan" : device_protection_plan,
            "premium_tech_support" : premium_tech_support,
            "streaming_tv" : streaming_tv,
            "streaming_movies" : streaming_movies,
            "streaming_music" : streaming_music,
            "unlimited_data" : unlimited_data,
            "contract" : contract,
            "paperless_billing" : paperless_billing,
            "payment_method" : payment_method,
            "monthly_charge" : monthly_charge,
            "total_charges" : total_charges,
            "total_long_distance_charges" : total_long_distance_charges,
            "total_revenue" : total_revenue
        }

        # Create loading animation while predicting
        with st.spinner("Sending data to prediction server ..."):
            res = requests.post("http://localhost:8080/predict", json = raw_data).json()

        # Parse the prediction result
        if res["error_msg"] != "":
            st.error("Error Occurs While Predicting: {}".format(res["error_msg"]))
        else:
            if res["res"] != "Stayed":
                st.warning("Customer is likely to CHURN.")
            else:
                st.success("Customer is likely to STAY.")