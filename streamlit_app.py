
import streamlit as st
import requests
import pandas as pd

st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000/predict"

# Example patient features, in the model's expected order
DEFAULT_FEATURES = [
    19.55, 28.77, 133.6, 1207.0, 0.0926,
    0.2063, 0.1784, 0.1144, 0.1893, 0.06232,
    0.8426, 1.199, 7.158, 106.4, 0.006356,
    0.04765, 0.03863, 0.01519, 0.01936, 0.005252,
    25.05, 36.27, 178.6, 1926.0, 0.1281,
    0.5329, 0.4251, 0.1941, 0.2818, 0.1005
]

st.title("🩺 Breast Cancer Prediction")
st.caption(
    "Enter patient measurements manually or adjust individual features "
    "and observe how the feature-value graph changes."
)

st.info(
    "Enter features in exactly the same order used during model training. "
    
)

# Initialize editable values
if "features" not in st.session_state:
    st.session_state.features = DEFAULT_FEATURES.copy()

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

mode = st.radio(
    "Choose input method",
    ["Manual Input", "Interactive Sliders"],
    horizontal=True
)

features = st.session_state.features.copy()

if mode == "Manual Input":
    with st.form("manual_form"):
        features_text = st.text_area(
            "Enter 30 comma-separated numerical features",
            value=", ".join(map(str, features)),
            height=130
        )

        submitted = st.form_submit_button(
            "Predict",
            type="primary",
            use_container_width=True
        )

    if submitted:
        try:
            values = [
                float(v.strip())
                for v in features_text.split(",")
                if v.strip()
            ]

            if len(values) != 30:
                st.error(
                    f"Expected 30 features; received {len(values)}."
                )
            elif not all(pd.notna(v) and abs(v) != float("inf")
                         for v in values):
                st.error("All features must be finite numbers.")
            else:
                st.session_state.features = values
                features = values

                try:
                    response = requests.post(
                        API_URL,
                        json={"features": features},
                        timeout=10
                    )
                    response.raise_for_status()
                    st.session_state.prediction_result = response.json()
                except requests.RequestException as error:
                    st.error(f"Could not connect to FastAPI: {error}")

        except ValueError:
            st.error("Please enter only numbers separated by commas.")

else:
    st.write("Adjust any feature below. The graph updates as you move a slider.")

    with st.expander("Edit features", expanded=True):
        for i, initial in enumerate(DEFAULT_FEATURES):
            # Sliders use example-relative ranges, not clinical limits.
            lower = 0.0
            upper = max(initial * 2.0, initial + 0.001)
            step = max(abs(initial) / 100, 0.000001)

            features[i] = st.slider(
                f"Feature {i + 1}",
                min_value=float(lower),
                max_value=float(upper),
                value=float(st.session_state.features[i]),
                step=float(step),
                key=f"feature_slider_{i}"
            )

    st.session_state.features = features

    submitted = st.button(
        "Predict",
        type="primary",
        use_container_width=True
    )

    if submitted:
        try:
            response = requests.post(
                API_URL,
                json={"features": features},
                timeout=10
            )
            response.raise_for_status()
            st.session_state.prediction_result = response.json()
        except requests.RequestException as error:
            st.error(f"Could not connect to FastAPI: {error}")

# Feature-value visualization
st.subheader("Feature Values")
chart_data = pd.DataFrame({
    "Feature": [f"Feature {i}" for i in range(1, 31)],
    "Value": st.session_state.features
}).set_index("Feature")

st.line_chart(chart_data, height=350)

with st.expander("View current feature values"):
    st.dataframe(chart_data, use_container_width=True)

# Show latest prediction
result = st.session_state.prediction_result

if result is not None:
    if "error" in result:
        st.error(result["error"])
    elif "prediction" in result:
        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Predicted class", result["prediction"])

        with col2:
            st.metric(
                "Class 1 probability",
                f"{result['class_1_probability']:.4%}"
            )

        st.caption(
            "Class meaning depends on your dataset's label mapping. "
            
        )
