import pandas as pd
import streamlit as st


@st.cache_data
def load_data(file_path):

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    df.drop_duplicates(inplace=True)

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    q1 = df[numeric_cols].quantile(0.25)
    q3 = df[numeric_cols].quantile(0.75)
    iqr = q3 - q1

    df = df[
        ~(
            (df[numeric_cols] < (q1 - 1.5 * iqr)) |
            (df[numeric_cols] > (q3 + 1.5 * iqr))
        ).any(axis=1)
    ]

    df.reset_index(drop=True, inplace=True)
    return df


def apply_global_filters(df):

    st.sidebar.markdown("## 🎛️ Global Filters")

    # ---------------- DEFAULTS ---------------- #
    defaults = {
        "quality": (int(df["quality"].min()), int(df["quality"].max())),
        "alcohol": (float(df["alcohol"].min()), float(df["alcohol"].max())),
        "ph": (float(df["pH"].min()), float(df["pH"].max())),
        "sulphates": (float(df["sulphates"].min()), float(df["sulphates"].max())),
        "acidity": (float(df["fixed acidity"].min()), float(df["fixed acidity"].max())),
    }

    # ---------------- INIT STATE ONLY ONCE ---------------- #
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # =========================
    # RESET BUTTON (BOTTOM SAFE)
    # =========================
    st.sidebar.markdown("---")

    if st.sidebar.button("🔄 Reset Filters", use_container_width=True):

        # IMPORTANT: update all at once
        st.session_state.update(defaults)

        st.rerun()

    # ---------------- SLIDERS ---------------- #
    quality = st.sidebar.slider(
        "🍷 Quality Range",
        int(df["quality"].min()),
        int(df["quality"].max()),
        key="quality"
    )

    alcohol = st.sidebar.slider(
        "🥂 Alcohol Range",
        float(df["alcohol"].min()),
        float(df["alcohol"].max()),
        key="alcohol"
    )

    ph = st.sidebar.slider(
        "🧪 pH Range",
        float(df["pH"].min()),
        float(df["pH"].max()),
        key="ph"
    )

    sulphates = st.sidebar.slider(
        "⚗️ Sulphates Range",
        float(df["sulphates"].min()),
        float(df["sulphates"].max()),
        key="sulphates"
    )

    acidity = st.sidebar.slider(
        "🧫 Fixed Acidity Range",
        float(df["fixed acidity"].min()),
        float(df["fixed acidity"].max()),
        key="acidity"
    )

    # ---------------- FILTER ---------------- #
    filtered_df = df[
        (df["quality"] >= quality[0]) &
        (df["quality"] <= quality[1]) &
        (df["alcohol"] >= alcohol[0]) &
        (df["alcohol"] <= alcohol[1]) &
        (df["pH"] >= ph[0]) &
        (df["pH"] <= ph[1]) &
        (df["sulphates"] >= sulphates[0]) &
        (df["sulphates"] <= sulphates[1]) &
        (df["fixed acidity"] >= acidity[0]) &
        (df["fixed acidity"] <= acidity[1])
    ]

    #------Download Button----------------------------------------
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📥 Export Data")

    csv = filtered_df.to_csv(index=False).encode("utf-8")

    st.sidebar.download_button(
        label="⬇️ Download Clean Data",
        data=csv,
        file_name="clean_filtered_wine_data.csv",
        mime="text/csv",
        use_container_width=True
    )
    st.markdown("""
<style>

/* DARK DOWNLOAD BUTTON */
div.stDownloadButton > button {
    background: rgba(17,24,39,0.9) !important;   /* dark glass */
    color: #94a3b8 !important;                   /* muted text */
    border: 1px solid rgba(148,163,184,0.2) !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    padding: 0.45rem 1.1rem !important;
    transition: all 0.2s ease !important;
    box-shadow: none !important;
}

/* HOVER (subtle only, no neon) */
div.stDownloadButton > button:hover {
    background: rgba(31,41,55,0.95) !important;
    color: #e5e7eb !important;
    border: 1px solid rgba(148,163,184,0.4) !important;
    transform: translateY(-1px);
}

/* ACTIVE CLICK */
div.stDownloadButton > button:active {
    transform: scale(0.98);
}

</style>
""", unsafe_allow_html=True)

    return filtered_df