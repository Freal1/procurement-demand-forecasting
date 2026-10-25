from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go

from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# =========================================================
# 1. KONFIGURASI
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

HISTORY_FILE = BASE_DIR / "historical_demand.csv"
XGB_FILE = BASE_DIR / "xgboost_model.json"
LGB_FILE = BASE_DIR / "lightgbm_model.joblib"
RF_FILE = BASE_DIR / "random_forest_model.joblib"
META_FILE = BASE_DIR / "ensemble_metadata.joblib"
SCALER_FILE = BASE_DIR / "target_scaler.joblib"
LSTM_FILE = BASE_DIR / "LSTM_Demand_Forecasting.keras"
LSTM_FEATURES_FILE = BASE_DIR / "LSTM_Features.json"

st.set_page_config(
    page_title="Procurement Demand Forecasting",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Procurement Demand Forecasting")
st.caption(
    "Peramalan kebutuhan pembelian berdasarkan historical demand per Article."
)


# =========================================================
# 2. MEMBACA HISTORICAL DATA
# =========================================================

@st.cache_data
def load_history():
    if not HISTORY_FILE.exists():
        raise FileNotFoundError(
            f"File {HISTORY_FILE.name} tidak ditemukan."
        )

    df = pd.read_csv(
        HISTORY_FILE,
        sep=None,
        engine="python",
        dtype=str,
        encoding="utf-8-sig",
    )

    df.columns = [
        str(c).replace("\ufeff", "").strip().lower()
        for c in df.columns
    ]

    required = {"article", "date", "demand"}

    if not required.issubset(df.columns):
        raise ValueError(
            "CSV harus mempunyai kolom Article, Date, dan Demand. "
            f"Kolom yang ditemukan: {df.columns.tolist()}"
        )

    df = df[["article", "date", "demand"]].copy()
    df.columns = ["Article", "Date", "Demand"]

    df["Article"] = df["Article"].astype(str).str.strip()

    date_text = df["Date"].astype(str).str.strip()

    df["Date"] = pd.to_datetime(
        date_text,
        format="%d/%m/%Y",
        errors="coerce",
    )

    missing_date = df["Date"].isna()

    if missing_date.any():
        df.loc[missing_date, "Date"] = pd.to_datetime(
            date_text[missing_date],
            errors="coerce",
            dayfirst=True,
        )

    demand_text = (
        df["Demand"]
        .astype(str)
        .str.strip()
        .str.replace(",", "", regex=False)
    )

    df["Demand"] = pd.to_numeric(
        demand_text,
        errors="coerce",
    )

    df = df.dropna(subset=["Article", "Date", "Demand"])

    df = df[
        (df["Article"] != "")
        & (df["Article"].str.lower() != "nan")
    ]

    if df.empty:
        raise ValueError("Tidak ada data historis yang valid.")

    df["Month_Start"] = (
        df["Date"].dt.to_period("M").dt.to_timestamp()
    )

    monthly = (
        df.groupby(
            ["Article", "Month_Start"],
            as_index=False,
        )["Demand"]
        .sum()
        .sort_values(["Article", "Month_Start"])
    )

    return monthly


def get_monthly_series(data, article):
    part = data.loc[
        data["Article"] == article,
        ["Month_Start", "Demand"],
    ].copy()

    if part.empty:
        raise ValueError(
            f"Tidak ada historical data untuk Article {article}."
        )

    series = part.set_index("Month_Start")["Demand"].sort_index()

    full_index = pd.date_range(
        start=series.index.min(),
        end=series.index.max(),
        freq="MS",
    )

    series = series.reindex(full_index, fill_value=0.0)
    series.index.name = "Bulan"

    return series.astype(float)


# =========================================================
# 3. MEMUAT TREE MODELS DAN METADATA
# =========================================================

@st.cache_resource
def load_tree_models():
    models = {}
    errors = {}

    if XGB_FILE.exists():
        try:
            model = XGBRegressor()
            model.load_model(str(XGB_FILE))
            models["XGBoost"] = model
        except Exception as exc:
            errors["XGBoost"] = str(exc)
    else:
        errors["XGBoost"] = "File model tidak ditemukan."

    if LGB_FILE.exists():
        try:
            models["LightGBM"] = joblib.load(LGB_FILE)
        except Exception as exc:
            errors["LightGBM"] = str(exc)
    else:
        errors["LightGBM"] = "File model tidak ditemukan."

    if RF_FILE.exists():
        try:
            models["Random Forest"] = joblib.load(RF_FILE)
        except Exception as exc:
            errors["Random Forest"] = str(exc)
    else:
        errors["Random Forest"] = "File model tidak ditemukan."

    metadata = {}

    if META_FILE.exists():
        try:
            metadata = joblib.load(META_FILE)
        except Exception as exc:
            errors["Metadata Ensemble"] = str(exc)
    else:
        errors["Metadata Ensemble"] = "Metadata ensemble tidak ditemukan."

    return models, metadata, errors


# =========================================================
# 4. MEMUAT LSTM DENGAN KOMPATIBILITAS DENSE
# =========================================================

def load_lstm_model():
    """
    Mencoba memuat model dengan Dense kompatibel.
    Jika deserialisasi tetap gagal, membangun ulang arsitektur
    yang terlihat dari konfigurasi dan mencoba memuat bobot.

    Pemuatan bobot hanya dilakukan jika file .keras dapat dibuka
    sebagai arsip dan bobotnya cocok dengan arsitektur.
    """

    if not LSTM_FILE.exists():
        return None, f"File {LSTM_FILE.name} tidak ditemukan."

    errors = []

    # -----------------------------------------------------
    # Percobaan 1: TensorFlow Keras
    # -----------------------------------------------------
    try:
        import tensorflow as tf

        class CompatibleDense(tf.keras.layers.Dense):
            @classmethod
            def from_config(cls, config):
                config = dict(config)
                config.pop("quantization_config", None)
                return cls(**config)

        model = tf.keras.models.load_model(
            str(LSTM_FILE),
            compile=False,
            custom_objects={
                "Dense": CompatibleDense,
                "keras.layers.Dense": CompatibleDense,
            },
            safe_mode=False,
        )

        return model, None

    except Exception as exc:
        errors.append(f"TensorFlow Keras: {exc}")

    # -----------------------------------------------------
    # Percobaan 2: Keras mandiri
    # -----------------------------------------------------
    try:
        import keras

        class CompatibleDense(keras.layers.Dense):
            @classmethod
            def from_config(cls, config):
                config = dict(config)
                config.pop("quantization_config", None)
                return cls(**config)

        model = keras.models.load_model(
            str(LSTM_FILE),
            compile=False,
            custom_objects={
                "Dense": CompatibleDense,
                "keras.layers.Dense": CompatibleDense,
            },
            safe_mode=False,
        )

        return model, None

    except Exception as exc:
        errors.append(f"Keras: {exc}")

    # -----------------------------------------------------
    # Percobaan 3: Bangun arsitektur dan muat bobot
    # -----------------------------------------------------
    try:
        import tensorflow as tf

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(5, 1)),
            tf.keras.layers.LSTM(
                64,
                activation="relu",
                return_sequences=True,
            ),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(
                32,
                activation="relu",
                return_sequences=False,
            ),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(
                16,
                activation="relu",
            ),
            tf.keras.layers.Dense(
                1,
                activation="linear",
            ),
        ])

        model.build(input_shape=(None, 5, 1))

        # load_weights dari file .keras bergantung pada format
        # dan versi Keras. Gagalnya langkah ini akan ditangkap.
        model.load_weights(str(LSTM_FILE))

        return model, None

    except Exception as exc:
        errors.append(f"Rebuild arsitektur/load weights: {exc}")

    return None, "\n\n".join(errors)


@st.cache_resource
def load_lstm_cached():
    return load_lstm_model()


# =========================================================
# 5. LOAD SCALER
# =========================================================

@st.cache_resource
def load_target_scaler():
    if not SCALER_FILE.exists():
        return None, f"File {SCALER_FILE.name} tidak ditemukan."

    try:
        return joblib.load(SCALER_FILE), None
    except Exception as exc:
        return None, str(exc)


# =========================================================
# 6. DIAGNOSTIK FITUR MODEL
# =========================================================

def get_expected_features(model):
    names = getattr(model, "feature_names_in_", None)

    if names is not None:
        return list(names)

    names = getattr(model, "feature_name_", None)

    if names is not None:
        return list(names)

    try:
        names = model.get_booster().feature_names
        if names:
            return list(names)
    except Exception:
        pass

    try:
        names = model.booster_.feature_name()
        if names:
            return list(names)
    except Exception:
        pass

    return None


def get_expected_feature_count(model):
    value = getattr(model, "n_features_in_", None)

    if value is not None:
        return int(value)

    try:
        return int(model.n_features_)
    except Exception:
        pass

    try:
        return int(model.num_feature())
    except Exception:
        pass

    return None


# =========================================================
# 7. MEMBANGUN FITUR TREE MODEL
# =========================================================

def build_feature_row(history_scaled, date):
    h = np.asarray(history_scaled, dtype=float)

    if len(h) < 12:
        raise ValueError(
            "Model fitur lag membutuhkan minimal 12 bulan data."
        )

    r3 = h[-3:]
    r6 = h[-6:]

    values = {
        "Year": int(date.year),
        "Month": int(date.month),
        "Month_sin": float(
            np.sin(2 * np.pi * date.month / 12)
        ),
        "Month_cos": float(
            np.cos(2 * np.pi * date.month / 12)
        ),
        "Is_High_Season": int(date.month in [11, 12]),
        "Is_Q4": int(date.month in [10, 11, 12]),
        "Lag_1": float(h[-1]),
        "Lag_2": float(h[-2]),
        "Lag_3": float(h[-3]),
        "Lag_6": float(h[-6]),
        "Lag_12": float(h[-12]),
        "Rolling_Mean_3": float(np.mean(r3)),
        "Rolling_Mean_6": float(np.mean(r6)),
        "Rolling_Std_3": float(np.std(r3, ddof=1)),
    }

    return pd.DataFrame([values])


def predict_tree_model(model, row, metadata_features):
    expected = get_expected_features(model)

    if expected:
        missing = [name for name in expected if name not in row.columns]

        if missing:
            raise ValueError(
                "Fitur yang dibutuhkan model tidak tersedia: "
                f"{missing}"
            )

        X = row[expected].copy()

    else:
        expected_count = get_expected_feature_count(model)

        if not metadata_features:
            raise ValueError(
                "Nama fitur tidak tersimpan pada model dan "
                "metadata.features kosong. Periksa script training."
            )

        metadata_features = list(metadata_features)

        missing = [
            name for name in metadata_features
            if name not in row.columns
        ]

        if missing:
            raise ValueError(
                f"Fitur metadata tidak tersedia: {missing}"
            )

        if (
            expected_count is not None
            and len(metadata_features) != expected_count
        ):
            raise ValueError(
                f"Model membutuhkan {expected_count} fitur, "
                f"metadata menyediakan {len(metadata_features)} fitur."
            )

        X = row[metadata_features].copy()

    if (
        get_expected_feature_count(model) is not None
        and X.shape[1] != get_expected_feature_count(model)
    ):
        raise ValueError(
            f"Jumlah fitur input {X.shape[1]} tidak sesuai dengan model."
        )

    prediction = np.asarray(model.predict(X)).reshape(-1)

    if len(prediction) == 0 or not np.isfinite(prediction[0]):
        raise ValueError("Prediksi model tidak valid.")

    return float(prediction[0])


# =========================================================
# 8. PREDIKSI LSTM
# =========================================================

def predict_lstm_model(model, history_scaled):
    input_shape = model.input_shape

    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    if len(input_shape) != 3:
        raise ValueError(
            f"Bentuk input LSTM tidak didukung: {input_shape}"
        )

    sequence_length = input_shape[1]
    feature_count = input_shape[2]

    if sequence_length is None or feature_count is None:
        raise ValueError(
            f"Dimensi input LSTM tidak pasti: {input_shape}"
        )

    if int(feature_count) != 1:
        raise ValueError(
            f"LSTM membutuhkan {feature_count} fitur per timestep."
        )

    history = np.asarray(history_scaled, dtype=np.float32)

    if len(history) < int(sequence_length):
        raise ValueError(
            f"LSTM membutuhkan minimal {sequence_length} data."
        )

    sequence = history[-int(sequence_length):].reshape(
        1,
        int(sequence_length),
        1,
    )

    prediction = model.predict(sequence, verbose=0)
    prediction = np.asarray(prediction).reshape(-1)

    if len(prediction) == 0 or not np.isfinite(prediction[0]):
        raise ValueError("LSTM tidak menghasilkan prediksi yang valid.")

    return float(prediction[0])


# =========================================================
# 9. FORECAST BULANAN
# =========================================================

def forecast(
    data,
    article,
    horizon,
    tree_models,
    metadata,
    lstm_model,
    scaler,
):
    raw = get_monthly_series(data, article)

    if len(raw) < 12:
        raise ValueError(
            "Article harus memiliki minimal 12 bulan data historis."
        )

    last_date = raw.index.max()

    forecast_dates = pd.date_range(
        start=last_date + pd.offsets.MonthBegin(1),
        periods=horizon,
        freq="MS",
    )

    scaled = scaler.transform(
        raw.to_numpy().reshape(-1, 1)
    ).reshape(-1)

    history_scaled = list(np.asarray(scaled, dtype=float))

    metadata_features = metadata.get("features", [])

    weights = metadata.get(
        "weights",
        metadata.get("ensemble_weights", {}),
    )

    # Menyamakan penulisan nama model.
    normalized_weights = {}

    if isinstance(weights, dict):
        aliases = {
            "xgb": "XGBoost",
            "xgboost": "XGBoost",
            "lgb": "LightGBM",
            "lgbm": "LightGBM",
            "lightgbm": "LightGBM",
            "rf": "Random Forest",
            "random forest": "Random Forest",
            "random_forest": "Random Forest",
            "lstm": "LSTM",
        }

        for key, value in weights.items():
            name = aliases.get(str(key).strip().lower())

            if name is not None:
                try:
                    normalized_weights[name] = float(value)
                except (TypeError, ValueError):
                    pass

    results = []
    errors = {}

    for date in forecast_dates:
        feature_row = build_feature_row(history_scaled, date)
        predictions_scaled = {}

        # Tree models
        for name, model in tree_models.items():
            try:
                predictions_scaled[name] = predict_tree_model(
                    model,
                    feature_row,
                    metadata_features,
                )
            except Exception as exc:
                errors[name] = str(exc)

        # LSTM

        if lstm_model is not None:
            try:
                st.write("### Diagnostik Prediksi LSTM")

                st.write("Input shape:", lstm_model.input_shape)
                st.write("Panjang data historis:", len(history_scaled))
                st.write("Lima data terakhir:", history_scaled[-5:])

                lstm_prediction = predict_lstm_model(
                    lstm_model,
                    history_scaled,
                )

                st.write("Prediksi dalam skala model:", lstm_prediction)

                predictions_scaled["LSTM"] = lstm_prediction

            except Exception as exc:
                errors["LSTM"] = repr(exc)
                st.error(f"LSTM gagal memprediksi: {repr(exc)}")
        else:
            st.error(
                "LSTM tidak dijalankan karena model belum berhasil dimuat."
            )

        # Ensemble hanya menggunakan bobot yang cocok.
        weighted = {
            name: weight
            for name, weight in normalized_weights.items()
            if name in predictions_scaled
            and np.isfinite(weight)
            and weight >= 0
        }

        ensemble_scaled = None

        if weighted and sum(weighted.values()) > 0:
            total_weight = sum(weighted.values())

            ensemble_scaled = sum(
                predictions_scaled[name] * weight
                for name, weight in weighted.items()
            ) / total_weight

        # Kembalikan hasil ke skala asli.
        original = {}

        for name, value in predictions_scaled.items():
            inverse_value = float(
                scaler.inverse_transform([[value]])[0, 0]
            )
            original[name] = max(0.0, inverse_value)

        ensemble_original = np.nan

        if ensemble_scaled is not None:
            ensemble_original = max(
                0.0,
                float(
                    scaler.inverse_transform(
                        [[ensemble_scaled]]
                    )[0, 0]
                ),
            )

        results.append({
            "Bulan": date.strftime("%B %Y"),
            "XGBoost": original.get("XGBoost", np.nan),
            "LightGBM": original.get("LightGBM", np.nan),
            "Random Forest": original.get("Random Forest", np.nan),
            "LSTM": original.get("LSTM", np.nan),
            "Ensemble": ensemble_original,
        })

        # Gunakan ensemble untuk autoregressive update jika tersedia.
        # Jika tidak, gunakan LSTM atau rata-rata model yang berhasil
        # hanya sebagai input lag, bukan sebagai hasil ensemble.
        if ensemble_scaled is not None:
            next_value = ensemble_scaled
        elif "LSTM" in predictions_scaled:
            next_value = predictions_scaled["LSTM"]
        elif predictions_scaled:
            next_value = float(
                np.mean(list(predictions_scaled.values()))
            )
        else:
            raise ValueError(
                f"Tidak ada model yang berhasil memprediksi "
                f"{date:%B %Y}. Periksa diagnostik model."
            )

        history_scaled.append(float(next_value))

    return pd.DataFrame(results), errors


# =========================================================
# 10. LOAD SEMUA KOMPONEN
# =========================================================

try:
    data = load_history()
except Exception as exc:
    st.error(f"Gagal membaca historical data: {exc}")
    st.stop()

tree_models, metadata, model_errors = load_tree_models()
lstm_model, lstm_error = load_lstm_cached()
scaler, scaler_error = load_target_scaler()


# =========================================================
# 11. SIDEBAR
# =========================================================

articles = sorted(data["Article"].unique())

with st.sidebar:
    st.header("⚙️ Pengaturan Forecast")

    article = st.selectbox(
        "Pilih Article",
        articles,
    )

    horizon = st.selectbox(
        "Periode Forecast",
        [1, 3, 6, 9, 12],
        index=4,
        format_func=lambda x: f"{x} bulan",
    )


# =========================================================
# 12. STATUS MODEL
# =========================================================

with st.expander("Status Model dan Diagnostik", expanded=False):
    st.write("### Tree Models")

    for name in ["XGBoost", "LightGBM", "Random Forest"]:
        if name in tree_models:
            st.success(f"{name}: berhasil dimuat")
        else:
            st.error(
                f"{name}: {model_errors.get(name, 'Tidak tersedia')}"
            )

    st.write("### LSTM")

    if lstm_model is not None:
        st.success("LSTM: berhasil dimuat")
        st.write("Input shape:", lstm_model.input_shape)
        st.write("Output shape:", lstm_model.output_shape)
    else:
        st.error(f"LSTM gagal dimuat:\n{lstm_error}")

    st.write("### Scaler")

    if scaler is not None:
        st.success("Target scaler berhasil dimuat")
    else:
        st.error(f"Target scaler gagal dimuat: {scaler_error}")

    st.write("### Metadata Ensemble")
    st.write("Features:", metadata.get("features", []))
    st.write(
        "Weights:",
        metadata.get(
            "weights",
            metadata.get("ensemble_weights", {}),
        ),
    )


# =========================================================
# 13. RINGKASAN ARTICLE
# =========================================================

series = get_monthly_series(data, article)

col1, col2, col3 = st.columns(3)

col1.metric("Article", article)
col2.metric("Bulan historis", len(series))
col3.metric(
    "Demand bulan terakhir",
    f"{series.iloc[-1]:,.2f}",
)


# =========================================================
# 14. GRAFIK HISTORICAL PEMBELIAN
#     TIDAK DIGABUNGKAN DENGAN FORECAST
# =========================================================

st.subheader(f"Historical Pembelian — {article}")

fig_history = go.Figure()

fig_history.add_trace(
    go.Scatter(
        x=series.index,
        y=series.values,
        mode="lines+markers",
        name="Demand Aktual",
        line=dict(width=2),
        marker=dict(size=5),
        hovertemplate=(
            "Bulan: %{x|%B %Y}<br>"
            "Demand Aktual: %{y:,.2f}"
            "<extra></extra>"
        ),
    )
)

fig_history.update_layout(
    template="plotly_dark",
    height=430,
    xaxis_title="Bulan",
    yaxis_title="Demand Aktual",
    hovermode="x unified",
    margin=dict(l=20, r=20, t=25, b=20),
)

st.plotly_chart(
    fig_history,
    use_container_width=True,
)

with st.expander("Lihat tabel historical pembelian"):
    history_table = (
        series.rename("Demand Aktual")
        .rename_axis("Bulan")
        .reset_index()
    )

    st.dataframe(
        history_table,
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# 15. JALANKAN FORECAST
# =========================================================

st.divider()
st.subheader(f"Forecast Demand — {article}")

if scaler is None:
    st.error(
        "Forecast belum dapat dijalankan karena target scaler "
        "tidak berhasil dimuat."
    )

    st.info(
        "Pastikan target_scaler.joblib adalah scaler yang digunakan "
        "ketika training. Jangan menggantinya dengan scaler baru "
        "tanpa melakukan training ulang."
    )

elif st.button("Buat Forecast", type="primary"):

    if not tree_models and lstm_model is None:
        st.error("Tidak ada model yang berhasil dimuat.")
        st.stop()

    try:
        result, prediction_errors = forecast(
            data=data,
            article=article,
            horizon=horizon,
            tree_models=tree_models,
            metadata=metadata,
            lstm_model=lstm_model,
            scaler=scaler,
        )

        # Tabel hasil prediksi
        st.dataframe(
            result.style.format({
                "XGBoost": "{:,.2f}",
                "LightGBM": "{:,.2f}",
                "Random Forest": "{:,.2f}",
                "LSTM": "{:,.2f}",
                "Ensemble": "{:,.2f}",
            }, na_rep="Tidak tersedia"),
            use_container_width=True,
            hide_index=True,
        )

        # Grafik forecast terpisah dari historical
        forecast_columns = [
            name for name in [
                "XGBoost",
                "LightGBM",
                "Random Forest",
                "LSTM",
                "Ensemble",
            ]
            if result[name].notna().any()
        ]

        if forecast_columns:
            fig_forecast = go.Figure()

            for name in forecast_columns:
                fig_forecast.add_trace(
                    go.Scatter(
                        x=result["Bulan"],
                        y=result[name],
                        mode="lines+markers",
                        name=name,
                    )
                )

            fig_forecast.update_layout(
                template="plotly_dark",
                height=430,
                xaxis_title="Bulan Forecast",
                yaxis_title="Prediksi Demand",
                hovermode="x unified",
                margin=dict(l=20, r=20, t=25, b=20),
            )

            st.plotly_chart(
                fig_forecast,
                use_container_width=True,
            )
        else:
            st.warning("Belum ada model yang menghasilkan prediksi.")

        # Ringkasan total prediksi per model
        st.subheader("Ringkasan Forecast")

        summary_columns = st.columns(5)

        model_names = [
            "XGBoost",
            "LightGBM",
            "Random Forest",
            "LSTM",
            "Ensemble",
        ]

        for col, name in zip(summary_columns, model_names):
            valid = result[name].dropna()

            if not valid.empty:
                col.metric(
                    f"Total {name}",
                    f"{valid.sum():,.2f}",
                )
            else:
                col.metric(
                    f"Total {name}",
                    "Tidak tersedia",
                )

        # Download
        st.download_button(
            "Unduh hasil forecast CSV",
            data=result.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"forecast_{article}.csv",
            mime="text/csv",
        )

        # Detail error
        if prediction_errors:
            with st.expander(
                "Detail model yang gagal memprediksi",
                expanded=True,
            ):
                for name, error in prediction_errors.items():
                    st.error(f"{name}: {error}")

        if result["LSTM"].notna().any():
            st.success("LSTM berhasil menghasilkan prediksi.")
        else:
            st.warning(
                "LSTM belum menghasilkan prediksi. "
                "Periksa status model dan detail error."
            )

        if result["Ensemble"].isna().all():
            st.warning(
                "Ensemble tidak tersedia. Periksa bobot pada "
                "ensemble_metadata.joblib. Aplikasi tidak otomatis "
                "menganggap rata-rata model sebagai ensemble stacking."
            )

    except Exception as exc:
        st.error(f"Forecast gagal: {exc}")
