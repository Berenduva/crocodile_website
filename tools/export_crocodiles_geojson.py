# tools/export_crocodiles_geojson.py
import os, json
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans

# --- Paths (repo root = one level up from /tools) ---
ROOT = os.path.dirname(os.path.dirname(__file__))
CSV_MAIN = os.path.join(ROOT, "data", "crocodile_dataset_cleaned.csv")
CSV_WITH_CLUSTERS = os.path.join(ROOT, "data", "crocodile_dataset_with_clusters.csv")
OUT_GEOJSON = os.path.join(ROOT, "data", "crocodiles.geojson")

# --- Load & normalize column names ---
df = pd.read_csv(CSV_MAIN)
df.columns = (
    df.columns
      .str.strip().str.lower()
      .str.replace(" ", "_", regex=False)
      .str.replace("/", "_", regex=False)
)

# --- Ensure required columns exist ---
req = ["country_region", "habitat_type", "scientific_name", "common_name"]
missing = [c for c in req if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

# --- If CSV already has a 'cluster' column, use it; otherwise compute it (your code) ---
if "cluster" not in df.columns:
    features = df[["country_region","habitat_type","scientific_name","common_name"]].astype(str).fillna("UNK")
    enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    encoded = enc.fit_transform(features)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(encoded)
    kmeans = KMeans(n_clusters=20, n_init=10, random_state=42)
    df["cluster"] = kmeans.fit_predict(scaled) + 1
    # also save a CSV with clusters for reuse
    df.to_csv(CSV_WITH_CLUSTERS, index=False)
else:
    # make sure clusters are 1..20 (not NaN / 0)
    df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce").fillna(0).astype(int)

# --- Country centroids (minimal set; add more if needed) ---
def norm(s): return str(s).strip().lower().replace(" ", "_").replace("/", "_")
centroids = {
    "algeria": (28.0339, 1.6596), "angola": (-11.2027, 17.8739), "benin": (9.3077, 2.3158),
    "botswana": (-22.3285, 24.6849), "burkina_faso": (12.2383, -1.5616), "cameroon": (7.3697, 12.3547),
    "central_african_republic": (6.6111, 20.9394), "chad": (15.4542, 18.7322),
    "congo": (-0.228, 15.8277), "democratic_republic_of_the_congo": (-2.9814, 23.8223),
    "cote_d_ivoire": (7.54, -5.5471), "ivory_coast": (7.54, -5.5471), "djibouti": (11.8251, 42.5903),
    "egypt": (26.8206, 30.8025), "ethiopia": (9.145, 40.4897), "gabon": (-0.8037, 11.6094),
    "ghana": (7.9465, -1.0232), "guinea": (9.9456, -9.6966), "kenya": (-0.0236, 37.9062),
    "liberia": (6.4281, -9.4295), "libya": (26.3351, 17.2283), "madagascar": (-18.7669, 46.8691),
    "malawi": (-13.2543, 34.3015), "mali": (17.5707, -3.9962), "mauritania": (21.0079, -10.9408),
    "mozambique": (-18.6657, 35.5296), "namibia": (-22.9576, 18.4904), "niger": (17.6078, 8.0817),
    "nigeria": (9.082, 8.6753), "rwanda": (-1.9403, 29.8739), "senegal": (14.4974, -14.4524),
    "sierra_leone": (8.4606, -11.7799), "somalia": (5.1521, 46.1996), "south_africa": (-30.5595, 22.9375),
    "south_sudan": (7.8627, 29.6947), "sudan": (12.8628, 30.2176), "tanzania": (-6.369, 34.8888),
    "togo": (8.6195, 0.8248), "uganda": (1.3733, 32.2903), "zambia": (-13.1339, 27.8493),
    "zimbabwe": (-19.0154, 29.1549),
    "mexico": (23.6345, -102.5528), "united_states": (37.0902, -95.7129), "usa": (37.0902, -95.7129),
    "brazil": (-14.235, -51.9253), "colombia": (4.5709, -74.2973), "peru": (-9.19, -75.0152),
    "venezuela": (6.4238, -66.5897), "ecuador": (-1.8312, -78.1834), "bolivia": (-16.2902, -63.5887),
    "india": (20.5937, 78.9629), "indonesia": (-0.7893, 113.9213), "pakistan": (30.3753, 69.3451),
    "bangladesh": (23.685, 90.3563), "sri_lanka": (7.8731, 80.7718), "china": (35.8617, 104.1954),
    "thailand": (15.87, 100.9925), "vietnam": (14.0583, 108.2772), "philippines": (12.8797, 121.774),
    "australia": (-25.2744, 133.7751), "papua_new_guinea": (-6.315, 143.9555), "myanmar": (21.9162, 95.956),
    "laos": (19.8563, 102.4955), "cambodia": (12.5657, 104.991),
}

df["latitude"]  = df["country_region"].map(lambda x: centroids.get(norm(x), (None, None))[0])
df["longitude"] = df["country_region"].map(lambda x: centroids.get(norm(x), (None, None))[1])

# --- Build GeoJSON ---
features = []
for _, row in df.dropna(subset=["latitude","longitude"]).iterrows():
    props = {
        "common_name": row.get("common_name",""),
        "scientific_name": row.get("scientific_name",""),
        "cluster": int(row.get("cluster", 0)),
        "conservation_status": row.get("conservation_status",""),
        "habitat_type": row.get("habitat_type",""),
        "observed_length_m": row.get("observed_length_(m)",""),
        "observed_weight_kg": row.get("observed_weight_(kg)",""),
        "country_region": row.get("country_region",""),
    }
    feat = {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [float(row["longitude"]), float(row["latitude"])]},
        "properties": props,
    }
    features.append(feat)

os.makedirs(os.path.dirname(OUT_GEOJSON), exist_ok=True)
with open(OUT_GEOJSON, "w", encoding="utf-8") as f:
    json.dump({"type":"FeatureCollection", "features": features}, f, ensure_ascii=False)

print(f"Wrote {len(features)} features with clusters to {OUT_GEOJSON}")
