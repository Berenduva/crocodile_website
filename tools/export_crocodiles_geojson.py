# tools/export_crocodiles_geojson.py
import os, json, time, csv
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans

# Optional geocoding (country centroids)
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

# -----------------------------
# 1) Paths (edit SRC_CSV if needed)
# -----------------------------
SRC_CSV = "data/crocodile_dataset_cleaned.csv"   # put your CSV here
OUT_CSV = "data/crocodile_dataset_with_clusters.csv"
OUT_GEOJSON = "data/crocodiles.geojson"
CENTROID_CACHE = "tools/country_centroids_cache.csv"

os.makedirs("data", exist_ok=True)
os.makedirs("tools", exist_ok=True)

# -----------------------------
# 2) Load & normalize columns
# -----------------------------
df = pd.read_csv(SRC_CSV)

df.columns = (
    df.columns
      .str.strip()
      .str.lower()
      .str.replace(" ", "_", regex=False)
      .str.replace("/", "_", regex=False)
)

required = ["country_region", "habitat_type", "scientific_name", "common_name"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in CSV: {missing}")

# -----------------------------
# 3) Clustering -> df['cluster'] in [1..20]
# -----------------------------
features = df[["country_region","habitat_type","scientific_name","common_name"]]
encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
encoded = encoder.fit_transform(features)

scaler = StandardScaler()
scaled = scaler.fit_transform(encoded)

kmeans = KMeans(n_clusters=20, n_init="auto", random_state=42)
df["cluster"] = kmeans.fit_predict(scaled) + 1

df.to_csv(OUT_CSV, index=False)
print(f"Saved: {OUT_CSV}")

# -----------------------------
# 4) Geocode unique countries (with local cache)
# -----------------------------
def read_cache(path):
    if not os.path.exists(path):
        return {}
    cache = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cache[row["country"]] = (float(row["lat"]), float(row["lng"]))
    return cache

def write_cache(path, mapping):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["country","lat","lng"])
        for k,(lat,lng) in mapping.items():
            w.writerow([k,lat,lng])

geolocator = Nominatim(user_agent="crocodile_mapper (contact@example.com)", timeout=10)
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.0, error_wait_seconds=2.0, max_retries=3)

countries = sorted(set(df["country_region"].dropna().astype(str)))
cache = read_cache(CENTROID_CACHE)
coords = {}

for c in countries:
    if c in cache:
        coords[c] = cache[c]
        continue
    try:
        loc = geocode(c)
        if loc:
            coords[c] = (float(loc.latitude), float(loc.longitude))
        else:
            coords[c] = (None, None)
    except (GeocoderTimedOut, GeocoderUnavailable, Exception):
        coords[c] = (None, None)
    time.sleep(1.0)

cache.update({k:v for k,v in coords.items() if v!=(None,None)})
write_cache(CENTROID_CACHE, cache)

df["latitude"]  = df["country_region"].map(lambda x: cache.get(str(x), (None,None))[0])
df["longitude"] = df["country_region"].map(lambda x: cache.get(str(x), (None,None))[1])

# -----------------------------
# 5) Write GeoJSON FeatureCollection
# -----------------------------
def feature(row):
    if pd.isna(row["latitude"]) or pd.isna(row["longitude"]):
        return None
    props = {
        "common_name": row.get("common_name", ""),
        "scientific_name": row.get("scientific_name",""),
        "cluster": int(row["cluster"]),
        "conservation_status": row.get("conservation_status",""),
        "habitat_type": row.get("habitat_type",""),
        "observed_length_m": row.get("observed_length_(m)",""),
        "observed_weight_kg": row.get("observed_weight_(kg)",""),
        "country_region": row.get("country_region",""),
    }
    return {
        "type": "Feature",
        "geometry": {"type":"Point", "coordinates":[float(row["longitude"]), float(row["latitude"])]},
        "properties": props
    }

features = []
for _, r in df.dropna(subset=["latitude","longitude"]).iterrows():
    f = feature(r)
    if f: features.append(f)

geo = {"type":"FeatureCollection", "features": features}
with open(OUT_GEOJSON, "w", encoding="utf-8") as f:
    json.dump(geo, f, ensure_ascii=False)

print(f"Saved: {OUT_GEOJSON}")
