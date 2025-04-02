import os
import json
import re
import pandas as pd
import requests
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
from dateutil import parser
from sklearn.cluster import DBSCAN
import folium
import unicodedata
import seaborn as sns
import contextily as ctx
import geopandas as gpd
from shapely.geometry import Point
import math
from wordcloud import WordCloud


from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("PLACES_API_KEY")
CACHE_FILE = "api_cache.json"
RICHMOND_CENTER = (37.571941, -77.516876)
UVA_CENTER = (38.034, -78.505)

DEFAULT_EPS = 0.0003
DEFAULT_MIN_SAMPLES = 4

MANUAL_LABELS = {
    0.0: "UVA Apartment",
    1.0: "Unknown 1",
    2.0: "Barracks Shopping Center",
    3.0: "Mad Bowl",
    4.0: "Main Street e.g. Atkins' Apartment",
    5.0: "University Ave Stop Sign",
    6.0: "Rice Hall",
    7.0: "Unknown 2",
    8.0: "Unknown 3",
    9.0: "Engineering Parking",
    10.0: "Unknown 4",
    11.0: "1407-1499 Grady Ave",
    12.0: "Stadium Road",
    13.0: "Dad's House (Richmond)",
    14.0: "6009-6007 Three Chopt Rd",
    15.0: "Mom's House (Richmond)",
    16.0: "113-115 Gaymont Rd",
    17.0: "6413-6401 Roselawn Rd",
    18.0: "5699-5605 VA-147",
    19.0: "Richmond",
    20.0: "Shops Around Dad's House",
    21.0: "973-801 US-29 BUS",
    22.0: "488-498 Ridge McIntire Rd",
    23.0: "1527-1617 US-250 BUS",
    24.0: "Mid-Corner",
    25.0: "University Avenue",
    26.0: "Unknown 5",
    27.0: "Unknown 6",
    28.0: "Unknown 7",
}
USE_MANUAL_LABELS = True


def clean_coordinate_string(s):
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode()
    s = s.replace("°", "").strip()
    return s

def load_cache(cache_file=CACHE_FILE):
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)
    else:
        return {}

def save_cache(cache_dict, cache_file=CACHE_FILE):
    with open(cache_file, 'w') as f:
        json.dump(cache_dict, f, indent=2)

def load_timeline_json(filename, days=30):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    points = []
    cutoff_date = datetime.now().astimezone() - timedelta(days=days)
    
    for segment in data.get("semanticSegments", []):
        start_time_str = segment["startTime"]
        start_dt = parser.parse(start_time_str)
        
        if start_dt < cutoff_date:
            continue
        
        if "timelinePath" in segment:
            for entry in segment["timelinePath"]:
                if "point" not in entry or "," not in entry["point"]:
                    continue
                lat_lon_str = entry["point"]
                lat_str, lon_str = lat_lon_str.split(",")
                lat = float(clean_coordinate_string(lat_str))
                lon = float(clean_coordinate_string(lon_str))
                
                time_str = entry["time"]
                try:
                    time_dt = parser.parse(entry["time"])
                except (ValueError, KeyError):
                    continue
                
                if time_dt >= cutoff_date:
                    points.append({
                        "timestamp": time_dt,
                        "latitude": lat,
                        "longitude": lon
                    })
    
    df = pd.DataFrame(points)
    if len(df) == 0:
        return pd.DataFrame(columns=["timestamp", "latitude", "longitude"])
    
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def cluster_points(df, eps=0.0005, min_samples=5):
    if df.empty or "latitude" not in df or "longitude" not in df:
        raise ValueError("timelinePaths DataFrame missing lat/lon or is empty.")
    coords = df[["latitude", "longitude"]].values
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(coords)
    df["cluster"] = labels
    return df

def get_cluster_centroids(df):
    centroids = df.groupby("cluster")[["latitude", "longitude"]].mean().reset_index()
    centroids = centroids[centroids["cluster"] != -1]
    return centroids


def query_place_details(place_id, api_key):
    if not place_id:
        return ""
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {
        "place_id": place_id,
        "fields": "editorial_summary,reviews",
        "key": api_key
    }
    try:
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        summary = data.get("result", {}).get("editorial_summary", {}).get("overview", "")
        reviews = data.get("result", {}).get("reviews", [])
        rev_texts = [rv.get("text", "") for rv in reviews[:3]]
        return summary + " " + " ".join(rev_texts)
    except Exception as e:
        print("Detail API error:", e)
        return ""


def query_google_places(lat, lon, api_key, cache_dict):
    lat_lon_key = f"{round(lat, 5)}:{round(lon, 5)}"
    if lat_lon_key in cache_dict:
        return cache_dict[lat_lon_key]
    
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "key": api_key,
        "location": f"{lat},{lon}",
        "radius": 50 # meters
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data.get("status") == "OK" and data.get("results"):
            candidates = [
                p for p in data["results"]
                if any(t in p.get("types", []) for t in ["point_of_interest", "establishment", "university", "school", "restaurant", "library"])
            ]
            first = candidates[0] if candidates else data["results"][0]

            details = query_place_details(first.get("place_id"), api_key)
            place_info = {
                "name": first.get("name"),
                "types": first.get("types", []),
                "vicinity": first.get("vicinity", ""),
                "description": details
            }

        else:
            place_info = {
                "name": None,
                "types": [],
                "vicinity": "",
                "description": ""
            }
    except Exception as e:
        print("API request error:", e)
        place_info = {
            "name": None,
            "types": [],
            "vicinity": "",
            "description": ""
        }
    
    cache_dict[lat_lon_key] = place_info
    return place_info


def label_cluster_types(centroids, api_key):
    cache_dict = load_cache(CACHE_FILE)
    place_info_list = []

    for i, row in centroids.iterrows():
        lat, lon = row["latitude"], row["longitude"]
        place_info = query_google_places(lat, lon, api_key, cache_dict)
        
        place_info_list.append({
            "cluster": row["cluster"],
            "latitude": lat,
            "longitude": lon,
            "name": place_info["name"],
            "types": place_info["types"],
            "vicinity": place_info["vicinity"],
            "description": place_info.get("description", "")
        })
    
    save_cache(cache_dict, CACHE_FILE)

    place_df = pd.DataFrame(place_info_list)
    place_df["semantic_label"] = place_df["types"].apply(simple_label)
    
    return place_df


def simple_label(types_list):
    if not types_list:
        return "Other"
    types_set = set([t.lower() for t in types_list])

    if "university" in types_set or "school" in types_set:
        return "School/University"
    if any(x in types_set for x in ["restaurant","food","bar","cafe"]):
        return "Restaurant/Cafe/Bar"
    if "gym" in types_set:
        return "Gym"
    if any(x in types_set for x in ["lodging","hotel"]):
        return "Hotel"
    if any(x in types_set for x in ["store","shopping_mall"]):
        return "Store/Shopping"
    return "Other"

def get_cluster_name(row):
    try:
        cluster_id = int(row["cluster"])
    except:
        cluster_id = row["cluster"]

    if USE_MANUAL_LABELS and cluster_id in MANUAL_LABELS:
        return MANUAL_LABELS[cluster_id]

    name = row.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    
    return f"Cluster {cluster_id}"


def plot_clusters_on_map(df_clusters, center_lat=38.034, center_lon=-78.505, zoom=17):
    output_html="docs/uva_clusters.html"
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom,
                   tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                   attr='Esri', name='Esri Satellite', overlay=False, control=True)

    for i, row in df_clusters.iterrows():
        name = get_cluster_name(row)
        popup_text = f"{name}<br>Label: {row.get('semantic_label','')}"
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=popup_text
        ).add_to(m)

        time_spent = row.get("duration_minutes", 0)
        radius = math.log(time_spent + 1) * 20

        folium.Circle(
            location=[row["latitude"], row["longitude"]],
            radius=radius,
            color='blue',
            fill=True,
            fill_opacity=0.3
        ).add_to(m)

    folium.LayerControl().add_to(m)
    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    m.save(output_html)
    print("Cluster Folium map saved to /docs/uva_clusters.html.")


def compute_time_spent(df):
    df_sorted = df.sort_values(["cluster", "timestamp"]).reset_index(drop=True)
    records = []
    current_cluster = None
    start_time = None
    end_time = None
    
    gap_threshold = 30 # minutes

    for i, row in df_sorted.iterrows():
        c = row["cluster"]
        t = row["timestamp"]

        if c == -1:
            continue

        if c != current_cluster:
            if current_cluster is not None:
                duration = (end_time - start_time).total_seconds() / 60.0
                records.append((current_cluster, start_time, end_time, duration))
            current_cluster = c
            start_time = t
            end_time = t
        else:
            gap = (t - end_time).total_seconds() / 60.0
            if gap > gap_threshold:
                duration = (end_time - start_time).total_seconds() / 60.0
                records.append((current_cluster, start_time, end_time, duration))
                start_time = t
                end_time = t
            else:
                end_time = t

    if current_cluster is not None:
        duration = (end_time - start_time).total_seconds() / 60.0
        records.append((current_cluster, start_time, end_time, duration))

    df_visits = pd.DataFrame(records, columns=["cluster", "start_time", "end_time", "duration_minutes"])
    return df_visits


from wordcloud import STOPWORDS

def create_wordcloud(place_df, output_name="place_wordcloud.png"):
    all_texts = []
    custom_stops = {"road", "street", "avenue", "drive", "place", "route", "way", "charlottesville", "richmond"}
    stopwords = STOPWORDS.union(custom_stops)

    for _, row in place_df.iterrows():
        weight = max(int(row.get("duration_minutes", 0) / 100), 1)

        tokens = []
        if row.get("name"):
            tokens += re.findall(r"\w+", row["name"])
        if row.get("vicinity"):
            tokens += re.findall(r"\w+", row["vicinity"])
        if row.get("types"):
            tokens += row["types"]

        all_texts.extend([t.lower() for t in tokens if len(t) > 3 and not t.isdigit()])

        if row.get("description"):
            description_words = re.findall(r"\w+", row["description"])
            for _ in range(weight):
                all_texts.extend([t.lower() for t in description_words if len(t) > 3 and not t.isdigit()])

    text_blob = " ".join(all_texts)

    wc = WordCloud(
        width=1000,
        height=500,
        background_color="white",
        stopwords=stopwords,
        max_words=1000,
        collocations=False
    ).generate(text_blob)

    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud of Google Places Data")
    plt.tight_layout()
    plt.savefig(output_name)
    plt.show()


def plot_time_spent_bar_chart(place_df):
    df_sorted = place_df.sort_values("duration_minutes", ascending=False)
    df_top10 = df_sorted.iloc[:10].copy()
    
    df_top10["label"] = df_top10.apply(get_cluster_name, axis=1)

    plt.figure(figsize=(10, 6))
    plt.bar(df_top10["label"], df_top10["duration_minutes"])
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Location (Top 10 by Time Spent)")
    plt.ylabel("Total Time (minutes)")
    plt.title("Top 10 Clusters by Time Spent")
    plt.tight_layout()
    plt.show()


def plot_split_clusters(df, centroids):
    import contextily as ctx
    import geopandas as gpd
    from shapely.geometry import Point

    for label, center in [("Richmond Area Clusters", RICHMOND_CENTER), ("UVA Area Clusters", UVA_CENTER)]:
        fig, ax = plt.subplots(figsize=(8, 8))
        df_area = df[(df["latitude"] > center[0]-0.03) & (df["latitude"] < center[0]+0.03) &
                     (df["longitude"] > center[1]-0.03) & (df["longitude"] < center[1]+0.03)]
        gdf = gpd.GeoDataFrame(df_area, geometry=gpd.points_from_xy(df_area.longitude, df_area.latitude), crs="EPSG:4326").to_crs(epsg=3857)
        gdf.plot(ax=ax, markersize=1, label="All Points")

        centroid_area = centroids[(centroids["latitude"] > center[0]-0.03) & (centroids["latitude"] < center[0]+0.03) &
                                  (centroids["longitude"] > center[1]-0.03) & (centroids["longitude"] < center[1]+0.03)]
        gdf_c = gpd.GeoDataFrame(centroid_area, geometry=gpd.points_from_xy(centroid_area.longitude, centroid_area.latitude), crs="EPSG:4326").to_crs(epsg=3857)
        gdf_c.plot(ax=ax, marker="X", color="red", markersize=50, label="Centroids")

        ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)
        ax.set_title(label)
        plt.legend()
        plt.tight_layout()
        plt.show()


def plot_time_weighted_clusters(df, centroids, df_visits):
    import contextily as ctx
    import geopandas as gpd
    from shapely.geometry import Point

    time_spent = df_visits.groupby("cluster")["duration_minutes"].sum().reset_index()
    centroids_merged = pd.merge(centroids, time_spent, on="cluster", how="left").fillna(0)

    for label, center in [("Richmond Time-Weighted Clusters", RICHMOND_CENTER), ("UVA Time-Weighted Clusters", UVA_CENTER)]:
        fig, ax = plt.subplots(figsize=(8, 8))
        df_area = df[(df["latitude"] > center[0]-0.03) & (df["latitude"] < center[0]+0.03) &
                     (df["longitude"] > center[1]-0.03) & (df["longitude"] < center[1]+0.03)]
        gdf = gpd.GeoDataFrame(df_area, geometry=gpd.points_from_xy(df_area.longitude, df_area.latitude), crs="EPSG:4326").to_crs(epsg=3857)
        gdf.plot(ax=ax, markersize=1, alpha=1.0, color='blue', label="All Points")

        c_area = centroids_merged[
            (centroids_merged["latitude"] > center[0]-0.03) & (centroids_merged["latitude"] < center[0]+0.03) &
            (centroids_merged["longitude"] > center[1]-0.03) & (centroids_merged["longitude"] < center[1]+0.03)
        ]
        gdf_c = gpd.GeoDataFrame(c_area, geometry=gpd.points_from_xy(c_area.longitude, c_area.latitude), crs="EPSG:4326").to_crs(epsg=3857)

        sizes = gdf_c["duration_minutes"].clip(10, 300)
        gdf_c.plot(ax=ax, markersize=sizes, alpha=0.5, color='red', label="Time-weighted Clusters")

        ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)
        ax.set_title(label)
        plt.legend()
        plt.tight_layout()
        plt.show()


def plot_temporal_heatmap(df):
    df_copy = df.copy()

    df_copy["timestamp"] = pd.to_datetime(df_copy["timestamp"], utc=True)
    df_copy["day"] = df_copy["timestamp"].dt.day_name()
    df_copy["hour"] = df_copy["timestamp"].dt.hour

    pivoted = df_copy.groupby(["day","hour"]).size().unstack(fill_value=0)
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivoted = pivoted.reindex(day_order)
    plt.figure(figsize=(12,6))
    sns.heatmap(pivoted, cmap="YlGnBu")
    plt.title("Temporal Heatmap: Visits by Day/Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Day of Week")
    plt.tight_layout()
    plt.show()


def print_top_10_places(place_df):
    df_top10 = place_df.sort_values("duration_minutes", ascending=False).head(10)
    
    for idx, row in df_top10.iterrows():
        cluster_id = row["cluster"]
        label = get_cluster_name(row)
        time_spent = row["duration_minutes"]
        types_list = row.get("types", [])
        
        type_str = ", ".join(types_list) if types_list else "N/A"
        
        print(f"Cluster: {cluster_id}")
        print(f"Label: {label}")
        print(f"Time Spent: {time_spent} minutes")
        print(f"Google Place Types: {type_str}")
        print("—"*40)


def main():
    df = load_timeline_json("Timeline.json", days=30)
    print("Loaded points:", len(df))
    if df.empty:
        print("No data points in time frame.")
        return

    df = cluster_points(df, eps=DEFAULT_EPS, min_samples=DEFAULT_MIN_SAMPLES)
    centroids = get_cluster_centroids(df)
    print("Number of clusters:", len(centroids))

    if len(centroids) == 0:
        print("No clusters found. Adjust parameters.")
        return

    if not API_KEY:
        print("No Places API key.")
        return

    place_df = label_cluster_types(centroids, API_KEY)
    print("\n----- PLACE DF -----")
    print(place_df)

    for i, row in place_df.iterrows():
        print(f"\nCluster {row['cluster']} - {row['name']}")
        print(f"Description:\n{row.get('description')}")

    df_visits = compute_time_spent(df)
    df_visits = df_visits[df_visits["duration_minutes"] >= 5]
    time_spent = df_visits.groupby("cluster")["duration_minutes"].sum().reset_index()
    time_spent.sort_values("duration_minutes", ascending=False, inplace=True)
    print("\n----- TIME SPENT BY CLUSTER (minutes) -----")
    print(time_spent)

    plot_split_clusters(df, centroids)
    plot_time_weighted_clusters(df, centroids, df_visits)

    if "duration_minutes" in place_df.columns:
        place_df.drop(columns=["duration_minutes"], inplace=True)
    place_df = pd.merge(place_df, time_spent, on="cluster", how="left")
    place_df["duration_minutes"] = place_df["duration_minutes"].fillna(0)


    plot_clusters_on_map(place_df, UVA_CENTER[0], UVA_CENTER[1], 15)
    plot_time_spent_bar_chart(place_df)
    create_wordcloud(place_df, "place_wordcloud.png")
    plot_temporal_heatmap(df)
    print_top_10_places(place_df)



if __name__ == "__main__":
    main()
