import json
import os
from datetime import datetime
from dateutil import parser
import pandas as pd
from folium.plugins import HeatMap
import folium
import unicodedata
from dotenv import load_dotenv

load_dotenv()

def clean_coordinate_string(s):
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode()
    s = s.replace("\u00b0", "").strip().replace("\u00b0", "")
    return s

def load_all_timeline_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    points = []
    for segment in data.get("semanticSegments", []):
        if "timelinePath" in segment:
            for entry in segment["timelinePath"]:
                if "point" not in entry or "," not in entry["point"]:
                    continue
                lat_str, lon_str = entry["point"].split(",")
                try:
                    lat = float(clean_coordinate_string(lat_str))
                    lon = float(clean_coordinate_string(lon_str))
                    timestamp = parser.parse(entry["time"])
                    points.append({"latitude": lat, "longitude": lon, "timestamp": timestamp})
                except Exception:
                    continue
    return pd.DataFrame(points)


def generate_heatmap_with_filter(df, output_html="docs/lifetime_heatmap_filtered.html"):
    if df.empty:
        print("No data to plot.")
        return

    df["year"] = df["timestamp"].dt.year.astype(str)
    df_grouped = df.groupby("year")

    center = [df["latitude"].mean(), df["longitude"].mean()]
    m = folium.Map(
        location=center,
        zoom_start=4,
        tiles="https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery"
    )

    layer_control = folium.LayerControl()

    for name, group in df_grouped:
        heat_data = group[["latitude", "longitude"]].values.tolist()
        layer = folium.FeatureGroup(name=name, overlay=True, control=True)
        HeatMap(
            heat_data,
            radius=12,
            blur=18,
            max_zoom=13,
            min_opacity=0.5,
            max_val=1.0
        ).add_to(layer)
        layer.add_to(m)

    all_years_heat_data = df[["latitude", "longitude"]].values.tolist()
    all_years_layer = folium.FeatureGroup(name="All Years", overlay=True, control=True)
    HeatMap(
        all_years_heat_data,
        radius=10, blur=8, max_zoom=13, min_opacity=0.4, max_val=1.0
    ).add_to(all_years_layer)
    all_years_layer.add_to(m)

    layer_control.add_to(m)

    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    m.save(output_html)
    print(f"Saved heatmap with filters to {output_html}")


import reverse_geocoder as rg

def save_unique_cities(df, output_file="docs/visited_cities.txt"):
    print("Extracting  cities from data...")

    coords = list(df[["latitude", "longitude"]].drop_duplicates().itertuples(index=False, name=None))
    results = rg.search(coords, mode='batch')

    cities = {res['name'] for res in results if 'name' in res}

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Visited Cities:\n")
        for city in sorted(cities):
            f.write(f"{city}\n")
        f.write(f"\nTotal unique cities: {len(cities)}\n")

    print(f"Saved {len(cities)} unique cities to {output_file}")


def add_city_markers(df, output_html="docs/lifetime_heatmap_cities.html"):
    print("Generating city map...")
    import reverse_geocoder as rg
    coords = list(df[["latitude", "longitude"]].drop_duplicates().itertuples(index=False, name=None))
    results = rg.search(coords, mode='batch')
    seen = set()
    city_points = []

    for res in results:
        key = (res['name'], res['admin1'], res['cc'])
        if key not in seen:
            seen.add(key)
            city_points.append((float(res['lat']), float(res['lon']), res['name'], res['admin1'], res['cc']))

    center = [df["latitude"].mean(), df["longitude"].mean()]
    m = folium.Map(
        location=center,
        zoom_start=3,
        tiles="https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery"
    )

    for lat, lon, name, admin, cc in city_points:
        folium.CircleMarker(
            location=(lat, lon),
            radius=4,
            color='blue',
            fill=True,
            fill_opacity=0.7,
            popup=f"{name}, {admin}, {cc}"
        ).add_to(m)

    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    m.save(output_html)
    print(f"Saved city map to {output_html}")

def save_country_state_stats(df, output_file="docs/country_state_stats.txt"):
    print("Extracting country and state stats...")
    coords = list(df[["latitude", "longitude"]].drop_duplicates().itertuples(index=False, name=None))
    results = rg.search(coords, mode='batch')

    countries = {}
    us_states = {}

    for res in results:
        cc = res.get("cc")
        admin1 = res.get("admin1")
        if cc:
            countries[cc] = countries.get(cc, 0) + 1
        if cc == "US" and admin1:
            us_states[admin1] = us_states.get(admin1, 0) + 1

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Visited Countries:\n")
        for c, count in sorted(countries.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{c}: {count} locations\n")

        f.write("\nVisited States:\n")
        for s, count in sorted(us_states.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{s}: {count} locations\n")

    print(f"Saved country and state stats to {output_file}")

def main():
    timeline_file = "Timeline.json"

    print("Loading timeline data...")
    df = load_all_timeline_data(timeline_file)
    print(f"Loaded {len(df)} GPS points from timeline.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    print("Generating heatmap with filtering...")
    generate_heatmap_with_filter(df)
    save_unique_cities(df)
    add_city_markers(df)
    save_country_state_stats(df)


if __name__ == "__main__":
    main()
