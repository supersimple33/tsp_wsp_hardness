import requests
import numpy as np
import pandas as pd

def get_mcdonalds_locations():
    """Fetches McDonald's locations in SF and Oakland using OpenStreetMap (Overpass API)."""
    print("Querying OpenStreetMap for McDonald's locations...")
    overpass_url = "http://overpass-api.de/api/interpreter"
    
    # Using exact match "=" instead of regex "~" is generally faster and safer
    overpass_query = """
    [out:json][timeout:25];
    area["name"="San Francisco"]["admin_level"="8"]->.sf;
    area["name"="Oakland"]["admin_level"="8"]->.oak;
    (
      node["brand"="McDonald's"](area.sf);
      way["brand"="McDonald's"](area.sf);
      node["brand"="McDonald's"](area.oak);
      way["brand"="McDonald's"](area.oak);
    );
    out center;
    """
    
    # Overpass requires a custom User-Agent header, otherwise it often returns a 403 Forbidden HTML page.
    headers = {
        'User-Agent': 'SFOakland-McDonalds-Distance-Matrix/1.0 (script-testing)'
    }
    
    response = requests.post(overpass_url, data={'data': overpass_query}, headers=headers)
    
    # Catch non-200 responses and print the actual text to see the API's complaint
    if response.status_code != 200:
        raise Exception(f"Overpass API Error (Status {response.status_code}): {response.text[:500]}")
        
    # Catch the JSON error specifically just in case
    try:
        data = response.json()
    except requests.exceptions.JSONDecodeError:
        raise Exception(f"Expected JSON but got something else. Server returned: {response.text[:500]}")

    elements = data.get('elements', [])
    
    locations = []
    for el in elements:
        lat = el.get('lat') or el.get('center', {}).get('lat')
        lon = el.get('lon') or el.get('center', {}).get('lon')
        name = el.get('tags', {}).get('name', "McDonald's")
        
        if lat and lon:
            locations.append({
                "id": el['id'], 
                "name": name, 
                "lat": lat, 
                "lon": lon
            })
            
    df = pd.DataFrame(locations)
    return df.drop_duplicates(subset=['lat', 'lon']).reset_index(drop=True)

def get_osrm_distance_matrix(coords):
    """Gets driving distances (in meters) using the public OSRM API."""
    print("Calculating road distances via OSRM...")
    
    # OSRM expects coordinates formatted as: lon1,lat1;lon2,lat2;...
    coord_string = ";".join([f"{lon},{lat}" for lon, lat in coords])
    
    # annotations=distance returns the distance matrix instead of travel times
    url = f"http://router.project-osrm.org/table/v1/driving/{coord_string}?annotations=distance"
    
    response = requests.get(url)
    data = response.json()
    
    if data.get('code') != 'Ok':
        raise Exception(f"OSRM API Error: {data.get('message', 'Unknown Error')}")
        
    # Return the matrix as a NumPy array
    return np.array(data['distances'])

def main():
    # 1. Get the locations
    df_locs = get_mcdonalds_locations()
    print(f"Found {len(df_locs)} locations.")
    
    # 2. Extract coordinate pairs
    coords = list(zip(df_locs['lon'], df_locs['lat']))
    
    # 3. Fetch the raw road distance matrix (in meters)
    raw_matrix = get_osrm_distance_matrix(coords)
    
    # 4. Convert meters to miles
    raw_matrix_miles = raw_matrix * 0.000621371
    
    # 5. Make it symmetric by averaging D[i,j] and D[j,i]
    print("Averaging the routes to create a symmetric matrix...")
    symmetric_matrix = (raw_matrix_miles + raw_matrix_miles.T) / 2
    
    # 6. Format into a Pandas DataFrame for easy viewing/export
    store_labels = [f"Store_{i} (Lat:{lat:.3f})" for i, lat in enumerate(df_locs['lat'])]
    df_symmetric = pd.DataFrame(symmetric_matrix, index=store_labels, columns=store_labels)
    
    # Optional: Save to CSV
    df_symmetric.to_csv("mcdonalds_sf_oakland_distances.csv")
    print("\nSaved output to 'mcdonalds_sf_oakland_distances.csv'.")
    print("\nPreview of Symmetric Matrix (in miles):")
    print(df_symmetric.iloc[:5, :5].round(2))

if __name__ == "__main__":
    main()