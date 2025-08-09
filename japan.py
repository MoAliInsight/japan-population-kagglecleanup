import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static

prefecture_coords = {
    'Aichi-ken': [35.1802, 136.9066], 'Akita-ken': [39.7199, 140.1024], 'Aomori-ken': [40.8244, 140.7400],
    'Chiba-ken': [35.6073, 140.1062], 'Ehime-ken': [33.8416, 132.7657], 'Fukui-ken': [36.0652, 136.2216],
    'Fukuoka-ken': [33.5903, 130.4017], 'Fukushima-ken': [37.7500, 140.4675], 'Gifu-ken': [35.3912, 136.7222],
    'Gunma-ken': [36.3907, 139.0604], 'Hiroshima-ken': [34.3853, 132.4553], 'Hokkaido': [43.0646, 141.3469],
    'Hyogo-ken': [34.6901, 135.1956], 'Ibaraki-ken': [36.3418, 140.4468], 'Ishikawa-ken': [36.5947, 136.6256],
    'Iwate-ken': [39.7036, 141.1527], 'Kagawa-ken': [34.3401, 134.0433], 'Kagoshima-ken': [31.5602, 130.5581],
    'Kanagawa-ken': [35.4475, 139.6425], 'Kochi-ken': [33.5597, 133.5311], 'Kumamoto-ken': [32.7897, 130.7417],
    'Kyoto-fu': [35.0211, 135.7556], 'Mie-ken': [34.7303, 136.5086], 'Miyagi-ken': [38.2688, 140.8721],
    'Miyazaki-ken': [31.9111, 131.4239], 'Nagano-ken': [36.6513, 138.1812], 'Nagasaki-ken': [32.7448, 129.8736],
    'Nara-ken': [34.6853, 135.8327], 'Niigata-ken': [37.9022, 139.0236], 'Oita-ken': [33.2382, 131.6126],
    'Okayama-ken': [34.6617, 133.9350], 'Okinawa-ken': [26.2124, 127.6809], 'Osaka-fu': [34.6864, 135.5200],
    'Saga-ken': [33.2494, 130.2988], 'Saitama-ken': [35.8569, 139.6489], 'Shiga-ken': [35.0045, 135.8686],
    'Shimane-ken': [35.4723, 133.0505], 'Shizuoka-ken': [34.9756, 138.3827], 'Tochigi-ken': [36.5657, 139.8836],
    'Tokushima-ken': [34.0657, 134.5593], 'Tokyo-to': [35.6895, 139.6917], 'Tottori-ken': [35.5039, 134.2382],
    'Toyama-ken': [36.6953, 137.2113], 'Wakayama-ken': [34.2260, 135.1675], 'Yamagata-ken': [38.2404, 140.3633],
    'Yamaguchi-ken': [34.1859, 131.4714], 'Yamanashi-ken': [35.6639, 138.5683]
}

@st.cache_data
def load_data():
    df = pd.read_csv('data/japan_population_data.csv')
    df_clean = df.dropna()
    df_clean = df_clean.rename(columns={
        'estimated_area': 'Area (km²)',
        'prefecture': 'Prefecture',
        'year': 'Year',
        'population': 'Population',
        'capital': 'Capital',
        'region': 'Region',
        'island': 'Island'
    })
    df_clean['Year'] = df_clean['Year'].round().astype(int)
    df_clean['Latitude'] = df_clean['Prefecture'].map(lambda x: prefecture_coords.get(x, [None, None])[0])
    df_clean['Longitude'] = df_clean['Prefecture'].map(lambda x: prefecture_coords.get(x, [None, None])[1])
    df_clean['Population Density'] = df_clean['Population'] / df_clean['Area (km²)']
    df_clean['Area per Person'] = df_clean['Area (km²)'] / df_clean['Population']
    df_clean['Population Share (%)'] = df_clean.groupby('Year')['Population'].transform(lambda x: x / x.sum() * 100)
    df_clean['Population Change'] = df_clean.groupby('Prefecture')['Population'].diff()
    df_clean['Population Change (%)'] = df_clean.groupby('Prefecture')['Population'].pct_change() * 100
    df_clean['Density Change'] = df_clean.groupby('Prefecture')['Population Density'].diff()
    df_clean['Population Rank'] = df_clean.groupby('Year')['Population'].rank(ascending=False, method='dense')
    df_clean['Density Rank'] = df_clean.groupby('Year')['Population Density'].rank(ascending=False, method='dense')
    df_clean['Region Total Population'] = df_clean.groupby(['Region', 'Year'])['Population'].transform('sum')
    df_clean[['Population Change', 'Population Change (%)', 'Density Change']] = df_clean[['Population Change', 'Population Change (%)', 'Density Change']].fillna(0)
    if 'population_share_year' in df_clean.columns:
        df_clean = df_clean.drop('population_share_year', axis=1)
    return df_clean

df_clean = load_data()

st.title("Japan Population Data Explorer")
st.markdown("""
Explore population trends across Japanese prefectures (1872–2015).
Use the filters to select specific prefectures, regions, islands, or years.
""")

st.sidebar.header("Filter Data")
prefectures = df_clean['Prefecture'].unique()
selected_prefectures = st.sidebar.multiselect("Select Prefectures", prefectures, default=['Tokyo-to', 'Aichi-ken'])
islands = df_clean['Island'].unique()
selected_islands = st.sidebar.multiselect("Select Islands", islands, default=islands)
min_year, max_year = int(df_clean['Year'].min()), int(df_clean['Year'].max())
selected_years = st.sidebar.slider("Select Year Range", min_year, max_year, (min_year, max_year))

filtered_df = df_clean[
    (df_clean['Prefecture'].isin(selected_prefectures) | (len(selected_prefectures) == 0)) &
    (df_clean['Island'].isin(selected_islands) | (len(selected_islands) == 0)) &
    (df_clean['Year'].between(selected_years[0], selected_years[1]))
]

st.write(f"Filtered data: {len(filtered_df)} rows")
st.dataframe(filtered_df)

st.header("Visualizations")
if not filtered_df.empty:
    st.subheader("Population Density Heatmap")
    folium_map = folium.Map(location=[36.0, 138.0], zoom_start=6, tiles="CartoDB Positron")
    heat_data = filtered_df[['Latitude', 'Longitude', 'Population Density']].dropna()
    heat_data = [[row['Latitude'], row['Longitude'], row['Population Density']] for _, row in heat_data.iterrows()]
    max_density = filtered_df['Population Density'].max() if not filtered_df['Population Density'].empty else 1
    heat_data = [[lat, lon, weight / max_density] for lat, lon, weight in heat_data]
    if heat_data:
        HeatMap(heat_data, radius=15, blur=20, max_zoom=10).add_to(folium_map)
        folium_static(folium_map)
    else:
        st.write("No data available for the heatmap after filtering.")

    st.subheader("Population Share Over Time")
    fig, ax = plt.subplots()
    sns.lineplot(data=filtered_df, x='Year', y='Population Share (%)', hue='Prefecture', style='Prefecture', ax=ax)
    plt.title('Population Share by Prefecture')
    plt.xlabel('Year')
    plt.ylabel('Population Share (%)')
    st.pyplot(fig)

    st.subheader("Population Change Over Time")
    fig, ax = plt.subplots()
    sns.lineplot(data=filtered_df, x='Year', y='Population Change (%)', hue='Prefecture', style='Prefecture', ax=ax)
    plt.title('Population Change (%) by Prefecture')
    plt.xlabel('Year')
    plt.ylabel('Population Change (%)')
    st.pyplot(fig)

else:
    st.write("No data to plot after filtering.")

st.header("Summary Statistics")
if not filtered_df.empty:
    st.subheader("Top 5 Prefectures by Average Population")
    prefecture_summary = filtered_df.groupby('Prefecture')[['Population', 'Population Density']].mean().round(2)
    st.dataframe(prefecture_summary.sort_values(by='Population', ascending=False).head(5))

    st.subheader("Regional Population Trends")
    region_trend = filtered_df.groupby(['Region', 'Year'])['Population'].sum().reset_index()
    st.dataframe(region_trend.head())
else:
    st.write("No data to display after filtering.")

st.subheader("Download Filtered Data")
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV", csv, "filtered_japan_population.csv", "text/csv")