# Post Disaster Assembly Area Adequacy Analysis

This is a graduation study from the Faculty of Engineering at Izmir University of Economics, supervised by Alper Demir. The study is a data analysis study: the core output is a set of findings about disaster preparedness and urban heat in Izmir, and the interactive map below is simply a way to present those findings rather than a live or continuously updating service. The study evaluates whether the official disaster assembly areas in Izmir, Turkey are large enough for the people living near them, and combines this with satellite based environmental data to study the Urban Heat Island effect across the city.

Interactive map (static snapshot of the analysis results): https://izmirriskmap.netlify.app

## What this study does

After an earthquake, people need to gather in open, safe spaces away from buildings during the first hours of the emergency. Turkish disaster management standards (AFAD) set a minimum requirement of 1.5 square meters of assembly area per person, reachable within a 500 metre walking distance during the acute survival phase. This study checks how well Izmir actually meets the 1.5 square meter standard, neighbourhood by neighbourhood, across all 30 districts and over 1,000 neighbourhoods.

The study also looks at a second problem that is connected to the first one. Many parts of Izmir suffer from the Urban Heat Island effect, where dense, built up areas with little vegetation get significantly hotter than the rest of the city. If an assembly area sits in one of these hot zones, it may be structurally safe but still uncomfortable or even dangerous during a summer disaster. The study identifies these overlapping risk areas and proposes that new green spaces in Izmir could serve two purposes at once: cooling the city down and acting as emergency gathering points.

This dual purpose green space idea is the central proposal of the environmental side of the study. It refers to placing new green areas in neighbourhoods that both lack enough assembly area and show high thermal stress, so that a single investment addresses disaster preparedness and urban heat reduction at the same time.

## Sustainable Development Goals

The study aligns directly with two United Nations Sustainable Development Goals. It supports SDG 11 (Sustainable Cities and Communities) by assessing the capacity and accessibility of post disaster assembly areas, and SDG 13 (Climate Action) by using satellite data to monitor the Urban Heat Island effect and propose green space as a mitigation measure.

## Data sources

The study pulls together data from several public sources:

- Population and neighbourhood growth rates from the Turkish Statistical Institute (TUIK)
- Assembly area locations and sizes from Izmir Metropolitan Municipality (IBB) open data
- Neighbourhood and district boundaries from OpenStreetMap
- Land Surface Temperature (LST) from Landsat 8 satellite imagery
- Vegetation index (NDVI) and built up index (NDBI) from Sentinel-2 satellite imagery

All satellite data was processed through Google Earth Engine. No commercial or paid data sources were used.

## How the analysis works

The analysis is built as a local data pipeline in Python rather than a live web service or production software, since the underlying data does not change in real time and the goal is analytical, not operational. The pipeline runs in four stages:

1. Raw demographic, spatial, and satellite data is loaded and cleaned.
2. A custom text matching algorithm links population records to assembly area records, since neighbourhood names are often written differently across different government datasets. Turkish characters are standardized and a fuzzy matching step (using a 0.82 similarity threshold) catches small spelling differences while avoiding incorrect matches.
3. For each neighbourhood, the pipeline calculates the existing assembly area per person, flags whether it meets the 1.5 square meter standard, and projects population for 2026 and 2027 using district level TUIK growth rates to see which neighbourhoods may become unsafe in the near future.
4. The results are brought together into a single interactive map for presentation.

Neighbourhoods that could not be reliably matched across all sources are kept as a separate "no data" category and are never treated as adequate. Their populations are counted as potentially at risk, which keeps the district level risk estimates conservative rather than optimistic.

Machine learning models (Random Forest, XGBoost, SVM, and KNN) were also tested to see if assembly area sizes could be predicted from population and environmental data. A simple linear regression of population against existing assembly area returned an R² of only 18.1 percent, so the approach was changed from predicting exact square meters (regression) to predicting size categories (a five tier classification), which reached around 51 percent accuracy. Both results point to the same conclusion: existing assembly areas in Izmir are not strongly explained by population density, which suggests that many were placed based on available land rather than a clear capacity plan.

## The interactive map

The map is the visualization layer of the analysis: the final processed dataset is rendered with Leaflet.js and exported as a single standalone HTML file, so the results can be opened in any browser without installing anything.

At the district level, the map shows a heat map style overview of vulnerability across Izmir. Clicking into a district zooms into the neighbourhood level, where each neighbourhood is colored green if it meets the 1.5 square meter standard, red if it does not, or grey if there was not enough data to evaluate it.

A search panel lets the user look up any neighbourhood by name. Selecting one shows its population, total assembly area, area per person, safety status, and environmental indicators (temperature, vegetation index, and built up index), each with a simple color coded label so the values are easy to read at a glance.

The panel also includes a population projection table. It shows the current numbers side by side with projected 2026 and 2027 figures, so it is possible to see whether a neighbourhood is expected to stay safe, become unsafe, or recover as its population changes.

A separate layer shows all 2,383 registered assembly areas in Izmir as points on the map, sized according to their actual area in square meters and color coded by size category. Clicking a point shows its address and exact size.

## Key findings

Assembly area adequacy varies sharply across the city. Dense, central districts such as Karabaglar, Konak, Buca, and Bornova have the highest number of neighbourhoods below the 1.5 square meter threshold. Karabaglar alone has 37 of its 48 neighbourhoods below the standard. When the gap is converted into a planning quantity, Buca needs the most additional assembly area at roughly 320,566 square meters, followed by Karabaglar at about 314,040 and Bornova at about 233,668. Less populated, greener districts on the outskirts of the city are generally well within the safety standard.

A worst case vulnerability measure, which combines the at risk population with the no data population and compares it to the total district population, reaches about 85.5 percent in Karabaglar and 83.6 percent in Konak. This is a conservative upper bound rather than a confirmed exposure value, but it shows how concentrated the risk is in the dense urban core.

The satellite data showed that 283 of the 1,074 neighbourhoods (about a quarter) qualify as thermal hotspots, meaning their surface temperature is above 36.4°C, which is 2 degrees Celsius over the city average. The analysis also confirmed a measurable link between vegetation and temperature: a 0.1 increase in NDVI corresponds to roughly a 1.2°C drop in surface temperature. This relationship is the empirical basis for the dual purpose green space proposal.

Cigli stands out as the clearest example of where this idea applies. It is the hottest district in the analysis, driven by its concentration of organized industrial zones, its position outside the bay breeze corridor that cools the coastal districts, and its very low green cover. Egekent neighbourhood in Cigli reaches a surface temperature of 41.09°C with an NDVI of only 0.28, making it one of the most heat stressed neighbourhoods in the dataset and a strong candidate for a green space that would cool the area and serve as a gathering point.

Overall, greener neighbourhoods are reliably cooler, but having more green space does not automatically mean a neighbourhood has enough assembly area. This is exactly the gap the study's dual purpose green space idea is meant to address.

## Tools and platforms used for the analysis

Python, Pandas, GeoPandas, scikit-learn, Folium, and Leaflet.js for the data processing and the resulting visualization. Google Earth Engine for satellite indicator extraction. All datasets and tools used are open source or publicly available.

## Team

Cenker Efe Tahan, Ege Sevindi, Efe Sonmez, and Osman Serhan Aydogan, Computer Engineering students at Izmir University of Economics, under the supervision of Alper Demir.
