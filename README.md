# Mapping Global Biodiversity
### Patterns, Trends, and Insights from GBIF Data

This repository presents a complete end-to-end biodiversity analysis pipeline built using a curated dataset from the Global Biodiversity Information Facility (GBIF). The project covers data cleaning, exploratory data analysis, spatial and temporal pattern extraction, diversity assessment, contributor analytics, and the development of an interactive dashboard for advanced visualization.

---

## Repository Structure

```
GBIF-Biodiversity-Analysis/
│
├── data/
│   └── APP_final_dashboard_dataset_8_clean.csv
│
├── notebooks/
│   └── 01_DataCleaning_and_EDA.ipynb
│
├── dashboard/
│   └── APP_final_dashboard_code.py
│
├── report/
│   ├── Final_Story_Report.pdf
│   └── main.tex
│
├── images/
│   └── (all visualizations and report images)
│
├── README.md
└── requirements.txt
```

---

## 1. Data Cleaning and Processing

The dataset was cleaned and prepared using the notebook:

```
notebooks/01_DataCleaning_and_EDA.ipynb
```

### Key Steps Performed:
- Removal of completely empty columns  
- Dropping `infraspecificEpithet` (>90% missing)  
- Parsing and validating `eventDate`, `year`, `month`, `day`  
- Converting numerical fields  
- Validating coordinate ranges  
- Standardizing and stripping text fields  
- Mapping ISO `countryCode` to full country names  
- Eliminating invalid or missing coordinate records  
- Generating a clean dataset for dashboard and analysis  

The cleaned dataset is saved as:

```
data/APP_final_dashboard_dataset_8_clean.csv
```

---

## 2. Exploratory Data Analysis (EDA)

The EDA phase identifies structural patterns, completeness, taxonomic distributions, spatial trends, temporal variations, and contributor activity.

### Visualizations Produced:
- Missing value distribution  
- Refined correlation heatmap  
- Top phyla and top species  
- Taxonomy treemap (Kingdom → Species)  
- Yearly and monthly observation trends  
- Time-animated map (1960–2023)  
- Global scatter map  
- Smoothed density map  
- Shannon diversity index analysis  
- Contributor insights:
  - `identifiedBy`
  - `recordedBy`
  - `rightsHolder`

All graphics are stored in the `images/` directory.

---

## 3. Interactive Dashboard

The dashboard integrates all processed and analyzed components into a streamlined interface.

```
dashboard/APP_final_dashboard_code.py
```

### Features:
- Filters for Country, Kingdom, Year, Month, Class, Order, Family, Genus, Species  
- Dataset overview  
- EDA visual summaries  
- Temporal trends  
- Spatial mapping (density, scatter, animation)  
- Taxonomic exploration  
- Diversity analysis  
- Contributor analytics  

### Run the Dashboard:
```
pip install -r requirements.txt
streamlit run dashboard/APP_final_dashboard_code.py
```

---

## 4. Final Story Report

The complete scientific narrative is documented in:

```
report/main.tex
report/Final_Story_Report.pdf
```

### The report includes:
- Introduction and motivation  
- Dataset characteristics  
- Cleaning methodology  
- EDA findings  
- Spatial and temporal biodiversity insights  
- Diversity and taxonomic patterns  
- Contributor analysis  
- Discussion, conclusion, and future work  

---

## Key Findings

- Biodiversity observations show strong geographic clustering, often aligning with sampling effort.  
- Observation frequency has increased substantially since the early 2000s.  
- Taxonomic representation is uneven across phyla and species.  
- Seasonal variation is visible in monthly trends.  
- Shannon diversity indicates high richness in biologically and sampling-rich regions.  
- Contributor metadata shows a small number of individuals and institutions generate the majority of records.  

---

## Requirements

```
pandas
numpy
plotly
streamlit
matplotlib
country_converter
scikit-learn
```

---

## Usage Summary

1. Clean and process the dataset using the notebook.  
2. Run the dashboard for interactive analysis.  
3. Explore the final report for the full scientific narrative.

---

## License

This project is intended for academic submission and educational use.
