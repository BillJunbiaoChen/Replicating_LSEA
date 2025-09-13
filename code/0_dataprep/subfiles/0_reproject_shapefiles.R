###############################################
## Project: Endogenous Amenities and Location Sorting
## Purpose: Reproject shapefiles into appropriate CRS
## Author: Milena Almagro & Tomas Dominguez-Iino
###############################################

library(pacman)
pacman::p_load(tidyverse, data.table, sf, broom)

options(warn = -1)
setwd(dirname(getwd()))
setwd('../..')
rm(list = setdiff(ls(), "wd"))

############################################### Wijk

# Change projection from Dutch system to WGS84 and subset to Amsterdam
shpfile <- st_read(dsn = path.expand("data/shapefiles/raw/"), layer = "wijk_2018_v3")

# Transform the shapefile to WGS84 (EPSG:4326)
shpfile <- st_transform(shpfile, crs = 4326)

# Keep and format variables of interest using dplyr::select()
shpfile <- shpfile %>%
  filter(GM_CODE == "GM0363" & WATER == "NEE") %>%
  dplyr::select(WK_CODE, WK_NAAM, GM_CODE, GM_NAAM, WATER, AF_TREINST, AF_OVERST) %>%
  rename(wk_code = WK_CODE, wk_name = WK_NAAM, gm_code = GM_CODE, gm_name = GM_NAAM, 
         water = WATER, d_trainst = AF_TREINST, d_trainhub = AF_OVERST)

# Convert wk_code to numeric and fix missing values
shpfile$wk_code <- as.character(parse_number(str_sub(shpfile$wk_code, -2)))
shpfile$d_trainst[shpfile$d_trainst == -99999999] <- NA
shpfile$d_trainhub[shpfile$d_trainhub == -99999999] <- NA

# Save shapefile
st_write(shpfile, dsn = "data/shapefiles/wijk.shp", driver = "ESRI Shapefile", delete_layer = TRUE)

############################################### Gebied
# Load the second shapefile and transform the CRS
shpfile <- st_read(dsn = path.expand("data/shapefiles/raw/"), layer = "GEBIEDEN22")
shpfile <- st_transform(shpfile, crs = 4326)

# Keep and format variables of interest
shpfile <- shpfile %>%
  filter(Gebied_cod != "Dxxx") %>%
  dplyr::select(Gebied_cod, Gebied, Stadsdeel_) %>%
  rename(gd_code = Gebied_cod, gd_name = Gebied, sd_code = Stadsdeel_)

# Convert gd_code to numeric
shpfile$gd_code <- as.character(parse_number(str_sub(shpfile$gd_code, -2)))

# Save the transformed shapefile
st_write(shpfile, dsn = "data/shapefiles/gebied.shp", driver = "ESRI Shapefile", delete_layer = TRUE)

print('0_reproject shapefiles complete')
