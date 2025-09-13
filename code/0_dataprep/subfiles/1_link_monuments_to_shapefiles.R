###############################################
## Project: Endogenous Amenities and Location Sorting
## Purpose: Link geocoordinates of historic monuments to shapefiles
## Author: Tomas Dominguez-Iino
###############################################

# Load packages 
library(pacman)
pacman::p_load(rstudioapi, tidyverse, data.table, readxl,
               raster, rgdal, maptools, broom, rgeos, sf)
options(warn = -1)               
setwd(dirname(getwd()))
setwd('../..')
rm(list = setdiff(ls(), "wd"))

#Link monuments to wijks using latitude and longitude

dfm = fread(paste0("data/bbga/raw/MONUMENTEN.csv"), select = c("OBJECTNUMMER","LNG","LAT"))
points = st_as_sf(dfm, coords = c("LNG", "LAT"), crs = 4326)
shapefile = st_read("data/shapefiles/wijk.shp")
df = st_join(shapefile, points)
rm(points, shapefile, dfm)

setDT(df)
df[, c("gm_code","gm_name","water","d_trainst","d_trainhub","geometry","OBJECTNUMMER"):= NULL]

monument_count = df[, .(.N), by = .(wk_code)]
setnames(monument_count, old = c('N'), new = c('monuments'))
write.csv(monument_count,"data/bbga/monument_count.csv", row.names = FALSE)

print('1_link_monuments_to_shapefile complete')

