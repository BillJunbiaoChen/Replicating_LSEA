###############################################
## Project: Endogenous Amenities and Location Sorting
## Purpose: Clean google trends data
## Author: Milena Almagro & Tomas Dominguez-Iino
###############################################
library(pacman)
pacman::p_load(tidyverse, data.table, zoo,
               raster, rgdal, maptools, broom, rgeos)

options(warn = -1)
setwd(dirname(getwd()))
setwd('../..')
rm(list = setdiff(ls(), "wd"))

###############################################

df = fread("data/googletrends/raw/multiTimeline.csv")
setnames(df, c('month','google_index'))

df[, month := as.Date(as.yearmon(month))]
df[, year := year(month)]
df[, google_index := parse_number(google_index)]

annual = df[ , .(mean=mean(google_index)), by=year][year <= 2019]
write.csv(annual, "data/googletrends/google_trends_annual.csv", row.names = FALSE)

print('0_clean_google_trends complete')


