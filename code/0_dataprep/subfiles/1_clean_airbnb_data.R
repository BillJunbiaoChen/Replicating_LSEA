###############################################
## Project: Endogenous Amenities and Location Sorting
## Purpose: Link geocoordinates to shapefiles
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

##########  Airbnb listing coordinates

#Append listings

column_list = c('id','last_scraped','host_id','host_since','host_response_rate','host_acceptance_rate',
                'host_is_superhost','host_total_listings_count','host_listings_count','calculated_host_listings_count',
                'neighbourhood','neighbourhood_cleansed','latitude','longitude','property_type','room_type','accommodates',
                'bathrooms','bedrooms','beds','bed_type','price','weekly_price','monthly_price','minimum_nights',
                'maximum_nights','calendar_updated', 'has_availability', 'availability_30','availability_60',
                'availability_90','availability_365','number_of_reviews','first_review','last_review',
                'review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin',
                'review_scores_communication','review_scores_location','review_scores_value','instant_bookable', 'reviews_per_month')

dfa = fread(paste0("data/airbnb/raw/listings (0).csv.gz"),  select = column_list)
dfa$last_scraped = as.Date(as.character(dfa$last_scraped), format="%Y-%m-%d")

for ( n in c(1:48) ){
  df = fread(paste0("data/airbnb/raw/listings (", n, ").csv.gz"),  select = column_list)
  df[, last_scraped := as.Date(as.character(last_scraped), format="%Y-%m-%d")]
  dfa = rbind(dfa, df)
}

#Link listings to wijks using latitude and longitude

points = st_as_sf(dfa, coords = c("longitude", "latitude"), crs = 4326, remove=FALSE)
shapefile = st_read("data/shapefiles/wijk.shp")
df = st_join(shapefile, points, left=FALSE)
rm(points, shapefile, dfa)


#Formatting and defining new variables

setDT(df)
df[, c("gm_code","gm_name","water","d_trainst","d_trainhub","geometry"):= NULL]

#Codes
df[room_type == 'Entire home/apt', c('room_type_code', 'entire_home') := list(1,1)]
df[room_type == 'Private room', c('room_type_code', 'private_room') := list(2,1)]
df[room_type == 'Shared room', c('room_type_code', 'shared_room') := list(3,1)]
df[ , (c('entire_home', 'private_room','shared_room')) := lapply(.SD, nafill, fill=0), .SDcols = c('entire_home', 'private_room','shared_room')]

#Strings to numeric
df[, host_response_rate := parse_number(host_response_rate)/100]
df[, host_acceptance_rate := parse_number(host_acceptance_rate)/100]
df[, price := parse_number(price)]
df[, weekly_price := parse_number(weekly_price)]
df[, monthly_price := parse_number(monthly_price)]
df[, host_id := as.character(host_id)]

#Date formatting
df[, host_since := as.Date(as.character(host_since), format="%Y-%m-%d")]
df[, first_review := as.Date(as.character(first_review), format="%Y-%m-%d")]
df[, last_review := as.Date(as.character(last_review), format="%Y-%m-%d")]
df[, host_since_year := year(host_since)]
df[, scrape_year := year(last_scraped)]
df[, scrape_month := month(last_scraped)]
df[, first_scraped := min(last_scraped), by = id]

#Days since last calendar update
df[, calendar_updated_num := parse_number(calendar_updated)]
df[calendar_updated == 'today', calendar_updated_num := 0]
df[calendar_updated == 'yesterday', calendar_updated_num := 1]
df[calendar_updated == 'a week ago', calendar_updated_num := 1]
df[calendar_updated == 'a month ago', calendar_updated_num := 1]
df[str_detect(calendar_updated, 'day'), calendar_updated_period := 1]
df[str_detect(calendar_updated, 'week'), calendar_updated_period := 7]
df[str_detect(calendar_updated, 'month'), calendar_updated_period := 30]
df[calendar_updated != 'never', days_since_last_calendar_update := calendar_updated_num*calendar_updated_period]
df[calendar_updated == 'never', days_since_last_calendar_update := last_scraped - first_scraped]
df[, days_since_last_calendar_update := parse_number(as.character(days_since_last_calendar_update))]

#Days since last review
df[, days_since_last_review := last_scraped - last_review]
df[is.na(last_review), days_since_last_review := last_scraped - first_scraped]
df[, days_since_last_review := parse_number(as.character(days_since_last_review))]

#Days since last activity
df[, days_since_last_activity := min(days_since_last_review, days_since_last_calendar_update), by=1:nrow(df)]
df[, first_activity := min(first_review, first_scraped), by=1:nrow(df)]
df[is.na(first_review), first_activity := first_scraped]
df[, last_activity := last_scraped - days_since_last_activity]
df[, first_activity_year := year(first_activity)]
df[, last_activity_year := year(last_activity)]

#Drop unused columns and order columns
df[, c("calendar_updated_num","calendar_updated_period"):= NULL]
setcolorder(df, c("id","host_id","scrape_year","scrape_month","last_scraped","first_scraped",
                  "wk_code","wk_name","neighbourhood","neighbourhood_cleansed"))
setorderv(df, c("id","last_scraped"))

#Drop duplicate id-wk_code pairs (due to spatial matching fuzzyness)
dt_id_wk_1 = unique(df, by = c("id","wk_code"))[,.(id,wk_code)]
dt_id_wk_2 = unique(dt_id_wk_1, by = c("id"))
df = merge(df, dt_id_wk_2, by=c('id', 'wk_code'), all.y=TRUE)

# Save data
write.csv(df, gzfile(paste0("data/airbnb/listings.csv.gz")), row.names = FALSE)

print('1_clean_airbnb_data complete')


