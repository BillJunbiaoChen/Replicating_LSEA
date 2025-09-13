###############################################
## Project: Endogenous Amenities and Location Sorting
## Purpose: Combine prepared files into panel data
## Author: Milena Almagro & Tomas Dominguez-Iino
###############################################

library(pacman)
pacman::p_load(rstudioapi, tidyverse, broom, data.table, 
               readstata13, readxl,
               lubridate,
               hrbrthemes, viridis, RColorBrewer,ggplot2,
               latex2exp, ggpubr)
options(warn = -1)
setwd(dirname(getwd()))
setwd('../..')
rm(list = setdiff(ls(), "wd"))

##################  BBGA

dt = read.csv(gzfile('data/bbga/bbga.csv','rt'), header=T)
setDT(dt)
dt = dt[year>=2008 & year<=2018,]

#Select all translated amenities
dt = dt[, c('year','wk_code','gsd','sd_code',
            'area_land','hect_green',
            'horeca_per_1000','horeca_locations_o',
            'hotel_total','hotel_locations','hotel_rooms','hotel_rooms_imp','hotel_beds_per_1000','hotel_beds_per_1000_imp','hotel_beds','hotel_beds_imp',
            'culture_per_1000','tourism_offices','tourism_offices_imp',
            'companies_total','stores_total','office_total','office_services','office_shops',
            'food_stores','nonfood_stores','supermarkets',
            'bars','bars_locations','coffee_shops','coffee_shops_imp','canteens',
            'fast_food','restaurants','restaurants_locations',
            'activity_ci','activity_ict','activity_bus_services',
            'education','primary_schools','primary_schools_imp','secondary_schools','secondary_schools_imp',
            'special_ed_schools','special_ed_schools_imp','school_care','school_care_imp',
            'nurseries','nurseries_imp','community_centers','community_centers_imp',
            'sports_amenities','care_facilities','care_facilities_imp')]

#Check data availability (share of wijks with data, by year)
availability = dcast(melt( dt[, lapply(.SD, function(x) round(100*sum(!is.na(x))/length(unique(dt$wk_code)),0)),  by =.(year)], id.vars = "year"), variable ~ year)


#Replace missing values with imputed values where possible
varnames = c('hotel_beds','tourism_offices','primary_schools','secondary_schools',
             'special_ed_schools','school_care','nurseries',
             'care_facilities')
for (varname in varnames){
  dt[is.na(get(varname)), noquote(varname) := get(paste0(varname, "_imp"))]
}

#Select final variables
dt[, education_services := education - primary_schools - secondary_schools - special_ed_schools]
dt[, tourism_offices_net_hotels := tourism_offices - hotel_locations]
dt = dt[, c('year','wk_code','gsd','sd_code',
            'tourism_offices','tourism_offices_net_hotels','bars_locations','restaurants_locations','food_stores','nonfood_stores',
            'education_services','school_care','nurseries','hotel_locations',
            'hotel_beds')]
dtb = dt



print(dtb)


##################  Other files to merge

#Airbnb listings
dta = read.csv(gzfile('data/airbnb/listings_by_wk.csv','rt'), header=T)
setDT(dta)

#Tourism
dtt = read_excel('data/tourism/tourism_figures.xlsx', sheet=1)
setDT(dtt)

#Merge
dt <- merge(dtb, dta, by=c('year','wk_code'), all.x = TRUE)
dt = merge(dt, dtt[, .(year, bed_occupancy_rate)], by=c('year'), all.x = TRUE)
dt = dt[year <= 2018,]

#Tourist population
dt[, pop_tourists_airbnb := airbnb_pop]
dt[, pop_tourists_airbnb_commercial := airbnb_pop_commercial]
dt[, pop_tourists_hotels := (bed_occupancy_rate/100)*hotel_beds]
dt[, pop_tourists_total := pop_tourists_hotels + pop_tourists_airbnb]


################## Final variables
dt = dt[, c('year','wk_code','gsd','sd_code',
            'tourism_offices','tourism_offices_net_hotels','bars_locations','restaurants_locations','food_stores','nonfood_stores',
            'education_services','nurseries','hotel_locations','hotel_beds','airbnb_beds','commercial',
            'pop_tourists_airbnb_commercial','pop_tourists_airbnb','pop_tourists_hotels','pop_tourists_total')]

setnames(dt, old = c('gsd') , new = c('gb_code'))
availability = dcast(melt( dt[, lapply(.SD, function(x) round(100*sum(!is.na(x))/length(unique(dt$wk_code)),0)),  by =.(year)], id.vars = "year"), variable ~ year)
write.csv(dt, paste0("data/constructed/neighborhood_panel.csv"), row.names = FALSE)

#Aggregate to gebied level
dt[, (c('wk_code')) := NULL] #drop unused variables
dt = dt[, lapply(.SD, sum, na.rm=TRUE), by = .(year, gb_code, sd_code)]
write.csv(dt, paste0("data/constructed/neighborhood_panel_gebied.csv"), row.names = FALSE)


################## Sanity checks
#Test matching to gebied level
dt_3 = dt[year==2017, list(year,gb_code,pop_tourists_airbnb,pop_tourists_airbnb_commercial)]
dt_2 = read.csv(gzfile('data/airbnb/listings_by_gb.csv','rt'), header=T)
setDT(dt_2)
dt_2[, pop_tourists_airbnb := airbnb_pop]
dt_2[, pop_tourists_airbnb_commercial := airbnb_pop_commercial]
dt_2 = dt_2[year==2017 & gbd_code!=0, list(year,gbd_code,pop_tourists_airbnb,pop_tourists_airbnb_commercial)]

print('3_construct_neighborhoods_panel complete')

