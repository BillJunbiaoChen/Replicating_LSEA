###############################################
## Project: Endogenous Amenities and Location Sorting
## Purpose: Clean bbga data
## Author: Milena Almagro & Tomas Dominguez-Iino
###############################################

library(pacman)
pacman::p_load(rstudioapi, tidyverse, data.table, readxl,
               raster, rgdal, maptools, broom, rgeos, sf, threadr, zoo)

options(warn = -1)
setwd(dirname(getwd()))
print(getwd())
setwd('../..')
rm(list = setdiff(ls(), "wd"))


###############################################

#Import translations
dtr = read_excel('data/bbga/translations.xlsx', sheet='bbga_2018')

original_names = c(tolower(dtr$name))
translated_names = c(tolower(dtr$name_translated))

#Translate BBGA columns
dt <- read.csv('data/bbga/raw/2018_BBGA_0412.csv', skip = 0)
setDT(dt)
names(dt)[names(dt) == 'X...niveau'] <- 'niveau'

setnames(dt, old = colnames(dt), new = tolower(colnames(dt))) #change columns to lowercase first
dt = dt[, ..original_names] #keep only variables that will be translated
setnames(dt, old = original_names, new = translated_names)

#Subset time period, geographic unit of analysis, and drop irrelevant neighborhoods
dt = dt[year>=2008 & year<=2019 & level==4 & sd_code!='Z',]
dt$gbd_code = as.character(parse_number(dt$gbd_code))
`%!in%` <- Negate(`%in%`)
dt = dt[gbd_code %!in% c('10','11','50','51')] #drop Westelijk Havengebied (Westpoort), Bedrijventerrein Sloterdijk (Nieuw-West), Ijburg Oost, Ijburg Zuid
availability = dcast(melt( dt[, lapply(.SD, function(x) round(100*sum(!is.na(x))/length(unique(dt$gbd_code)),0)),  by =.(year)], id.vars = "year"), variable ~ year)


############### Imputations for missing values

varnames = c('area_land','hotel_beds','hotel_rooms','hotel_beds_per_1000','tourism_offices','coffee_shops',
             'community_centers','primary_schools','secondary_schools','special_ed_schools','school_care','nurseries',
             'addresses_total', 'addresses_private_rental','pop_high_skill_p', 'pop_total','average_income',
             'care_facilities')
varnames_fixed = c('area_land')
varnames_variable = c('hotel_beds','hotel_rooms','hotel_beds_per_1000','tourism_offices','coffee_shops',
                      'community_centers','primary_schools','secondary_schools','special_ed_schools','school_care','nurseries',
                      'addresses_total', 'addresses_private_rental','pop_high_skill_p', 'pop_total','average_income',
                      'care_facilities') 
varnames_continuous = c('community_centers')

keys = c('year', 'gbd_code')
varlist = append(keys, varnames)
dti = dt[, ..varlist]
setorder(dti, gbd_code, year)

for (varname in varnames_fixed){
  #For variables that are fixed over time, keep a single value
  dti[, noquote(paste0(varname, "_imp")) := max(get(varname), na.rm=T), by = gbd_code]
  dti[, (c(noquote(varname) )) := NULL] #drop unused variables
}

# Get indices

for (varname in varnames_variable){
  dti[is.finite(get(varname)), has_data :=1]
  dti[!is.finite(get(varname)), has_data :=0]
  dti[, data_points := sum(has_data), by = gbd_code]
  
  #Assign single value to locations with one data point
  dti[data_points==1,  noquote(varname):= max(get(varname), na.rm=T), by = gbd_code]
  #Assign zeros to locations with no data at all
  dti[data_points==0, noquote(varname):= 0, by = gbd_code]
  
  #Interpolate/extrapolate data with more than 1 data point
  dti[data_points>1, noquote(paste0(varname, "_ip")) := round(na_interpolate(get(varname)),0), by = gbd_code] #interpolation for discrete variables
  dti[data_points>1 & varname %in% varnames_continuous, noquote(paste0(varname, "_ip")) := round(na_interpolate(get(varname)),1), by = gbd_code] #interpolation for continous variables
  dti[data_points>1, noquote(paste0(varname, "_ep")) := na_extrapolate(get(paste0(varname, "_ip"))), by = gbd_code] #extrapolation


  #Define imputed values
  dti[, noquote(paste0(varname, "_imp")) := get(varname)]
  dti[is.na(get(paste0(varname,"_imp"))), noquote(paste0(varname, "_imp")) := get(paste0(varname,"_ip"))]
  dti[is.na(get(paste0(varname,"_imp"))), noquote(paste0(varname, "_imp")) := get(paste0(varname,"_ep"))]
  dti[get(paste0(varname,"_imp"))<=0, noquote(paste0(varname, "_imp")) :=0]
  
  #Drop unnecessary variables
  dti[, (c('has_data','data_points', noquote(varname), noquote(paste0(varname, "_ip")), noquote(paste0(varname, "_ep")) )) := NULL] #drop unused variables
}


print(dti)

#Test dataframe for verification
test = merge(dt[,..varlist], dti, by=c('year','gbd_code'), all.x = TRUE)
setcolorder(test, order(colnames(test))) 
setorder(test, gbd_code, year)
test = test %>% relocate(year, gbd_code)

#Final dataframe
dtm = merge(dt, dti, by=c('year','gbd_code'), all.x = TRUE)
setcolorder(dtm, order(colnames(dtm))) 
setorder(dtm, year, gbd_code)
dtm = dtm %>% relocate(year, level, level_name, sd_code, sd_name, gbd_code, gbd_name, gsd)

### Save zipped file
setnames(dtm, old = 'gbd_code', new = 'wk_code')
write.csv(dtm, "data/bbga/bbga.csv", row.names = FALSE)

print('2_clean_bbga complete')


