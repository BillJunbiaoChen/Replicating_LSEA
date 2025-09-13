###############################################
## Project: Endogenous Amenities and Location Sorting
## Purpose: Construct AirBnB supply data from raw files
## Author: Milena Almagro & Tomas Dominguez-Iino
###############################################

library(pacman)
pacman::p_load(rstudioapi, tidyverse, data.table, readxl,
               raster, rgdal, maptools, broom, rgeos, sf,
               plyr)
options(warn = -1)
setwd(dirname(getwd()))
setwd('../..')
rm(list = setdiff(ls(), "wd"))

for (location in c('wk_code','gbd_code') ){
  
  ###############################################
  dt = fread("data/airbnb/listings.csv.gz")
    
  ###### Add gebied codes
  dtg = fread('data/shapefiles/wijk_to_gb.csv')
  setnames(dtg, old = c('wk','gb') , new = c('wk_code','gbd_code'))
  dt = merge(dtg, dt, by=c('wk_code'), all.y=TRUE)
  `%!in%` <- Negate(`%in%`)
  dt = dt[wk_code %!in% c('10','11','50','51')] #drop Westelijk Havengebied (Westpoort), Bedrijventerrein Sloterdijk (Nieuw-West), Ijburg Oost, Ijburg Zuid
  
  ###### Choose location code
  
  if(location == 'wk_code'){
    dt$loc_code = dt$wk_code
  }else{
    dt$loc_code = dt$gbd_code
  }
  
  ###### Definitions of listing types
  dt$high_avail = 1*(dt$availability_365>=180 | dt$availability_90>=45) #offers over X nights per year, or Y nights per quarter
  dt$high_avail_active = 1*(dt$availability_365>=90 & dt$days_since_last_activity<=30) #offers over X nights per year and is active
  dt$ehha = dt$entire_home*dt$high_avail
  
  dt[, high_avail_active_count:= sum(high_avail_active,na.rm=T), .(id, host_id, scrape_year)]
  dt[, number_of_reviews_soy:= min(number_of_reviews,na.rm=T), .(id, host_id, scrape_year)]
  dt[, number_of_reviews_eoy:= max(number_of_reviews,na.rm=T), .(id, host_id, scrape_year)]
  dt[, new_reviews:= number_of_reviews_eoy-number_of_reviews_soy]
  dt[, new_bookings:= new_reviews/0.67]
  dt[, new_nights:= new_bookings*3.9]
  dt[, new_guests:= new_nights*accommodates]
  dt[accommodates>0, price_per_guest:= price/accommodates]
  
  dt$commercial1 = 1*(dt$new_reviews>=60*0.67/3.9)*dt$entire_home*(dt$new_reviews>0) #1. over 60 estimated nights booked per year (we assume 67% of guests leave a review and the average guest books 3.9 nights).
  dt$commercial2 = 1*(dt$availability_365>=90 & dt$instant_bookable=='t')*dt$entire_home*(dt$new_reviews>0)  #2. is actively offering over X nights per year with the instant booking feature turned on.
  dt$commercial3 = 1*(dt$high_avail_active_count>=2)*dt$entire_home*(dt$new_reviews>0)  #3. has actively offered over X nights per year at least twice in the past year.
  dt$commercial4 = 1*(dt$calculated_host_listings_count>1)
  
  dt$commercial_1  = pmax(dt$commercial1, na.rm=T)  #satisfies at least one of conditions 1,2, or 3.
  dt$commercial_12  = pmax(dt$commercial1,dt$commercial2, na.rm=T)  #satisfies at least one of conditions 1,2, or 3.
  dt$commercial_123  = pmax(dt$commercial1,dt$commercial2,dt$commercial3, na.rm=T)  #satisfies at least one of conditions 1,2, or 3.
  dt$commercial_1234  = pmax(dt$commercial1,dt$commercial2,dt$commercial3,dt$commercial4, na.rm=T)  #satisfies at least one of conditions 1,2, or 3.
  dt$commercial  = dt$commercial_123
  
  ###### Flag active years
  dt = dt[, c('id','loc_code','wk_code','gbd_code','scrape_year','first_activity_year','last_activity_year',
              'commercial1','commercial2','commercial3',
              'commercial','commercial_1','commercial_12','commercial_123','commercial_1234',
              'entire_home','high_avail','high_avail_active','ehha',
              'price','price_per_guest','minimum_nights','accommodates','new_nights','new_guests',
              'review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin',
              'review_scores_communication','review_scores_location','review_scores_value',
              'new_reviews','number_of_reviews')]
  
  without_data = c('2007','2008','2008','2009','2010','2011','2012','2013','2014')
  for (year in without_data){
    dt[first_activity_year <= year, paste('active',year,sep = "_") := 1] #we assume listing is active for ALL years between its first activity year and 2014.
    dt[first_activity_year > year, paste('active',year,sep = "_") := 0 ]
  }
  
  with_data = c('2015','2016','2017','2018','2019')
  for (year in with_data){
    dt[last_activity_year == year, paste('active',year,sep = "_") := 1 ]
    dt[last_activity_year != year, paste('active',year,sep = "_") := 0 ]
  }
  
  setorder(dt, id)
  
  
  ###### Listing counts at the location-level
  
  # Total active listings
  dt_id = dt[, .(active_2007=max(active_2007), active_2008 = max(active_2008),
                 active_2009=max(active_2009), active_2010 = max(active_2010),active_2011=max(active_2011),
                 active_2012=max(active_2012), active_2013 = max(active_2013),active_2014=max(active_2014),
                 active_2015=max(active_2015), active_2016 = max(active_2016),active_2017=max(active_2017),
                 active_2018=max(active_2018), active_2019 = max(active_2019)), by=list(id,loc_code)]
  
  dt_wk = dt_id[, .(active_2007=sum(active_2007), active_2008 = sum(active_2008),
                    active_2009=sum(active_2009), active_2010 = sum(active_2010),active_2011=sum(active_2011),
                    active_2012=sum(active_2012), active_2013 = sum(active_2013),active_2014=sum(active_2014),
                    active_2015=sum(active_2015), active_2016 = sum(active_2016),active_2017=sum(active_2017),
                    active_2018=sum(active_2018), active_2019 = sum(active_2019)), by=list(loc_code)]
  
  dt_count = reshape(dt_wk, direction = "long", varying = list(names(dt_wk)[2:14]), v.names = 'total', 
                     idvar = c("loc_code"), timevar = "year", times = 2007:2019)
  
  # Other types of active listings
  for (listing_type in c('commercial1','commercial2','commercial3','commercial','entire_home','high_avail','high_avail_active','ehha') ){
    
    dt_id = dt[get(listing_type)==1, .(active_2007=max(active_2007), active_2008 = max(active_2008),
                                       active_2009=max(active_2009), active_2010 = max(active_2010),active_2011=max(active_2011),
                                       active_2012=max(active_2012), active_2013 = max(active_2013),active_2014=max(active_2014),
                                       active_2015=max(active_2015), active_2016 = max(active_2016),active_2017=max(active_2017),
                                       active_2018=max(active_2018), active_2019 = max(active_2019)), by=list(id, loc_code)]
    
    dt_wk = dt_id[, .(active_2007=sum(active_2007), active_2008 = sum(active_2008),
                      active_2009=sum(active_2009), active_2010 = sum(active_2010),active_2011=sum(active_2011),
                      active_2012=sum(active_2012), active_2013 = sum(active_2013),active_2014=sum(active_2014),
                      active_2015=sum(active_2015), active_2016 = sum(active_2016),active_2017=sum(active_2017),
                      active_2018=sum(active_2018), active_2019 = sum(active_2019)), by=list(loc_code)]
    
    dt_aux = reshape(dt_wk, direction = "long", varying = list(names(dt_wk)[2:14]), v.names = listing_type, 
                     idvar = c("loc_code"), timevar = "year", times = 2007:2019)
    dt_count = merge(dt_count, dt_aux, by=c('year','loc_code') , all.x=TRUE)
    
  }
  
  setorder(dt_count, year, loc_code)
  dt_count[is.na(dt_count)] <- 0
  
  ###### Prices at the location-level
  dt_id = dt[price<999 & price>1 & commercial==1, .(price=mean(price), minimum_nights = mean(minimum_nights),
                                                    price_per_guest=mean(price_per_guest)), by=list(id, loc_code, scrape_year)]
  
  dt_price = dt_id[, year := scrape_year][, .(price_mean=mean(price),  price_median=median(price),
                                              minimum_nights_mean = mean(minimum_nights), minimum_nights_median = median(minimum_nights),
                                              price_per_guest_mean=mean(price_per_guest), price_per_guest_median=median(price_per_guest)), by=list(year,loc_code)]
  
  ###### Capacity at the location-level
  dt_id = dt[, .(active_2007=max(active_2007*accommodates), active_2008 = max(active_2008*accommodates),
                 active_2009=max(active_2009*accommodates), active_2010 = max(active_2010*accommodates),active_2011=max(active_2011*accommodates),
                 active_2012=max(active_2012*accommodates), active_2013 = max(active_2013*accommodates),active_2014=max(active_2014*accommodates),
                 active_2015=max(active_2015*accommodates), active_2016 = max(active_2016*accommodates),active_2017=max(active_2017*accommodates),
                 active_2018=max(active_2018*accommodates), active_2019 = max(active_2019*accommodates)), by=list(id, loc_code)]
  dt_wk = dt_id[, .(active_2007=sum(active_2007), active_2008 = sum(active_2008),
                    active_2009=sum(active_2009), active_2010 = sum(active_2010),active_2011=sum(active_2011),
                    active_2012=sum(active_2012), active_2013 = sum(active_2013),active_2014=sum(active_2014),
                    active_2015=sum(active_2015), active_2016 = sum(active_2016),active_2017=sum(active_2017),
                    active_2018=sum(active_2018), active_2019 = sum(active_2019)), by=list(loc_code)]
  
  dt_beds = reshape(dt_wk, direction = "long", varying = list(names(dt_wk)[2:14]), v.names = 'airbnb_beds', 
                    idvar = c("loc_code"), timevar = "year", times = 2007:2019)
  
  ###### Occupancy rate at the location-level
  dt_id = dt[!is.na(new_guests), .(new_guests=median(new_guests), accommodates=median(accommodates)), by=list(id, loc_code, scrape_year)]
  dt_occupancy = dt_id[!is.na(new_guests), year := scrape_year][, .(airbnb_occupancy_rate=sum(new_guests)/(sum(accommodates)*365)), by=list(year, loc_code)]
  
  ###### Occupancy rate at the location-level - only commercial listings
  dt_id = dt[!is.na(new_guests) & commercial==1, .(new_guests=median(new_guests), accommodates=median(accommodates)), by=list(id, loc_code, scrape_year)]
  dt_occupancy_commercial = dt_id[!is.na(new_guests), year := scrape_year][, .(airbnb_occupancy_rate_commercial=sum(new_guests)/(sum(accommodates)*365)), by=list(year, loc_code)]
  
  ###### Airbnb population per commercial listing 
  dt_id = dt[!is.na(new_guests), .(new_guests=median(new_guests), commercial=max(commercial)), by=list(id, loc_code, scrape_year)]
  dt_id = dt_id[!is.na(new_guests), year := scrape_year][, .(airbnb_pop=sum(new_guests/365),
                                                             airbnb_pop_commercial=sum(new_guests*commercial/365),
                                                             commercial=sum(commercial)), by=list(year, loc_code)]
  dt_id[airbnb_pop==0, airbnb_pop_per_commercial_listing:=0][airbnb_pop_commercial==0, airbnb_pop_per_commercial_listing:=0]
  dt_id[airbnb_pop>0 & commercial>0, airbnb_pop_per_commercial_listing:=airbnb_pop/commercial][airbnb_pop_commercial>0 & commercial>0, airbnb_pop_commercial_per_commercial_listing:=airbnb_pop_commercial/commercial]
  dt_id[airbnb_pop>0 & commercial==0, airbnb_pop_per_commercial_listing:=NA][airbnb_pop_commercial>0 & commercial==0, airbnb_pop_commercial_per_commercial_listing:=NA]
  dt_pop_commercial = dt_id[, list(year, loc_code, airbnb_pop, airbnb_pop_commercial, 
                                   airbnb_pop_per_commercial_listing, airbnb_pop_commercial_per_commercial_listing)]
  
  #### Reviews
  dt_id = dt[!is.na(new_guests), .(new_guests=median(new_guests), commercial=max(commercial), 
                                   review_scores_rating=mean(review_scores_rating),
                                   review_scores_accuracy=mean(review_scores_accuracy),review_scores_cleanliness=mean(review_scores_cleanliness),
                                   review_scores_checkin=mean(review_scores_checkin),review_scores_communication=mean(review_scores_communication),
                                   review_scores_location=mean(review_scores_location),review_scores_value=mean(review_scores_value),
                                   number_of_reviews=max(number_of_reviews), new_reviews=max(new_reviews)),by=list(id, loc_code, scrape_year)]
  dt_id1 = dt_id[!is.na(new_guests), year := scrape_year][, .(review_scores_rating=mean(review_scores_rating,na.rm=T),
                                                              review_scores_accuracy=mean(review_scores_accuracy,na.rm=T),
                                                              review_scores_cleanliness=mean(review_scores_cleanliness,na.rm=T),
                                                              review_scores_checkin=mean(review_scores_checkin,na.rm=T),
                                                              review_scores_communication=mean(review_scores_communication,na.rm=T),
                                                              review_scores_location=mean(review_scores_location,na.rm=T),
                                                              review_scores_value=mean(review_scores_value,na.rm=T),
                                                              number_of_reviews_mean=mean(number_of_reviews), new_reviews_mean=mean(new_reviews),
                                                              number_of_reviews_sum=sum(number_of_reviews), new_reviews_sum=sum(new_reviews)), by=list(year, loc_code)]
  dt_id2 = dt_id[!is.na(new_guests) & commercial==1, year := scrape_year][, .(number_of_reviews_commercial_mean=mean(number_of_reviews), new_reviews_commercial_mean=mean(new_reviews),
                                                             number_of_reviews_commercial_sum=sum(number_of_reviews), new_reviews_commercial_sum=sum(new_reviews)), by=list(year, loc_code)]
  dt_reviews=merge(dt_id1,dt_id2, by=c('year', 'loc_code'))
  dt_reviews = dt_reviews[, list(year, loc_code,
                                 review_scores_rating,review_scores_accuracy,review_scores_cleanliness,review_scores_checkin,
                                 review_scores_communication,review_scores_location,review_scores_value,
                                 number_of_reviews_mean, number_of_reviews_sum,new_reviews_mean, new_reviews_sum,
                                 number_of_reviews_commercial_mean, number_of_reviews_commercial_sum,new_reviews_commercial_mean, new_reviews_commercial_sum)]
  
  ### Accommodates
  dt_id = dt[!is.na(new_guests) & commercial==1, .(new_guests=median(new_guests), accommodates =mean(accommodates)), by=list(id, loc_code, scrape_year)]
  dt_accommodates = dt_id[!is.na(new_guests), year := scrape_year][, .(mean_accommodates =mean(accommodates),  median_accommodates=median(accommodates),
                                                                       sum_accommodates=sum(accommodates)), by=list(year,loc_code)]
  
  
  ###### Guests at the location-level
  dt_id = dt[!is.na(new_guests), .(new_guests=median(new_guests)), by=list(id, loc_code, scrape_year)]
  dt_guests = dt_id[!is.na(new_guests), year := scrape_year][, .(overnight_stays=sum(new_guests)), by=list(year, loc_code)]
  
  ###### Nights at the wijk-level
  dt_id = dt[!is.na(new_nights), .(new_nights=median(new_nights)), by=list(id, loc_code, scrape_year)]
  dt_nights = dt_id[!is.na(new_nights), year := scrape_year][, .(overnight_stays_nights=sum(new_nights)), by=list(year, loc_code)]
  
  ######  Final dataset
  dt_final = merge(dt_count, dt_price, by=c('year','loc_code') , all.x=TRUE)
  dt_final = merge(dt_final, dt_beds, by=c('year','loc_code') , all.x=TRUE)
  dt_final = merge(dt_final, dt_occupancy, by=c('year','loc_code') , all.x=TRUE)
  dt_final = merge(dt_final, dt_occupancy_commercial, by=c('year','loc_code') , all.x=TRUE)
  dt_final = merge(dt_final, dt_pop_commercial, by=c('year','loc_code') , all.x=TRUE)
  dt_final = merge(dt_final, dt_nights, by=c('year','loc_code') , all.x=TRUE)
  dt_final = merge(dt_final, dt_guests, by=c('year','loc_code') , all.x=TRUE)
  dt_final = merge(dt_final, dt_reviews, by=c('year','loc_code') , all.x=TRUE)
  dt_final = merge(dt_final, dt_accommodates, by=c('year','loc_code') , all.x=TRUE)
  
  if(location == 'wk_code'){
    
    ######  Wijk-level variables dataset
    dt_final =  dt_final[,.(year, loc_code,
                            total, entire_home,ehha, commercial, 
                            airbnb_beds,airbnb_pop, airbnb_pop_commercial, 
                            airbnb_occupancy_rate, airbnb_occupancy_rate_commercial,
                            price_mean, price_median, review_scores_location, mean_accommodates,
                            overnight_stays,overnight_stays_nights)]
    setnames(dt_final, old = c('loc_code') , new = c('wk_code'))
    write.csv(dt_final, "data/airbnb/listings_by_wk.csv", row.names = FALSE)
    
    ######  Annual variables dataset
    dt = dt_final[,.(listings_active=sum(total,na.rm=T), listings_entire_home=sum(entire_home,na.rm=T), 
                     listings_commercial=sum(commercial,na.rm=T),
                     overnight_stays = sum(overnight_stays,na.rm=T), overnight_stays_nights=sum(overnight_stays_nights,na.rm=T),
                     price_mean_mn = mean(price_mean,na.rm=T), price_mean_md = median(price_mean,na.rm=T),
                     price_median_mn = mean(price_median,na.rm=T), price_median_md = median(price_mean,na.rm=T) ), by=year]
    write.csv(dt, "data/airbnb/listings_by_year.csv", row.names = FALSE)
    
  }else{
    
    ######  Gebied-level variables dataset
    dt_final =  dt_final[,.(year, loc_code,
                            total, entire_home,ehha, commercial, 
                            airbnb_beds,airbnb_pop, airbnb_pop_commercial, 
                            airbnb_occupancy_rate, airbnb_occupancy_rate_commercial,
                            price_mean, price_median, review_scores_location, mean_accommodates,
                            overnight_stays,overnight_stays_nights)]
    setnames(dt_final, old = c('loc_code') , new = c('gbd_code'))
    write.csv(dt_final, "data/airbnb/listings_by_gb.csv", row.names = FALSE)
    write.csv(dt_final, "data/final/inputs/listings_by_gb.csv", row.names = FALSE)
    
  }

}

print('2_construct_airbnb_supply complete')
