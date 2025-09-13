###############################################
## Project: Endogenous Amenities and Location Sorting
## Purpose: Generate stylized facts for paper
## Author: Milena Almagro & Tomas Dominguez-Iino
###############################################

library(pacman)
pacman::p_load(rstudioapi, tidyverse, broom, data.table, 
               readstata13, readxl,
               lubridate,
               sf,
               hrbrthemes, viridis, RColorBrewer,ggplot2,ggpubr,
               latex2exp, DescTools)
options(warn = -1)
setwd(dirname(getwd()))
print(getwd())
rm(list=ls())
`%!in%` <- Negate(`%in%`)


##################  File locations

#World tourist-to-local ratio
dtw = merge(read_excel('../data/tourism/raw/esta_report.xlsx', sheet=1), 
            read_excel('../data/tourism/raw/esta_report.xlsx', sheet=2), by='city')
setDT(dtw)
dtw[, tourists_per_resident := arrivals_2018/population_2019]
setorder(dtw, -tourists_per_resident)

#City population data
dtc = read.csv(gzfile("../data/bbga/bbga.csv","rt"), header=T)
setDT(dtc)
varnames = c('hotel_beds','tourism_offices','nurseries','school_care','primary_schools','secondary_schools')
for (varname in varnames){
  dtc[is.na(get(varname)), noquote(varname) := get(paste0(varname, "_imp"))]
}
dtc[, education_services := education - primary_schools - secondary_schools]
dtc[, tourism_offices_net_hotels := tourism_offices - hotel_locations]
dtc_y = dtc[,.(population=sum(pop_total)), by=year]
dtc_y$year <- floor_date(as.Date(as.character(dtc_y$year), format="%Y"),'year')

#Hotel room and bed data - BBGA
dttb = dtc[,c('year','hotel_beds_imp','hotel_rooms_imp')]
setDT(dttb)
setnames(dttb, c('hotel_beds_imp','hotel_rooms_imp'),c('hotel_beds','hotel_rooms'))
dttb = dttb %>% filter(year < 2018)
dttb_y = dttb[, lapply(.SD, sum, na.rm=TRUE), by=year]
dttb_y$year <- floor_date(as.Date(as.character(dttb_y$year), format="%Y"),'year')

#Hotel data - Excel with tourism reports
dtte = read_excel('../data/tourism/tourism_figures.xlsx', sheet = 1)
setDT(dtte)
dtte = dtte %>% filter(year < 2018)
dtte_y = dtte
dtte_y$year <- floor_date(as.Date(as.character(dtte_y$year), format="%Y"),'year')

#Airbnb listings
dta = read.csv(gzfile("../data/airbnb/listings_by_wk.csv","rt"), header=T)
setDT(dta)
dta_y = dta[,.(listings_active=sum(total,na.rm=T), listings_entire_home=sum(entire_home,na.rm=T),listings_commercial=sum(commercial,na.rm=T)), by=year]
dta_y$year <- floor_date(as.Date(as.character(dta_y$year), format="%Y"),'year')

#Shapefile
dts = sf::read_sf(dsn=path.expand("../data/shapefiles/"), layer="wijk")
dts <- st_simplify(dts, dTolerance = 0.1, preserveTopology = T)
dts$wk_code = parse_number(dts$wk_code)
dts = dts[dts$wk_code %!in% c(10,11,50,51),] 



##################  1. Tourism flows, Airbnb listings, and hotels

#Visitors per 100 residents

dt =  merge(dtte_y, dtc_y, by = "year")
dt$nights_per_resident = dt$overnight_stays_total/(dt$population*365/100)

figure = ggplot(data = dt,  aes(x=year, y=nights_per_resident, group=1) ) +
  geom_line(linewidth=0.25) +
  geom_point(size=2.5) +
  theme_classic() +
  labs(subtitle='Overnight stays per 100 residents') +
  theme(aspect.ratio = 0.7,
        axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        axis.line = element_line(size=0.25),
        axis.ticks = element_line(size=0.25),
        text=element_text(family="Palatino")
        ) +
  scale_x_date(date_breaks="1 year", date_labels = "%Y")

pdf(paste0("../output/figures/stylized_facts/overnightstays_per_resident.pdf"), width = 4, height = 3)
print(figure)
dev.off()



#Hotel rooms and Airbnb listings - adjusted by availability

dt =  merge(dttb_y, dta_y, by = "year")
dt =  merge(dt, dtte_y[,c('year','number_hotels')], by = "year")
dt[, x1 := hotel_rooms]
dt[, x2 := listings_active]
dt[, x3 := listings_commercial]
dt = melt(dt[,c('year','x1','x2','x3')], id.vars = c("year"))

line_labels = c("Hotel rooms","Active Airbnb listings", "Active commercially-run Airbnb listings")
figure = ggplot(data = dt,  aes(x=year, y=value)) +
  geom_line(aes(linetype = variable, color=variable), linewidth=0.25) +
  geom_point(aes(shape = variable, color=variable), size=2.5) +
  labs(subtitle="Hotel rooms and Airbnb listings") +
  theme_classic() +
  theme(aspect.ratio = 0.7,
        axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        legend.title = element_blank(),
        legend.position = c(0.45,0.45),
        legend.background=element_blank(),
        axis.line = element_line(size=0.25),
        axis.ticks = element_line(size=0.25),
        text=element_text(family="Palatino")
  ) +
  scale_linetype_manual(labels = line_labels, values = c("solid", "solid", "solid"))+
  scale_shape_manual(labels = line_labels, values = c(15,19,17)) + 
  scale_colour_manual(labels = line_labels, values = c("darkgray","black","black")) +
  scale_x_date(date_breaks="1 year", date_labels = "%Y") 

pdf(paste0("../output/figures/stylized_facts/hotelrooms_and_listings_commercial.pdf"), width = 4, height = 3)
print(figure)
dev.off()








##################  2. Spatial distribution of hotels and Airbnb listings

dt =  merge(dtc, dta, by = c('wk_code',"year"))
dt[, 'commercial_share_rental':= commercial/(addresses_private_rental)]
dt[, 'commercial_share_market':= commercial/(addresses_private_rental + addresses_owner_occupant)]
dt[, 'commercial_share_total':= commercial/(addresses_total)]
dt[, 'hotel_beds_per_resident':= hotel_beds/(pop_total)]


# Summary stats of housing market

dtx = dt[year==2017,]
summary(dtx$commercial_share_rental)
summary(dtx$commercial_share_market)
summary(dtx$commercial_share_total)

dtx_y = dt[,.(rental_share=sum(addresses_private_rental)/sum(addresses_total), owner_share=sum(addresses_owner_occupant)/sum(addresses_total), 
              social_share=sum(addresses_owner_corporation)/sum(addresses_total), total=sum(addresses_total),
              commercial_share_rental=sum(commercial)/sum(addresses_private_rental),
              commercial_share_market=sum(commercial)/sum(addresses_owner_occupant+addresses_private_rental),
              commercial_share_total=sum(commercial)/sum(addresses_total) ), by=year]
print(dtx_y)



# Maps

for (year_n in c(2011,2013,2015,2017)){
  
  dtd = dt[year == year_n, .(year, wk_code, commercial_share_rental) ]
  dtd[commercial_share_rental==0, commercial_share_rental:=NA]
  
  dtd <- merge(dts, dtd, by='wk_code')
    
  figure =  ggplot() +
    geom_sf(data=dtd, aes(fill=commercial_share_rental), linewidth=0.1)+
    scale_fill_fermenter(palette = "Reds", direction = 1, na.value = "white", breaks=c(0.025,0.05,0.075,0.1,0.15,0.2), limits=c(-0.01,0.5)) +
    theme_void() +
    labs(title=year_n ,fill='') + 
    theme(legend.position = "right",  legend.direction = "vertical", 
          legend.text = element_text(size= 12),
          legend.key.size = unit(0.8, "cm"),
          legend.key.width = unit(0.6,"cm"),
          plot.title = element_text(size = 14, hjust=0.1, vjust=0),
          text=element_text(family="Palatino")
          )
  print(figure)
  dev.off()
  
  if(year_n==2011){figure_1=figure}
  if(year_n==2013){figure_2=figure}
  if(year_n==2015){figure_3=figure}
  if(year_n==2017){figure_4=figure}
  
}

figure=ggarrange(figure_1,figure_2,figure_4, nrow = 1, ncol=3, common.legend = TRUE, legend="right")
pdf(paste0("../output/figures/stylized_facts/commercial_listings_sh_p_combined.pdf"), width = 8.5, height = 2.5)
print(figure)
dev.off()



for (year_n in c(2008,2011,2013,2014,2015,2017)){
  
  dtd = dt[year == year_n, .(year, wk_code, hotel_beds_per_resident) ]
  dtd[hotel_beds_per_resident==0, hotel_beds_per_resident:=NA]

  dtd <- merge(dts, dtd, by='wk_code')
  figure = ggplot()+
    geom_sf(data=dtd, aes(fill=hotel_beds_per_resident), linewidth=0.1)+
    scale_fill_fermenter(palette = "Reds", direction = 1, na.value = "white", breaks=c(0.1,0.25,0.5,0.75,1,2), limits=c(-0.01,10)) +
    theme_void() +
    labs(title=year_n ,fill='') + 
    theme(legend.position = "right",  legend.direction = "vertical", 
          legend.text = element_text(size= 12),
          legend.key.size = unit(0.8, "cm"),
          legend.key.width = unit(0.6,"cm"),
          plot.title = element_text(size = 14, hjust=0.1, vjust=0),
          text=element_text(family="Palatino")
    )
  print(figure)
  dev.off()
  
  if(year_n==2011){figure_1=figure}
  if(year_n==2013){figure_2=figure}
  if(year_n==2015){figure_3=figure}
  if(year_n==2017){figure_4=figure}

}

figure=ggarrange(figure_1,figure_2,figure_4, nrow = 1, ncol=3, common.legend = TRUE, legend="right")
pdf(paste0("../output/figures/stylized_facts/hotel_beds_per_capita_combined.pdf"), width = 8.5, height = 2.5)
print(figure)
dev.off()




################## 3. Amenities are tilting towards tourists

map_breaks = c(-50,-20,-10,-5,0,5,10,20,50)
map_limits = c(-100,100)

#Food stores
dt1 = dtc[year==2011]
dt2 = dtc[year==2017]
dt = merge(dt1, dt2, by='wk_code')
dt[, growth := 100*(food_stores.y-food_stores.x)/food_stores.x]
dt = dt[, .(wk_code, food_stores.x, food_stores.y, growth)]
dt <- merge(dts, dt, by="wk_code")

figure_food_stores =  ggplot()+
  geom_sf(data=dt, aes(fill=growth), linewidth=0.1)+
  scale_fill_fermenter(palette = "RdBu", direction = -1, na.value = "white", breaks=map_breaks, limits=map_limits ) +
  theme_void() +
  labs(title = TeX(paste0('Food stores')), fill='') + 
  theme(legend.position = "right",  legend.direction = "vertical", 
        legend.text = element_text(size= 12),
        legend.key.size = unit(1.25,"cm"),
        legend.key.width = unit(0.5,"cm"),
        legend.title = element_blank(),
        plot.title = element_text(size = 14, hjust=0),
        text=element_text(family="Palatino"))
print(figure_food_stores)
dev.off()

#Non-food stores
dt1 = dtc[year==2011]
dt2 = dtc[year==2017]
dt = merge(dt1, dt2, by='wk_code')
dt[, growth := 100*(nonfood_stores.y-nonfood_stores.x)/nonfood_stores.x]
dt = dt[, .(wk_code, nonfood_stores.x, nonfood_stores.y, growth)]
dt <- merge(dts, dt, by="wk_code")

figure_nonfood_stores =  ggplot()+
  geom_sf(data=dt, aes(fill=growth), linewidth=0.1)+
  scale_fill_fermenter(palette = "RdBu", direction = -1, na.value = "white", breaks=map_breaks, limits=map_limits ) +
  theme_void() +
  labs(title = TeX(paste0('Non-food stores')), fill='') + 
  theme(legend.position = "right",  legend.direction = "vertical", 
        legend.text = element_text(size= 12),
        legend.key.size = unit(1.25,"cm"),
        legend.key.width = unit(0.5,"cm"),
        legend.title = element_blank(),
        plot.title = element_text(size = 14, hjust=0),
        text=element_text(family="Palatino"))
print(figure_nonfood_stores)
dev.off()


#Restaurants
dt1 = dtc[year==2011]
dt2 = dtc[year==2017]
dt = merge(dt1, dt2, by='wk_code')
dt[, growth := 100*(restaurants_locations.y-restaurants_locations.x)/restaurants_locations.x]
dt = dt[, .(wk_code, restaurants_locations.x, restaurants_locations.y, growth)]
dt <- merge(dts, dt, by="wk_code")

figure_restaurants =  ggplot()+
  geom_sf(data=dt, aes(fill=growth), linewidth=0.1)+
  scale_fill_fermenter(palette = "RdBu", direction = -1, na.value = "white", breaks=map_breaks, limits=map_limits ) +
  theme_void() +
  labs(title = TeX(paste0('Restaurants')), fill='') + 
  theme(legend.position = "right",  legend.direction = "vertical", 
        legend.text = element_text(size= 12),
        legend.key.size = unit(1.25,"cm"),
        legend.key.width = unit(0.5,"cm"),
        legend.title = element_blank(),
        plot.title = element_text(size = 14, hjust=0),
        text=element_text(family="Palatino"))
print(figure_restaurants)
dev.off()


#Bars
dt1 = dtc[year==2011]
dt2 = dtc[year==2017]
dt = merge(dt1, dt2, by='wk_code')
dt[, growth := 100*(bars_locations.y-bars_locations.x)/bars_locations.x]
dt = dt[, .(wk_code, bars_locations.x, bars_locations.y, growth)]
dt <- merge(dts, dt, by="wk_code")

figure_bars =  ggplot()+
  geom_sf(data=dt, aes(fill=growth), linewidth=0.1)+
  scale_fill_fermenter(palette = "RdBu", direction = -1, na.value = "white", breaks=map_breaks, limits=map_limits ) +
  theme_void() +
  labs(title = TeX(paste0('Bars')), fill='') + 
  theme(legend.position = "right",  legend.direction = "vertical", 
        legend.text = element_text(size= 12),
        legend.key.size = unit(1.25,"cm"),
        legend.key.width = unit(0.5,"cm"),
        legend.title = element_blank(),
        plot.title = element_text(size = 14, hjust=0),
        text=element_text(family="Palatino"))
print(figure_bars)
dev.off()


#Touristic amenities
dt1 = dtc[year==2011]
dt2 = dtc[year==2017]
dt = merge(dt1, dt2, by='wk_code')
dt[, growth := 100*(tourism_offices_net_hotels.y-tourism_offices_net_hotels.x)/tourism_offices_net_hotels.x]
dt = dt[, .(wk_code, tourism_offices_net_hotels.x, tourism_offices_net_hotels.y, growth)]
dt <- merge(dts, dt, by="wk_code")

figure_touristic =  ggplot()+
  geom_sf(data=dt, aes(fill=growth), linewidth=0.1)+
  scale_fill_fermenter(palette = "RdBu", direction = -1, na.value = "white", breaks=map_breaks, limits=map_limits ) +
  theme_void() +
  labs(title = TeX(paste0('Touristic amenities')), fill='') + 
  theme(legend.position = "right",  legend.direction = "vertical", 
        legend.text = element_text(size= 12),
        legend.key.size = unit(1.25,"cm"),
        legend.key.width = unit(0.5,"cm"),
        legend.title = element_blank(),
        plot.title = element_text(size = 14, hjust=0),
        text=element_text(family="Palatino"))
print(figure_touristic)
dev.off()


#Nurseries
dt1 = dtc[year==2011]
dt2 = dtc[year==2017]
dt = merge(dt1, dt2, by='wk_code')
dt[, growth := 100*(nurseries.y-nurseries.x)/nurseries.x]
dt[, change_nurseries :=nurseries.y-nurseries.x]
dt[, change_nurseries_p := 100*(nurseries.y-nurseries.x)/nurseries.x]
dt[, change_tourism_offices := tourism_offices_net_hotels.y-tourism_offices_net_hotels.x]
dt[, change_tourism_offices_p := 100*(tourism_offices_net_hotels.y-tourism_offices_net_hotels.x)/tourism_offices_net_hotels.x]
dt = dt[, .(wk_code, nurseries.y, nurseries.x, growth,
            change_nurseries,change_nurseries_p,
            change_tourism_offices,tourism_offices_net_hotels.y, tourism_offices_net_hotels.x,
            change_tourism_offices_p)]
dt <- merge(dts, dt, by="wk_code")

figure_nurseries =  ggplot()+
  geom_sf(data=dt, aes(fill=growth), linewidth=0.1)+
  scale_fill_fermenter(palette = "RdBu", direction = -1, na.value = "white", breaks=map_breaks, limits=map_limits ) +
  theme_void() +
  labs(title = TeX(paste0('Nurseries')), fill='') + 
  theme(legend.position = "right",  legend.direction = "vertical", 
        legend.text = element_text(size= 12),
        legend.key.size = unit(1.25,"cm"),
        legend.key.width = unit(0.5,"cm"),
        legend.title = element_blank(),
        plot.title = element_text(size = 14, hjust=0),
        text=element_text(family="Palatino"))
print(figure_nurseries)
dev.off()

figure=ggarrange(figure_touristic,figure_restaurants,figure_bars,figure_food_stores, figure_nonfood_stores,figure_nurseries, nrow = 2, ncol=3, common.legend = TRUE, legend="right")
pdf(paste0("../output/figures/stylized_facts/growth_amenities_combined.pdf"), width = 8.25, height = 4.5)
print(figure)
dev.off()



#Nurseries vs tourism offices
dt$level_name = '2011-17 pp change'
dt=setDT(dt)
figure = ggplot(data = dt[change_tourism_offices_p<200 & change_nurseries_p<200,], aes(x = change_tourism_offices_p, y = change_nurseries_p, shape=level_name)) +
  geom_point(size=1) +
  geom_smooth(method='lm', se=F, size=0.25, color='black') +
  labs(x = TeX(paste0('touristic amenities')), y=TeX(paste0('nurseries')), subtitle = "2011-17 pp change")   +
  theme_classic() +
  theme(axis.title.x = element_text(hjust = 1, size=12),
        axis.title.y = element_text(hjust = 1, size=12),
        plot.subtitle = element_text(size = 12, hjust=0.5),
        aspect.ratio = 0.75,
        panel.grid.minor = element_blank(),
        legend.position = 'none',
        text=element_text(family="Palatino"),
        axis.line = element_line(size=0.25),
        axis.ticks = element_line(size=0.25))
pdf(paste0("../output/figures/stylized_facts/binscatter_change_p_nurseries_touristic_amenities.pdf"), width = 3.75, height = 3.25)
print(figure)
dev.off()

#Summary stats
dt_sum = dt[change_nurseries_p>=0 | change_nurseries_p<=0,.(wk_code,change_nurseries_p)][is.finite(change_nurseries_p)]
dt_sum_neg = dt_sum[change_nurseries_p<=0,]
share_negative = nrow(dt_sum_neg)/nrow(dt_sum)
median_negative = median(dt_sum_neg$change_nurseries_p)

#Tourists per sq. km vs touristic amenities
dt <- merge(dtc, dta, by=c("year","wk_code"))
dt <- merge(dt, dtte[,c('year','bed_occupancy_rate')], by=c("year"))
dt = dt[,.(year, level_name, wk_code, airbnb_occupancy_rate, bed_occupancy_rate, airbnb_beds,hotel_beds,pop_total,tourism_offices_net_hotels) ]
dt <- merge(dt, dtc[year==2017,c('wk_code','area_land')], by=c("wk_code"))
dt[, pop_tourists_airbnb := (airbnb_occupancy_rate)*airbnb_beds][is.na(pop_tourists_airbnb), pop_tourists_airbnb := 0]
dt[, pop_tourists_hotels := (bed_occupancy_rate/100)*hotel_beds][is.na(pop_tourists_hotels), pop_tourists_hotels := 0]
dt[, tourist_share := (pop_tourists_hotels+pop_tourists_airbnb)/(pop_tourists_hotels+pop_tourists_airbnb+pop_total)]
dt[, tourism_offices_per_sqkm := tourism_offices_net_hotels/(area_land*100)]
dt[, tourists_per_sqkm := (pop_tourists_hotels+pop_tourists_airbnb)/(area_land*100)]
dt[, tourists_per_local := (pop_tourists_hotels+pop_tourists_airbnb)/(pop_total)]

#Scatter plot in levels
dtav = dt[year>=2011, list(level_name, wk_code, tourists_per_sqkm, tourism_offices_per_sqkm)]
dtav = dtav[, lapply(.SD, mean, na.rm=TRUE), by=list(level_name, wk_code)]
dtav$level_name = '2011-17 average'
figure = ggplot(data = dtav, aes(x = tourists_per_sqkm, y = tourism_offices_per_sqkm, shape=level_name)) +
  geom_point(size=1) +
  geom_smooth(method='lm', se=F, size=0.25, color='black') +
  labs(x = TeX(paste0("tourists/km2")), y=TeX(paste0("touristic amenities/km2")), subtitle = "2011-17 average")  +
  theme_classic() +
  theme(axis.title.x = element_text(hjust = 1, size=12),
        axis.title.y = element_text(hjust = 1, size=12),
        plot.subtitle = element_text(size = 12, hjust=0.5),
        aspect.ratio = 0.75,
        panel.grid.minor = element_blank(),
        legend.position = 'none',
        text=element_text(family="Palatino"),
        axis.line = element_line(size=0.25),
        axis.ticks = element_line(size=0.25))
pdf(paste0("../output/figures/stylized_facts/binscatter_tourists_touristic_amenities.pdf"), width = 3.75, height = 3.25)
print(figure)
dev.off()







################## 4. Demographic composition is changing heterogeneously across neighborhoods

# Ethnicity
dtc[, pop_share_dutch:= pop_dutch/pop_total]
dtc[, pop_share_morocco:= pop_morocco/pop_total]
dtc[, pop_share_antilles:= pop_antilles/pop_total]
dtc[, pop_share_suriname:= pop_suriname/pop_total]
dtc[, pop_colonies:= pop_suriname + pop_antilles]
dtc[, pop_share_colonies:= pop_colonies/pop_total]
dtc[, pop_share_turkey:= pop_turkey/pop_total]
dtc[, pop_share_w:= pop_w/pop_total]
dtc[, pop_share_othernonw:= pop_o_non_w/pop_total]
dtc[, pop_noneu:= pop_morocco+pop_antilles+pop_suriname+pop_turkey+pop_turkey+pop_o_non_w]
dtc[, pop_share_noneu:= pop_noneu/pop_total]
dtc[, pop_nondutch:=  pop_total-pop_dutch]
dtc[, pop_share_nondutch:=  1-pop_share_dutch]

# Household type
dtc[, pop_share_single:= pop_single_hh/pop_hh]
dtc[, pop_share_children:= pop_hh_w_child/pop_hh]
dtc[, pop_share_married_no_child:=pop_married_no_child/pop_hh]
dtc[, pop_share_married_w_child:=pop_married_w_child/pop_hh]
dtc[, pop_share_unmarried_no_child:=pop_unmarried_no_child/pop_hh]
dtc[, pop_share_unmarried_w_child:=pop_unmarried_w_child/pop_hh]

# Age
dtc[, pop_age_young:= pop15_19+pop20_24+pop25_29+pop30_34]
dtc[, pop_age_middle:= pop35_39+pop40_44+pop45_49+pop50_54+pop55_59+pop60_64]
dtc[, pop_age_old:= pop65_69+pop70_74+pop75_79+pop80_84+pop85_89+pop90_94+pop95_99+pop100plus]
dtc[, pop_share_age_young:= pop_age_young/pop_total]
dtc[, pop_share_age_middle:= pop_age_middle/pop_total]
dtc[, pop_share_age_old:= pop_age_old/pop_total]

# Skill type
dtc[, pop_share_skill_low:= pop_low_skill_p/100]
dtc[, pop_share_skill_medium:= pop_medium_skill_p/100]
dtc[, pop_share_skill_high:= pop_high_skill_p/100]

# Income level
dtc[, pop_share_incomeq1:= income_q1/100]
dtc[, pop_share_incomeq2:= income_q2/100]
dtc[, pop_share_incomeq3:= income_q3/100]
dtc[, pop_share_incomeq4:= income_q4/100]
dtc[, pop_share_incomeq5:= income_q5/100]

#Merge years
dt1 = dtc[year==2011]
dt2 = dtc[year==2016]
dt = merge(dt1, dt2, by='wk_code')

#Changes in population (share)
variable_list = c('pop_share_dutch','pop_share_morocco','pop_share_antilles','pop_share_suriname','pop_share_colonies','pop_share_turkey','pop_share_w','pop_share_othernonw','pop_share_noneu','pop_share_nondutch',
                  'pop_share_single','pop_share_children','pop_share_married_no_child','pop_share_married_w_child','pop_share_unmarried_no_child','pop_share_unmarried_w_child',
                  'pop_share_age_young','pop_share_age_middle','pop_share_age_old',
                  'pop_share_skill_low','pop_share_skill_medium','pop_share_skill_high',
                  'pop_share_incomeq1','pop_share_incomeq2','pop_share_incomeq3','pop_share_incomeq4','pop_share_incomeq5') 
variable_names = c('Dutch population share','Moroccan population share','Antillean population share','Surinamese population share','Dutch-colonial population share','Turkish population share','non-Dutch, Western population share','non-Western population share','non-European population share','non-Dutch population share',
                  'single population share','population share w/children','population share married w/o children','population share married w/children','population share single w/o children ','population share single w/children',
                  'young population share','middle-aged population share','old population share',
                  'low-skilled population share','medium-skilled population share','high-skilled population share',
                  'share w/income in bottom national quintile','share w/income in national quintile 2', 'share w/income in national quintile 3','share w/income in national quintile 4', 'share w/income in top national quintile') 
n=1
for (variable in variable_list){
  dt[, growth :=  get(noquote(paste0(variable,'.y'))) - get(noquote(paste0(variable,'.x')))]
  dtm <- merge(dts, dt, by="wk_code")
  figure = ggplot()+
    geom_sf(data=dtm, aes(fill=growth), linewidth=0.1)+
    scale_fill_fermenter(palette = "RdBu", direction = -1, na.value = "white", breaks=c(-0.1,-0.05,-0.025,-0.01,0,0.01,0.025,0.05,0.1), limits=c(-0.2,0.2) ) +
    theme_void() +
    labs(title = TeX(paste0("$\\Delta$ ",variable_names[n])), fill='') + 
    theme(legend.position = "right",  legend.direction = "vertical", 
          legend.text = element_text(size= 12),
          legend.key.size = unit(1.15, "cm"),
          legend.key.width = unit(0.6,"cm"),
          plot.title = element_text(size = 13, hjust=0),
          panel.background = element_rect(fill='white','white',0,'blank','white'),
          text=element_text(family="Palatino"))
  #pdf(paste0("../output/figures/stylized_facts/growth_",variable,".pdf"), width = 5, height = 3)
  print(figure)
  dev.off()
  n=n+1
  
  if(variable=='pop_share_dutch'){figure_1=figure}
  if(variable=='pop_share_noneu'){figure_2=figure}
  if(variable=='pop_share_incomeq5'){figure_3=figure}
  if(variable=='pop_share_incomeq1'){figure_4=figure}
  if(variable=='pop_share_married_w_child'){figure_5=figure}
  if(variable=='pop_share_unmarried_no_child'){figure_6=figure}
  
}

figure=ggarrange(figure_1,figure_2, nrow = 1, ncol=2, common.legend = TRUE, legend="right")
pdf(paste0("../output/figures/stylized_facts/growth_pop_share_ethnic_combined.pdf"), width = 8, height = 3)
print(figure)
dev.off()

figure=ggarrange(figure_3,figure_4, nrow = 1, ncol=2, common.legend = TRUE, legend="right")
pdf(paste0("../output/figures/stylized_facts/growth_pop_share_income_combined.pdf"), width = 8, height = 3)
print(figure)
dev.off()

figure=ggarrange(figure_5,figure_6, nrow = 1, ncol=2, common.legend = TRUE, legend="right")
pdf(paste0("../output/figures/stylized_facts/growth_pop_share_hhcomposition_combined.pdf"), width = 8, height = 3)
print(figure)
dev.off()
