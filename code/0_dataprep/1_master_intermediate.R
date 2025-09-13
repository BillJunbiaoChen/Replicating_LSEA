################################################################################
# Project: Location Sorting and Endogenous Amenities
# Purpose: Master file for R scripts related to data cleaning
# Author: Milena Almagro
################################################################################
wd <- paste0(dirname(getwd()),'/0_dataprep/subfiles')
options(warn = -1)

print(wd)
setwd(wd)
source('0_clean_googletrends.R', local = TRUE)
setwd(wd)

source('0_reproject_shapefiles.R', local = TRUE)
setwd(wd)

source('1_clean_airbnb_data.R', local = TRUE)
setwd(wd)

source('1_link_monuments_to_shapefiles.R', local = TRUE)
setwd(wd)

source('2_clean_bbga.R', local = TRUE)
setwd(wd)

source('2_construct_airbnb_supply.R', local = TRUE)
setwd(wd)

source('3_construct_neighborhood_panel.R', local = TRUE)
setwd(wd)
