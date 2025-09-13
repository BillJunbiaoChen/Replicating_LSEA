* ------------------------------------------------------------------------------
* Project: Location Sorting and Endogenous Amenities
* Purpose: Master file for final cleaning steps in Stata 
* Author: Milena Almagro
* ------------------------------------------------------------------------------
log using "3_master_dataprep_final.log", replace

* Get the current directory
local current_dir = c(pwd)

* Go one level up by finding the position of the last '/'
local last_slash_pos = strrpos("`current_dir'", "/")
local parent_dir = substr("`current_dir'", 1, `last_slash_pos' - 1)

* Go two levels up by finding the position of the last '/' in the parent directory
local last_slash_pos_parent = strrpos("`parent_dir'", "/")
local grandparent_dir = substr("`parent_dir'", 1, `last_slash_pos_parent' - 1)

* Set the global variable
global DROPBOX_AMENITIES "`grandparent_dir'"

display "Global DROPBOX_AMENITIES is set to: $DROPBOX_AMENITIES"

cd "$DROPBOX_AMENITIES/code/0_dataprep/subfiles"
do 4_neighborhood_panel_housing_characteristics.do
cd "$DROPBOX_AMENITIES/code/0_dataprep/subfiles"
do 5_merge_population_counts.do
cd "$DROPBOX_AMENITIES/code/0_dataprep/subfiles"
do 6_construct_panel_structural_amenities.do

log close
