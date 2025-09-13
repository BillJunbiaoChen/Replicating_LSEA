* ------------------------------------------------------------------------------
* Project: Location Sorting and Endogenous Amenities
* Purpose: Master file for demand estimation
* Author: Milena Almagro
* ------------------------------------------------------------------------------

ssc install estout, replace

* Get the current directory
local current_dir = c(pwd)

* Go one level up by finding the position of the last '/'
local last_slash_pos = strrpos("`current_dir'", "/")
local parent_dir = substr("`current_dir'", 1, `last_slash_pos' - 1)

* Go two levels up by finding the position of the last '/' in the parent directory
local last_slash_pos_parent = strrpos("`parent_dir'", "/")
local grandparent_dir = substr("`parent_dir'", 1, `last_slash_pos_parent' - 1)

local last_slash_pos_grandparent = strrpos("`grandparent_dir'", "/")
local greatgrandparent_dir = substr("`grandparent_dir'", 1, `last_slash_pos_grandparent' - 1)

display "`grandparent_dir'"

* Set the global variable
global DROPBOX_AMENITIES "`grandparent_dir'"

cd "$DROPBOX_AMENITIES/code/2_estimation/demand"
do 1_commercial_airbnb_demand_with_outside_option.do

cd "$DROPBOX_AMENITIES/code/2_estimation/demand"
do 1_gmm_estimation_location_choice_static.do

cd "$DROPBOX_AMENITIES/code/2_estimation/demand"
do 1_gmm_estimation_location_choice.do

shell sed -i.bak ‘s/^xb//’ “$DROPBOX_AMENITIES/output/tables/gmm_demand_location_choice_estimates.tex”
rm “$DROPBOX_AMENITIES/output/tables/gmm_demand_location_choice_estimates.tex.bak”