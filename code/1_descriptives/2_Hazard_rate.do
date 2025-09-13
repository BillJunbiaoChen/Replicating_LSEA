* ------------------------------------------------------------------------------
* Project: Location Sorting and Endogenous Amenities
* Purpose: Generate hazard rate graphic
* Author: Milena Almagro
* ------------------------------------------------------------------------------

set scheme s2mono // Set color scheme for graphs
graph set eps fontface "Palatino"

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

cd "$DROPBOX_AMENITIES/data/cbs/exports/230828_0500_Revisions ECMA Hazard Rate - Summer 2023"

import delimited hazard_rate, clear

twoway line movement p_tenure, xlabel(1 (4) 16,labsize(medlarge)) ///
 ylabel(0.0 (0.05) 0.15, nogrid labsize(medlarge)) ///
xtitle("Past tenure location", size(medlarge)) ///
ytitle("Probability of moving", size(medlarge)) ///
 graphregion(color(white)) 
graph export "$DROPBOX_AMENITIES/output/figures/stylized_facts/hazard_rate.png", replace
