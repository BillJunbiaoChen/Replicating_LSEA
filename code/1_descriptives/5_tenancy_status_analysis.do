* ------------------------------------------------------------------------------
* Project: Location Sorting and Endogenous Amenities
* Purpose: Check changes in tenancy status by sd
* Author: Milena Almagro
* ------------------------------------------------------------------------------
* Get the current directory
local current_dir = c(pwd)

set scheme s2mono // Set color scheme for graphs
graph set eps fontface "Palatino"

* Go one level up by finding the position of the last '/'
local last_slash_pos = strrpos("`current_dir'", "/")
local parent_dir = substr("`current_dir'", 1, `last_slash_pos' - 1)

* Go two levels up by finding the position of the last '/' in the parent directory
local last_slash_pos_parent = strrpos("`parent_dir'", "/")
local grandparent_dir = substr("`parent_dir'", 1, `last_slash_pos_parent' - 1)

* Set the global variable
global DROPBOX_AMENITIES "`grandparent_dir'"

display "Global DROPBOX_AMENITIES is set to: $DROPBOX_AMENITIES"

cd "$DROPBOX_AMENITIES/"



****** Merge everything together, start with covariates ******
import delimited "data/constructed/neighborhood_panel_gebied.csv", clear
rename gb_code gb

*** construct crosswalk from gebied to stadsteel **
preserve
keep if year == 2008
keep sd_code gb
export delimited "data/constructed/crosswalk_gebied_sd.csv", replace
restore

* Construct population tourists
bysort year: egen total_tourists_city = sum(pop_tourists_total)
gen pct_tourists = pop_tourists_total/total_tourists_city

cd "data/constructed/"
* Merge to tenancy counts
merge 1:1 year gb using  gb_tenancy_counts
keep if _merge == 3
drop _merge

collapse (sum) tenancy_status*, by(sd_code year)


forval i =1/3{
	bysort sd_code (year): gen new_units_`i' = tenancy_status_`i'-tenancy_status_`i'[_n-1]
	bysort sd_code (year): gen growth_units_`i' = (tenancy_status_`i'-tenancy_status_`i'[_n-1])/tenancy_status_`i'[_n-1]
	gen share_`i' = tenancy_status_`i'/(tenancy_status_1+tenancy_status_2+tenancy_status_3)
	bysort sd_code (year): gen change_share_`i' = share_`i'-share_`i'[_n-1]
}

encode sd_code, gen(id)
xtset id year

xtline share_1, overlay
xtline share_2, overlay
xtline share_3, overlay


xtline change_share_1, overlay
xtline change_share_2, overlay
xtline change_share_3, overlay


xtline tenancy_status_1, overlay
xtline tenancy_status_2, overlay
xtline tenancy_status_3, overlay


xtline new_units_1, overlay
xtline new_units_2, overlay
xtline new_units_3, overlay

xtline growth_units_1, overlay
xtline growth_units_2, overlay
xtline growth_units_3, overlay
