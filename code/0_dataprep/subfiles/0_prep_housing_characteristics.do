* ------------------------------------------------------------------------------
* Project: Endogenous Amenities and Location Sorting
* Purpose: Prepare exported data on housing characteristics
* Author: Milena Almagro
* ------------------------------------------------------------------------------

cd "$DROPBOX_AMENITIES/data"
import delimited "cbs/exports/wijk_housing_characteristics/Wijk- Housing characteristics - April 2022.csv", clear

* Crosswalk from wijks (neighborhoods) to gebieds (districts)
gen wk = wk_code - 36300
merge m:1 wk using "shapefiles/wijk_to_gb"

keep year wk gb total_units total_usage_1 total_usage_9 total_usage_* total_stage_* total_new_units_by_usage_1 area_by_usage_1 area_sq_m year_built 

* Compute counts and weighted average by gebied
local varlist "units usage_1 usage_2 usage_3 usage_4 usage_5 usage_6 usage_7 usage_8 usage_9 usage_10 usage_11 stage_1 stage_2 stage_3 stage_4 stage_5 new_units_by_usage_1"
foreach var in `varlist'{
	rename total_`var' `var'
	bysort year gb: egen total_`var' = sum(`var')
}

gen _area_by_usage_1 = area_by_usage_1*usage_1
gen _area_sq_m = area_sq_m*units
gen _year_built = year_built*units

local varlist "area_sq_m year_built"
foreach var in `varlist'{
	bysort year gb: egen total_`var' = sum(_`var')
	gen w_`var' = total_`var'/total_units 
}

bysort year gb: egen total_area_by_usage_1 = sum(_area_by_usage_1)
gen w_area_by_usage_1 = total_area_by_usage_1/total_usage_1

drop _*

* Collapse at the gebied level
collapse total_units total_usage_*  total_stage_* total_new_units_by_usage_1 w_*,by(year gb)

* Rename
rename w_area_sq_m area_sq_m
rename w_year_built year_built
rename w_area_by_usage_1 area_by_usage_1

* Recenter year built to 1850
gen log_year_built_new = log(year_built-1850)

cd "$DROPBOX_AMENITIES/data/constructed/"
save gebied_housing_characteristics_from_wk, replace
export delimited gebied_housing_characteristics_from_wk, replace
