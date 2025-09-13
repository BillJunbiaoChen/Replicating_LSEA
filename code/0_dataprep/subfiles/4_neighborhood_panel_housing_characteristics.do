* ------------------------------------------------------------------------------
* Project: Location Sorting and Endogenous Amenities
* Purpose: Construct neighborhood panel of covariates
* Author: Milena Almagro
* ------------------------------------------------------------------------------
cd "$DROPBOX_AMENITIES/data"

****** Merge everything together, start with covariates ******
import delimited "constructed/neighborhood_panel_gebied.csv", clear
rename gb_code gb

*** construct crosswalk from gebied to stadsteel **
preserve
keep if year == 2008
keep sd_code gb
export delimited "constructed/crosswalk_gebied_sd.csv", replace
restore

* Construct population tourists
bysort year: egen total_tourists_city = sum(pop_tourists_total)
gen pct_tourists = pop_tourists_total/total_tourists_city

* Merge to tenancy counts
merge 1:1 year gb using  constructed/gb_tenancy_counts
keep if _merge == 3
drop _merge

* Merge to rent 
merge 1:1 year gb using constructed/imputed_rent
keep if _merge == 3
drop _merge

* Merge to transaction values 
merge 1:1 year gb using constructed/transaction_values
keep if _merge == 3
drop _merge

* Merge to housing characteristics 
merge m:1 year gb using constructed/gebied_housing_characteristics_from_wk
keep if _merge == 3
drop _merge

keep year gb sd_code tourism_offices tourism_offices_net_hotels bars_locations restaurants_locations food_stores ///
nonfood_stores nurseries ///
hotel_beds airbnb_beds pop_tourists* ///
tenancy_status_1 tenancy_status_2 tenancy_status_3 ///
net_rent_hat_rf ///
 area_by_usage_1 total_usage_1 total_stage_5 

* Prepare covariates
gen social_housing = tenancy_status_3 
gen log_social_housing = log(social_housing)
gen log_rent_meter = log(12*net_rent_hat_rf/area_by_usage_1)
gen log_area_by_usage_1 = log(area_by_usage_1)
replace total_stage_5 = 0 if total_stage_5 == .
replace total_stage_5 = total_stage_5 + 1
gen log_stage_5 = log(total_stage_5)
rename area_by_usage_1 area_sq_m
gen unit_rent = 12*net_rent_hat_rf 

* Rename amenities
local varlist "tourism_offices_net_hotels restaurants_locations bars_locations food_stores nonfood_stores nurseries"
local i = 1
foreach var in `varlist' {
	gen amenity_`i' = `var'
	gen log_amenity_`i' = log(`var')
	local i = `i' + 1
}

* Generate dummies
tab gb, gen(dummy_gb)
tab year, gen(dummy_year)

* Keep relevant variables
keep log_* dummy* *amenity* year gb sd_code pop* total_usage_1 tenancy* *beds* area_sq_m unit_rent
 
save constructed/gebied_covariates_panel, replace
export delimited constructed/gebied_covariates_panel, replace
export delimited final/inputs/gebied_covariates_panel, replace

