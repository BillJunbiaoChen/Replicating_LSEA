* ------------------------------------------------------------------------------
* Project: Endogenous Amenities and Location Sorting
* Purpose: Prepare exported data on income at the group-year level
* Author: Milena Almagro
* ------------------------------------------------------------------------------


* Read exported data 
cd "$DROPBOX_AMENITIES/data"
import excel "cbs/exports/group_characteristics/Group Characteristics - April 2022", sheet("Yearly Income") first clear

* Drop if district is unknown (gb == 0) or missing (gb == .)
drop if gb == 0 | gb == .
* Drop undefined groups 
drop if combined == .

* Weighted mean of income across locations with population counts as weights
gen aux = num_hh*disposable_income
bysort year combined : egen total_income = sum(aux)
bysort year combined: egen total_hh = sum(num_hh)
gen yearly_income = total_income/total_hh

bysort year: egen total_city_income = sum(aux)
bysort year: egen total_city_hh = sum(num_hh)
gen yearly_city_income = total_city_income/total_city_hh

* Weighted median of income across locations with population counts as weights
bysort year combined (disp): gen cum_num_hh = sum(num_hh)
bysort year combined (disp): gen ind = (cum_num_hh< total_hh/2)
bysort year combined (ind disp): gen median_yearly_income = disposable_income[1]

* Collapse to yearly observations
collapse median_yearly_income yearly_income yearly_city_income, by(year combined_cluster)
rename yearly_income disposable_income
rename yearly_city_income disposable_city_income
rename median_yearly_income median_disposable_income

* Compute average income at the group-year level
bysort combined: egen avg_disposable_income = mean(disposable_income)
bysort combined: egen avg_median_disposable_income = mean(median_disposable_income)

* Add infromation about tourist income from Loon and Rouwendal (2017)
* https://link.springer.com/article/10.1007/s10824-017-9293-1
set obs `=_N+12'
replace combined = 7 if combined == .
bysort combined: replace year =  _n +2007 if combined == 7
replace disposable_income = 139*365 if combined == 7
replace avg_disposable_income = 139*365 if combined == 7
replace median_disposable_income = 139*365 if combined == 7
replace avg_median_disposable_income = 139*365 if combined == 7
replace disposable_city_income = 139*365 if combined == 7

save constructed/annual_income, replace
export delimited constructed/annual_income, replace
export delimited final/inputs/annual_income, replace



