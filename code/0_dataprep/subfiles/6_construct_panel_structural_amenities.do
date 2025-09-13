* ------------------------------------------------------------------------------
* Project: Location Sorting and Endogenous Amenities
* Purpose: Construct neighborhood panel for estimation of amenity supply
* Author: Milena Almagro
* ------------------------------------------------------------------------------

cd "$DROPBOX_AMENITIES/data"
use constructed/gebied_covariates_panel, clear

* Merge with population
merge 1:m gb year using constructed/gebied_population_w_tourists_counts_panel
drop _merge

* Merge with income
merge m:1 combined year using constructed/annual_income
drop _merge

* Merge with expenditure shares 
merge m:1 combined using constructed/expenditure_shares
drop _merge

* Restrict years 
keep if year >= 2008 & year <= 2018

* Keep only inside locations
keep if gb != 101

* Generate total budget
gen budget_h = (1-exp_sh)*num_hh*disposable_income
gen inc_sh = (1-exp_sh)*disposable_income

keep amenity* budget_h* disposable_income* num_hh inc_sh* tenancy_status* *beds* gb combined year pop_tourists_hotels pop_tourists_airbnb sd_code 

* total number of amenities and groups
local S = 6
local K = 7

* Reshape to wide 
egen group = group(gb year)
reshape wide budget_h disposable_income num_hh inc_sh, i(group) j(combined)

* Construct instruments
gen bartik_1 = inc_sh1*tenancy_status_1
gen bartik_2 = inc_sh2*tenancy_status_1
gen bartik_3 = inc_sh3*tenancy_status_2
gen bartik_4 = inc_sh4*tenancy_status_2
gen bartik_5 = inc_sh5*tenancy_status_3
gen bartik_6 = inc_sh6*tenancy_status_3
gen bartik_7 = inc_sh7*hotel_beds

reshape long amenity_, i(group) j(s)
rename amenity_ amenity 

*Construct interactions
forval j = 1/`S'{
	forval k = 1/`K'{
		gen budget_h_`j'_`k' = 0
		replace budget_h_`j'_`k' = budget_h`k' if s == `j'
	}
}

* Construct instrument interactions
local varlist "bartik_1 bartik_2 bartik_3 bartik_4 bartik_5 bartik_6 bartik_7"
forval j = 1/`S'{
	foreach var in `varlist' {
		gen z_b_`j'_`var' = 0
		replace z_b_`j'_`var' = `var' if s == `j'
	}

}

sort s gb year 
tab gb, gen(dummy_j)
tab year, gen(dummy_year)

bysort gb year: egen total_amenity = sum(amenity)

egen sd_code_year = group(sd_code year)
replace sd_code_year = . if year == 2008

tab sd_code_year, gen(int_dummy)
forval i =1/70{
	replace int_dummy`i' = 0 if sd_code_year == .
}


sort s gb year 

keep amenity total_amenity *budget_h* z* dummy_* gb s year int_*


save constructed/panel_amenities_structural_estimation, replace
export delimited constructed/panel_amenities_structural_estimation, replace	 
export excel using "constructed/panel_amenities_structural_estimation.xlsx", replace firstrow(variables)
export delimited final/inputs/panel_amenities_structural_estimation, replace	 

gen log_total_amenity = log(total_amenity)
reg log_total_amenity dummy*





