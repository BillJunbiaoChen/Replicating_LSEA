* ------------------------------------------------------------------------------
* Project: Location Sorting and Endogenous Amenities
* Purpose: GMM estimation of location choice model
* Author: Milena Almagro; Tables: Sriram Tolety
* ------------------------------------------------------------------------------
set type double

* Convert to dta and append
cd "$DROPBOX_AMENITIES/data/constructed"
forval i = 1/3{
	import delimited "prep_reduced_eccp_data_g_`i'", clear
	save prep_reduced_eccp_data_g_`i', replace
}

use prep_reduced_eccp_data_g_1.dta, clear
save prep_reduced_eccp_data_all, replace
forval i =2/3{
	use prep_reduced_eccp_data_g_`i'.dta, clear
	append using prep_reduced_eccp_data_all
	save prep_reduced_eccp_data_all, replace
}
rename period year
replace year = year + 2007
rename tau_vec_full tau
save prep_reduced_eccp_data_all, replace

////// Construct instruments /////////
use "gebied_covariates_panel", clear 

* Bartik instruments *
gen bartik_sh_2011 = log_social_housing*(year >= 2011)
gen bartik_sh_2015 = log_social_housing*(year >= 2015)
gen sh_pop_airbnb = pop_tourists_airbnb/total_usage_1
gen bartik_hotels =  sh_pop_airbnb*(year >= 2017)

* BLP instruments *
bysort year sd_code (gb):  gen N_sd = _N
local varlist "log_area_by_usage_1 log_social_housing log_stage_5"
 
foreach var in `varlist' {
	bysort year (gb):  egen total_`var' = sum(`var')
	bysort year sd_code (gb):  egen total_sd_`var' = sum(`var')
	bysort year (sd_code gb):  gen loom_`var' = (total_`var'-total_sd_`var')/(_N-N_sd)
}

* Merge to transition probabilities
cd "$DROPBOX_AMENITIES/data/constructed"
merge 1:m gb year using prep_reduced_eccp_data_all

keep if year >= 2008 & year <= 2018

/////// Regressions //////

* For quick output 
eststo clear
local mc_varlist tau gamma_0_vec gamma_1_vec gamma_2_vec
local endo_varlist log_rent_meter log_amenity_1 log_amenity_2 log_amenity_3 log_amenity_4 log_amenity_5 log_amenity_6


forval i = 1/3{
		qui eststo ivregress_g`i': ivreg2 y dummy_gb1-dummy_gb22 dummy_year* `mc_varlist'  ///
		log_social_housing log_area_by_usage_1 ///
		(`endo_varlist' = ///
		loom_log_social_housing loom_log_stage_5 log_stage_5 loom_log_area_by_usage_1 bartik_sh_2011 bartik_sh_2015 bartik_hotels ) if combined == `i', noconstant 
		predict y_hat_`i' if combined == `i' , xb
		predict res_`i' if combined == `i' , res
		
		estimates store M`i'

		local varlist `mc_varlist'
		gen u_hat_`i' = y_hat_`i' if combined == `i'
		
		foreach var in `varlist'{
			replace u_hat_`i' = u_hat_`i' - _b[`var']*`var' if combined == `i'
			replace u_hat_`i' = 0 if gb == 23
			
		}
		mat V=e(V)
		esttab mat(V) using "$DROPBOX_AMENITIES/output/estimates/var_cov_`i'.csv", replace mlab(none)
}
esttab ivregress_*, drop(dummy*) stats(N widstat)
estout ivregress_* using "$DROPBOX_AMENITIES/output/estimates/ivreg_demand_location_choice_estimates.csv", cells(b(fmt(10))) replace 
estout ivregress_* using "$DROPBOX_AMENITIES/data/final/estimates/ivreg_demand_location_choice_estimates.csv", cells(b(fmt(10)))  replace
esttab ivregress_*  using "$DROPBOX_AMENITIES/output/tables/ivreg_demand_location_choice_estimates.tex", replace nogap drop( dummy* /// 
log_area_by_usage_1 log_social_housing) mtitles("Group 1" "Group 2" "Group 3") scalar (N) b(%5.3f) se(%5.3f) star(* 0.10 ** 0.05 *** 0.01) ///
nonumbers rename(log_rent_meter_new "Rent" log_amenity_1 "Tourism Offices" log_amenity_2 "Restaurants" log_amenity_3 "Bars" log_amenity_4 "Food Stores" log_amenity_5 "Nonfood Stores" log_amenity_6 "Nurseries" tau "High Location Capital" gamma_0_vec "Intra-City Moving Cost" gamma_1_vec "Bilateral Moving Cost"  gamma_2_vec "In/Out of City Moving Cost")


* Save indirect utility static
preserve
keep year p_gb p_tau renewal gb combined res* y_hat* u_hat* 
gen u_hat = .


forval i = 1/3{
	
	replace u_hat = u_hat_`i' if combined == `i'
	
}
collapse u_hat , by(gb year combined)

drop if year == .

export delimited "$DROPBOX_AMENITIES/output/estimates/demand_location_choice_u_hat.csv", replace
export delimited "$DROPBOX_AMENITIES/data/final/estimates/demand_location_choice_u_hat.csv", replace
restore

* GMM
eststo clear
local mc_varlist tau gamma_0_vec gamma_1_vec gamma_2_vec
local endo_varlist log_rent_meter log_amenity_1 log_amenity_2 log_amenity_3 log_amenity_4 log_amenity_5 log_amenity_6

forval i = 1/3{
		 eststo ivregress_g`i': gmm ( y -{xb: dummy_gb1-dummy_gb22 dummy_year* `mc_varlist' ///
		log_social_housing log_area_by_usage_1 ///
		`endo_varlist'}) if combined == `i',  ///
		instruments(dummy_gb1-dummy_gb22 dummy_year*  `mc_varlist' ///
				   log_social_housing log_area_by_usage_1 ///
				   loom_log_social_housing loom_log_stage_5 log_stage_5 loom_log_area_by_usage_1 bartik_sh_2011 bartik_sh_2015 bartik_hotels, ///
				   noconstant)
		
 }

esttab ivregress_*, drop(dummy*) stats(N widstat)
estout ivregress_* using "$DROPBOX_AMENITIES/output/estimates/gmm_demand_location_choice_estimates.csv", cells(b) replace
esttab ivregress_* using "$DROPBOX_AMENITIES/output/tables/gmm_demand_location_choice_estimates.tex", ///
    replace nogap drop(dummy* log_area_by_usage_1 log_social_housing) ///
	drop(_eq_xb) ///
    mtitles("Older Families" "Singles" "Younger Families") ///
    stats(N, fmt(%9.0fc) labels("\(N\)")) ///
    b(%5.3f) se(%5.3f) star(* 0.10 ** 0.05 *** 0.01) ///
    nonumbers ///
    rename(log_rent_meter "Log Rent" ///
           log_amenity_1 "Log Touristic Amenities" ///
           log_amenity_2 "Log Restaurants" ///
           log_amenity_3 "Log Bars" ///
           log_amenity_4 "Log Food Stores" ///
           log_amenity_5 "Log Nonfood Stores" ///
           log_amenity_6 "Log Nurseries" ///
           tau "High Location Capital" ///
           gamma_0_vec "Intra-City Moving Cost" ///
           gamma_1_vec "Bilateral Moving Cost" ///
           gamma_2_vec "In/Out of City Moving Cost") ///
    prehead("\begin{table}[ht!]" ///
            "\footnotesize" ///
            "\centering" ///
            "\caption{Preference parameter demand estimation results}\label{tab:demand_estimation_locals}" ///  // Corrected label
            "\scalebox{1}{" ///
            "\def\sym#1{\ifmmode^{#1}\else\(^{#1}\)\fi}" ///
            "\begin{tabular}{lccc}" ///
            "\toprule" ///
            "& \multicolumn{3}{c}{Dependent variable: Relative} \\" ///
            "& \multicolumn{3}{c}{Likelihood of Renewal Paths} \\" ///
            "\cmidrule{2-4}") ///
    posthead("\midrule") ///
    prefoot("\midrule") ///
    postfoot("\bottomrule" ///
             "\end{tabular}" ///
             "}" ///
             "\legend{" ///
             "This table presents regression results of preference parameters for a dynamic location choice model for 22 districts in Amsterdam for 2008-2019. We estimate preference parameters separately for three groups via two-step optimal GMM. The dependent variable is differences in path likelihoods, after normalizing with respect to the outside option. After this normalization, each type has 46 possible states (23 past locations and two location capital categories), 22 possible actions, and 21 possible renewal actions over 11 years, which leads to 233,772 possible states and two-step path combinations. We omit exogenous controls---the log of social housing units and the log of the average apartment in square meters---for the ease of exposition. Two-step efficient GMM standard errors in parenthesis. \sym{*}\(p<0.10\), \sym{**}\(p<0.05\), \sym{***}\(p<0.01\).}" ///
             "\end{table}")

* Estimating joint model to determine if coefficients are different amongst different groups - these checks are not included in the paper itself, but included for completeness.
/*
local varlist gamma_0_vec gamma_1_vec gamma_2_vec log_rent_meter log_amenity_1 log_amenity_2 log_amenity_3 log_amenity_4 log_amenity_5 log_amenity_6 log_social_housing log_area_by_usage_1 loom_log_social_housing loom_log_stage_5 log_stage_5 loom_log_area_by_usage_1 bartik_sh_2011 bartik_sh_2015 bartik_hotels
		
forval i =1/3{
	foreach var in `varlist'{
		gen `var'_`i' = (combined == `i')*`var'
		}
	
	forval j =1/22{
		gen dummy_gb`j'_`i' = (combined == `i')*dummy_gb`j'
	}

	forval y = 1/11{
		gen dummy_year`y'_`i' = (combined == `i')*dummy_year`y'
	}
	
}
ivreg2 y dummy_gb*_* dummy_year*_* gamma_0_vec_* gamma_1_vec_* gamma_2_vec_* ///
		log_social_housing_* log_area_by_usage_1_* ///
		(log_amenity_*_* log_rent_meter_* = ///
		loom_log_social_housing_* loom_log_stage_5_* log_stage_5_* loom_log_area_by_usage_1_* bartik_sh_2011_* bartik_sh_2015_* bartik_hotels_* ), noconstant 
		
forval i =1/2{
	local next = `i' + 1
	forval j = `next'/3{
		forval s = 1/6{
			testnl  _b[log_rent_meter_`j']*_b[log_amenity_`s'_`i'] =  _b[log_rent_meter_`i']*_b[log_amenity_`s'_`j']
		}
	}
}
*/
