* ------------------------------------------------------------------------------
* Project: Endogenous Amenities and Location Sorting
* Purpose: Prepare exported data on population counts
* Author: Milena Almagro
* ------------------------------------------------------------------------------


* Create population counts by district within Amsterdam 
cd "$DROPBOX_AMENITIES/data/cbs/exports/group_characteristics"
import excel "Group Characteristics - April 2022", sheet("Yearly Income") first clear
keep year gb combined num_hh

* Drop if district unknown (gb == 0)
drop if gb == 0
cd "$DROPBOX_AMENITIES/data/constructed/"
drop if year == .
drop if gb == .
drop if combined == .
export delimited gebied_population_counts_panel, replace
save gebied_population_counts_panel, replace

cd "$DROPBOX_AMENITIES/data/final/inputs"
export delimited gebied_population_counts_panel, replace

* Create population counts by state variable
forval i = 1/4{
	cd "$DROPBOX_AMENITIES/data/cbs/exports/MNL_coefficients"
	import excel "MNL Coefficients - April 2022", sheet("Predicted probabilities `i'") first clear
	collapse num_hh, by(year p_gb p_tau)
	gen combined_cluster = `i'
	cd "$DROPBOX_AMENITIES/data/constructed/"
	export delimited population_counts_by_state_g_`i', replace
}

* Append all groups
cd "$DROPBOX_AMENITIES/data/constructed/"
import delimited population_counts_by_state_g_1, clear
save population_counts_by_state, replace 
forval i = 2/4{
	import delimited population_counts_by_state_g_`i', clear
	append using population_counts_by_state
	save population_counts_by_state, replace
}
