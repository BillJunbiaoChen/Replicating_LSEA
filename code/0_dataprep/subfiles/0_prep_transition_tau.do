* ------------------------------------------------------------------------------
* Project: Endogenous Amenities and Location Sorting
* Purpose: Prepare exported data on demographic groups probability transitions
* Author: Milena Almagro
* ------------------------------------------------------------------------------

cd "$DROPBOX_AMENITIES/data"

forval i =1/3{
	import excel "cbs/exports/220331_0500_Transition location tenure - April 2022/Transition location tenure - April 2022", sheet("Transition `i'") first clear
	export delimited constructed/tau_transition_probs_`i', replace
}

import delimited constructed/tau_transition_probs_1, clear
save constructed/tau_transition_probs_all, replace
forval i =2/3{
	import delimited constructed/tau_transition_probs_`i', clear
	append using constructed/tau_transition_probs_all
	save constructed/tau_transition_probs_all, replace
}
export delimited final/inputs/tau_transition_probs_all, replace
