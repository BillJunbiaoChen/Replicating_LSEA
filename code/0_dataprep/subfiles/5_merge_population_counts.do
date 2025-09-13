* ------------------------------------------------------------------------------
* Project: Location Sorting and Endogenous Amenities
* Purpose: Combine population counts with tourists
* Author: Milena Almagro
* ------------------------------------------------------------------------------

cd "$DROPBOX_AMENITIES/data"
use constructed/gebied_population_counts_panel, clear

egen group = group(gb year)

reshape wide num_hh, i(group) j(combined_cluster)

merge 1:1 gb year using constructed/gebied_covariates_panel
drop _merge

keep group gb year num_hh* pop*

gen num_hh7 = pop_tourists_hotels + pop_tourists_airbnb

drop pop_*

reshape long num_hh, i(group) j(combined_cluster)

replace num_hh = 0 if num_hh == .

save constructed/gebied_population_w_tourists_counts_panel, replace
export delimited final/inputs/gebied_population_w_tourists_counts_panel, replace
