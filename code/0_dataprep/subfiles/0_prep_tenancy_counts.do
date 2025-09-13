* ------------------------------------------------------------------------------
* Project: Endogenous Amenities and Location Sorting
* Purpose: Prepare exported data on housing tenancy counts 
* Author: Milena Almagro
* ------------------------------------------------------------------------------

* Save tenancy status by gebied (district)
cd "$DROPBOX_AMENITIES/data/cbs/exports/220224_0500_Gebied - Tenancy counts - February 2022"
import delimited "gebied_tenancy_status_counts", clear

local varlist "num_obs0 num_obs1 num_obs2 num_obs3"

forval i=1/3 {
	rename num_obs`i' tenancy_status_`i'
}

keep year gb tenancy*
cd "$DROPBOX_AMENITIES/data/constructed"
save gb_tenancy_counts, replace
export delimited gb_tenancy_counts, replace

