* ------------------------------------------------------------------------------
* Project: Location Sorting and Endogenous Amenities
* Purpose: Explore graph heuristics for paper - clusters etc.
* Author: Milena Almagro
* ------------------------------------------------------------------------------

set scheme s2mono // Set color scheme for graphs
graph set eps fontface "Palatino"
clear

* Get the current directory
local current_dir = c(pwd)

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

import excel "data/cbs/exports/Vrijgegeven220328_0500_Group Characteristics - April 2022/Group Characteristics - April 2022.xlsx", sheet("Heuristic indexes") firstrow // Import the heuristics file
keep if Numerofclusters != .  // Drop empty observations
label variable Numerofclusters "Number of Clusters" // Label for clarity

/* Destring variables so we can graph them */
foreach var of varlist CH*{
	replace `var' = "" if `var' == "NA"
	destring `var', replace
}


// Graph the CH clusters and Elbow clusters
cd "output/figures/stylized_facts/"
twoway line CH1 Numerofclusters || line CH2 Numerofclusters || line CH3 Numerofclusters, title("Optimal Number of Clusters for Calinski-Harabasz Index", size(large)) ///
legend(label(1 "Homeowners") label(2 "Renters") label(3 "Social Housing") cols(3)) ///
 ytitle(" ") xtitle(,size(medlarge)) graphregion(color(white)) ylab(, nogrid labsize(medium)) xlabel(1(1)8 ,labsize(large)) 
graph export CH_all.pdf, replace


twoway line Elbow1 Numerofclusters || line Elbow2 Numerofclusters || line Elbow3 Numerofclusters, title("Optimal Number of Clusters for Elbow Method", size(large)) ///
legend(label(1 "Homeowners") label(2 "Renters") label(3 "Social Housing") cols(3)) ///
 ytitle(" ") xtitle(,size(medlarge)) graphregion(color(white)) ylab(, nogrid labsize(medium)) xlabel(1(1)8 ,labsize(large)) 
graph export elbow_all.pdf, replace
