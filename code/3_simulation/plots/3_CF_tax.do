* ------------------------------------------------------------------------------
* Project: Location Sorting and Endogenous Amenities
* Purpose: Plotting tax counterfactuals
* Author: Sriram Tolety
* ------------------------------------------------------------------------------

set scheme s2mono // Set color scheme for graphs
graph set eps fontface "Palatino"
ssc install grc1leg2

* Get the current directory
local current_dir = c(pwd)

* Go one level up by finding the position of the last '/'
local last_slash_pos = strrpos("`current_dir'", "/")
local parent_dir = substr("`current_dir'", 1, `last_slash_pos' - 1)

* Go two levels up by finding the position of the last '/' in the parent directory
local last_slash_pos_parent = strrpos("`parent_dir'", "/")
local grandparent_dir = substr("`parent_dir'", 1, `last_slash_pos_parent' - 1)

local last_slash_pos_grandparent = strrpos("`grandparent_dir'", "/")
local greatgrandparent_dir = substr("`grandparent_dir'", 1, `last_slash_pos_grandparent' - 1)

display "`grandparent_dir'"

* Set the global variable
global DROPBOX_AMENITIES "`grandparent_dir'"


args var1 var2

if "`var1'" == "" | "`var2'" == "" {
    display as error "Error: Two arguments required. Usage: stata-se/mp -e do 3_CF_tax.do <Elasticity> <Bootstrap Flag>"
    exit 1
}

local gamma_part
if "`var2'" == "True" {
    local gamma_part "gamma_B_`var1'"
}
else if "`var2'" == "False" {
    local gamma_part "gamma_`var1'"
}
else {
    display as error "Error: Second argument must be 'True' or 'False'."
    exit 1
}

cd  "$DROPBOX_AMENITIES/data/simulation_results/`gamma_part'/counterfactuals/"

set scheme s2mono
clear



* ------------------------------------------------------------------------------
* Short-term rental tax vs. Touristic amenity tax (welfare effects)
* ------------------------------------------------------------------------------

import delim CS_pp_airbnb_tax, clear
keep in 1/9
rename v1 tax
replace tax = (_n-1)*0.01
forval i =2/4 {
	rename v`i' v`i'_airbnb_city
}
save CSs_w, replace

import delim CS_pp_amenity_tax, clear
keep in 1/9
rename v1 tax
replace tax = (_n-1)*0.01
forval i =2/4 {
	rename v`i' v`i'_amenity_city
}
merge 1:1 tax using CSs_w
drop _m
save CSs_w, replace

cd "$DROPBOX_AMENITIES/output/figures/"

graph set eps fontface "Palatino Linotype"
twoway line v2_amenity_city v2_airbnb_city tax, graphregion(color(white))  ylabel(, nogrid) legend(pos(9) ring(0) label(1 "Amenities") label(2 "Airbnb") r(3) region(lwidth(none))) ytitle("Consumer Surplus (% income)") xtitle("Tax rate")  title("Older Families")

* graph export cs_w_tax_1_city_taxes.pdf, replace



graph set eps fontface "Palatino Linotype"
twoway line v3_amenity_city v3_airbnb_city tax, graphregion(color(white))  ylabel(, nogrid) legend(pos(9) ring(0) label(1 "Amenities") label(2 "Airbnb") r(3) region(lwidth(none))) ytitle("Consumer Surplus (% income)") xtitle("Tax rate")  title("Singles")

* graph export cs_w_tax_2_city_taxes.pdf, replace


graph set eps fontface "Palatino Linotype"
twoway line v4_amenity_city v4_airbnb_city tax, graphregion(color(white))  ylabel(, nogrid) legend(pos(9) ring(0) label(1 "Amenities") label(2 "Airbnb") r(3) region(lwidth(none))) ytitle("Consumer Surplus (% income)") xtitle("Tax rate")  title("Younger Families")

* graph export cs_w_tax_3_city_taxes.pdf, replace



graph set eps fontface "Palatino Linotype"
twoway line v2_amenity_city tax, ytitle("Amenities Consumer Surplus (% income)", axis(1)) graphregion(color(white))  ylabel(, nogrid) || line v2_airbnb_city tax, yaxis(2) ytitle("Airbnb Consumer Surplus (% income)", axis(2)) xtitle("Tax rate") graphregion(color(white))  ylabel(, nogrid) legend(pos(9) ring(0) label(1 "Amenities") label(2 "Airbnb") r(3) region(lwidth(none))) title("Older Families")


* graph export cs_w_tax_1_city_taxes_second_option.pdf, replace





cd "$DROPBOX_AMENITIES/output/figures/"

* ------------------------------------------------------------------------------
* Create Combined Graphs with Common Legend
* ------------------------------------------------------------------------------


* Reload the merged data
use "$DROPBOX_AMENITIES/data/simulation_results/`gamma_part'/counterfactuals/CSs_w.dta", clear

* Ensure data is sorted by tax for plotting
sort tax

forval i = 1/3 {
    local group_num = `i' + 1
    local current_label
    if `i' == 1 {
        local current_label "Older Families"
    }
    else if `i' == 2 {
        local current_label "Singles"
    }
    else if `i' == 3 {
        local current_label "Younger Families"
    }

    * Create Consumer Surplus (% income) Plots
    twoway (line v`group_num'_amenity_city tax) || ///
           (line v`group_num'_airbnb_city tax), ///
        ytitle("Consumer surplus (% income)", size(large)) ///
        xtitle("Tax rate", size(large)) ///
        legend(col(2) ring(0) ///
               label(1 "Tax on Touristic Amenities") ///
               label(2 "Tax on Short-term Rentals") ///
               size(vlarge)) ///
        graphregion(color(white)) ///
        title("`current_label'", size(huge)) ///
        xlabel(, labsize(large)) ///
        ylabel(, nogrid labsize(large)) ///
        name(tax_CF_group`i', replace)
}

grc1leg2 tax_CF_group1 tax_CF_group2 tax_CF_group3, ///
    row(1) ///
    xcommon ///
    graphregion(color(white)) ///
    ysize(6) ///
    xsize(15)
graph export tax_CF_group_combined.pdf, replace
