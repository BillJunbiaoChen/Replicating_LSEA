* ------------------------------------------------------------------------------
* Project: Endogenous amenities
* Purpose: Generate robustness of equilibrium plots
* Author: Sriram Tolety
* ------------------------------------------------------------------------------

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
    display as error "Error: Two arguments required. Usage: stata-se/mp -e do stability_plots.do <Elasticity> <Bootstrap Flag>"
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

local full_path "$DROPBOX_AMENITIES/data/simulation_results/`gamma_part'/stability/amenities"
cd "`full_path'"


*cd  "$DROPBOX_AMENITIES/data/simulation_results/gamma_B_152/stability/amenities"

set scheme s2mono
clear



import delim stability_averages_median, rowrange(:6) clear
save CSs_w, replace


* Summarize all relevant y-variables to find global min and max
summarize v1 v2 v3 v4 v5 v6 v7 v8 v9, meanonly
local y_min = r(min)
local y_max = r(max)

local y_padding = (`y_max' - `y_min') * 0.05
local y_min_padded = `y_min' - `y_padding'
local y_max_padded = `y_max' + `y_padding'

* ------------------------------------------------------------------------------
* Amenities
* ------------------------------------------------------------------------------
graph set eps fontface "Palatino Linotype"
twoway line v2 v3 v4 v5 v6 v7 v1, graphregion(color(white))  ylabel(0(.2)1, nogrid) legend(pos(10) ring(0) label(1 "Touristic Amenities") label(2 "Restaraunts") label(3 "Bars") label(4 "Food stores") label(5 "Non-food stores") label(6 "Nurseries") r(3) region(lwidth(none))) ytitle("Max % Difference to Baseline Equilibrium", size(large)) xtitle("% Disturbance", size(large))  title("Equilbrium Amenity Value Stability", size(vlarge))

graph export "$DROPBOX_AMENITIES/output/figures/eq_amenity_stability_median.png", replace



* ------------------------------------------------------------------------------
* Airbnb
* ------------------------------------------------------------------------------
graph set eps fontface "Palatino Linotype"
twoway line v8 v1, graphregion(color(white))  ylabel(0(.2)1, nogrid) legend(pos(9) ring(0) label(1 "Airbnb Prices") r(3) region(lwidth(none))) ytitle("Max % Difference to Baseline Equilibrium", size(large)) xtitle("% Disturbance", size(large))  title("Equilbrium STR Price Stability", size(vlarge)) yscale(range(0 1))

graph export "$DROPBOX_AMENITIES/output/figures/eq_airbnb_stability_median.png", replace





* ------------------------------------------------------------------------------
* Rent
* ------------------------------------------------------------------------------
graph set eps fontface "Palatino Linotype"
twoway line v9 v1, graphregion(color(white))  ylabel(0(.2)1, nogrid) legend(pos(9) ring(0) label(1 "Rental Prices") r(3) region(lwidth(none))) ytitle("Max % Difference to Baseline", size(large)) xtitle("% Disturbance", size(large))  title("Equilbrium Rental Price Stability", size(vlarge)) yscale(range(0 1))

graph export "$DROPBOX_AMENITIES/output/figures/eq_rent_stability_median.png", replace




* ------------------------------------------------------------------------------
* Airbnb + Rent
* ------------------------------------------------------------------------------
graph set eps fontface "Palatino Linotype"
twoway line v8 v9 v1, graphregion(color(white))  ylabel(0(.2)1, nogrid) legend(pos(9) ring(0) label(1 "STR Prices") label(2 "Rental Prices") r(3) region(lwidth(none))) ytitle("Max % Difference to Baseline Equilibrium", size(large)) xtitle("% Disturbance", size(large))  title("Equilbrium STR and Rental Price Stability", size(vlarge)) yscale(range(0 1))

graph export "$DROPBOX_AMENITIES/output/figures/eq_prices_stability_median.png", replace




