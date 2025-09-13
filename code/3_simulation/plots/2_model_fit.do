* ------------------------------------------------------------------------------
* Project: Endogenous Amenities
* Purpose: Create model fit plots
* Author: Sriram Tolety
* ------------------------------------------------------------------------------
set scheme s2mono // Set color scheme for graphs
graph set eps fontface "Palatino"

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
    display as error "Error: Two arguments required. Usage: stata-se/mp -e do 2_model_fit.do <Elasticity> <Bootstrap Flag>"
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

cd  "$DROPBOX_AMENITIES/data/simulation_results/`gamma_part'/equilibrium_objects/"


set scheme s2mono
clear




* ------------------------------------------------------------------------------
* 2017
* ------------------------------------------------------------------------------

* ------------------------------------------------------------------------------
* Rent
* ------------------------------------------------------------------------------


import delimited initial_r, clear
rename v1 r_observed_2017
gen gb = _n
tempfile r_observed
save `r_observed', replace

clear
import delimited r_endo, clear
rename v1 r_2017
gen gb = _n 
merge 1:1 gb using `r_observed', nogenerate
sort r_observed_2017

corr r_2017 r_observed_2017
local corr = round(r(rho), .001)

label variable r_2017 "Simulated Rent, 2017"
label variable r_observed_2017 "Observed Rent, 2017"

reg r_2017 r_observed_2017
local b: di %04.3f = _b[r_observed_2017]
local se: di %04.3f = _se[r_observed_2017]
local r2: di %04.3f = e(r2)

graph set eps fontface "Palatino Linotype"
twoway (scatter r_2017 r_observed_2017, msymbol(circle)) (lfitci r_2017 r_observed_2017, fcolor(%20) acolor(%5) fitplot(line) lpattern(_)),  graphregion(color(white))  xlabel(100(25)250) legend(off order( 2 "95% Confidence Interval" 3 "Linear Best Fit")) ytitle("Simulated Rent, 2017") xtitle("Observed Rent, 2017") xlabel(110(20)250) ylabel(, nogrid) subtitle("0`corr' correlation", size(medlarge)) text(300 110 "Slope: `b' (`se')" "R2: `r2'", place(right) justification(left) size(large))  title("Simulated vs. Observed Rents, 2017")


graph export ../../../../output/figures/model_fit/r_2017_scatter.pdf, replace

* ------------------------------------------------------------------------------
* Airbnb Prices
* ------------------------------------------------------------------------------
import delimited initial_p, clear
rename v1 p_observed_2017
gen gb = _n
tempfile p_observed
save `p_observed', replace

clear
import delimited p_endo, clear
rename v1 p_2017
gen gb = _n 
merge 1:1 gb using `p_observed', nogenerate
sort p_observed_2017

corr p_2017 p_observed_2017
local corr = round(r(rho), .001)

label variable p_2017 "Simulated STR Prices, 2017"
label variable p_observed_2017 "Observed STR Prices, 2017"

reg p_2017 p_observed_2017
local b: di %04.3f = _b[p_observed_2017]
local se: di %04.3f = _se[p_observed_2017]
local r2: di %04.3f = e(r2)

graph set eps fontface "Palatino Linotype"
twoway (scatter p_2017 p_observed_2017, msymbol(circle)) (lfitci p_2017 p_observed_2017, fcolor(%20) acolor(%5) fitplot(line) lpattern(_)),  graphregion(color(white))  xlabel(100(25)200) legend(off order( 2 "95% Confidence Interval" 3 "Linear Best Fit")) ytitle("Simulated STR Daily Price, 2017") xtitle("Observed STR Daily Price, 2017") ylabel(, nogrid) subtitle("0`corr' correlation", size(medlarge)) text(200 100 "Slope: `b' (`se')" "R2: `r2'", place(right) justification(left) size(large))  title("Simulated vs. Observed STR Prices, 2017")

graph export ../../../../output/figures/model_fit/p_2017_scatter.pdf, replace


* ------------------------------------------------------------------------------
* Amenities
* ------------------------------------------------------------------------------
import delimited initial_a, clear
rename v1 touristic_observed_2017
rename v2 restaurants_observed_2017
rename v3 bars_observed_2017
rename v4 food_observed_2017
rename v5 nonfood_observed_2017
rename v6 nurseries_observed_2017

tempfile a_observed
gen n = _n
save `a_observed'

clear
import delimited a_endo, clear
rename v1 touristic_simulated_2017
rename v2 restaurants_simulated_2017
rename v3 bars_simulated_2017
rename v4 food_simulated_2017
rename v5 nonfood_simulated_2017
rename v6 nurseries_simulated_2017
gen n = _n 
merge 1:1 n using `a_observed', nogenerate

* Touristic *
corr touristic_simulated_2017 touristic_observed_2017
local corr: di %4.3f `r(rho)'

label variable touristic_simulated_2017 "Simulated Touristic Amenities, 2017"
label variable touristic_observed_2017 "Observed Touristic Amenities, 2017"

reg touristic_simulated_2017 touristic_observed_2017
local b: di %04.3f = _b[touristic_observed_2017]
local se: di %04.3f = _se[touristic_observed_2017]
local r2: di %04.3f = e(r2)

graph set eps fontface "Palatino Linotype"
twoway (scatter touristic_simulated_2017 touristic_observed_2017, msymbol(circle)) (lfitci touristic_simulated_2017 touristic_observed_2017, fcolor(%20) acolor(%5) lpattern(_)),  graphregion(color(white)) xlabel(0(500)1500, labsize(medlarge)) ylabel(0(500)2000, labsize(medlarge))  legend(off order( 2 "95% Confidence Interval" 3 "Linear Best Fit")) xtitle("Observed Touristic Amenities, 2017", size(large)) ytitle("Simulated Touristic Amenities, 2017", size(large)) ylabel(, nogrid) text(1750 0 "Slope: `b' (`se')" "R2: `r2'", place(right) justification(left) size(medlarge)) title("Simulated vs. Observed Touristic Amenities, 2017", size(vlarge))
graph export ../../../../output/figures/model_fit/touristic_2017_scatter.pdf, replace







* Restaurants *
corr restaurants_simulated_2017 restaurants_observed_2017
local corr: di %4.3f `r(rho)'

label variable restaurants_simulated_2017 "Simulated Restaurants, 2017"
label variable restaurants_observed_2017 "Observed Restaurants, 2017"

reg restaurants_simulated_2017 restaurants_observed_2017
local b: di %04.3f = _b[restaurants_observed_2017]
local se: di %04.3f = _se[restaurants_observed_2017]
local r2: di %04.3f = e(r2)

graph set eps fontface "Palatino Linotype"
twoway (scatter restaurants_simulated_2017 restaurants_observed_2017, msymbol(circle)) (lfitci restaurants_simulated_2017 restaurants_observed_2017, fcolor(%20) acolor(%5) lpattern(_)),  graphregion(color(white)) xlabel(0(200)600, labsize(medlarge)) ylabel(0(200)600, labsize(medlarge)) legend(off order( 2 "95% Confidence Interval" 3 "Linear Best Fit")) xtitle("Observed Restaurants, 2017", size(large)) ytitle("Simulated Restaurants, 2017", size(large)) ylabel(, nogrid) text(525 0 "Slope: `b' (`se')" "R2: `r2'", place(right) justification(left) size(large))  title("Simulated vs. Observed Restaurants, 2017", size(vlarge))

graph export ../../../../output/figures/model_fit/restaurants_2017_scatter.pdf, replace






* Bars
corr bars_simulated_2017 bars_observed_2017
local corr: di %4.3f `r(rho)'

label variable bars_simulated_2017 "Simulated Bars, 2017"
label variable bars_observed_2017 "Observed Bars, 2017"

reg bars_simulated_2017 bars_observed_2017
local b: di %04.3f = _b[bars_observed_2017]
local se: di %04.3f = _se[bars_observed_2017]
local r2: di %04.3f = e(r2)

graph set eps fontface "Palatino Linotype"
twoway (scatter bars_simulated_2017 bars_observed_2017, msymbol(circle)) (lfitci bars_simulated_2017 bars_observed_2017, fcolor(%20) acolor(%5) lpattern(_)),  graphregion(color(white))  legend(off order( 2 "95% Confidence Interval" 3 "Linear Best Fit")) ytitle("Simulated Bars, 2017") ylabel(, nogrid) subtitle("`corr' correlation", size(medlarge)) text(300 0 "Slope: `b' (`se')" "R2: `r2'", place(right) justification(left) size(large))  title("Simulated vs. Observed Bars, 2017")

graph export ../../../../output/figures/model_fit/bars_2017_scatter.pdf, replace


* Food *
corr food_simulated_2017 food_observed_2017
local corr: di %4.3f `r(rho)'

label variable food_simulated_2017 "Simulated Food stores, 2017"
label variable food_observed_2017 "Observed Food stores, 2017"

reg food_simulated_2017 food_observed_2017
local b: di %04.3f = _b[food_observed_2017]
local se: di %04.3f = _se[food_observed_2017]
local r2: di %04.3f = e(r2)

graph set eps fontface "Palatino Linotype"
twoway (scatter food_simulated_2017 food_observed_2017, msymbol(circle)) (lfitci food_simulated_2017 food_observed_2017, fcolor(%20) acolor(%5) lpattern(_)),  graphregion(color(white))  legend(off order( 2 "95% Confidence Interval" 3 "Linear Best Fit")) ytitle("Simulated Food stores, 2017") ylabel(, nogrid) subtitle("`corr' correlation", size(medlarge)) text(400 0 "Slope: `b' (`se')" "R2: `r2'", place(right) justification(left) size(large))  title("Simulated vs. Observed Food stores, 2017")

graph export ../../../../output/figures/model_fit/food_2017_scatter.pdf, replace



* Non Food *
corr nonfood_simulated_2017 nonfood_observed_2017
local corr: di %4.3f `r(rho)'

label variable nonfood_simulated_2017 "Simulated Non-nonfood stores, 2017"
label variable nonfood_observed_2017 "Observed Non-nonfood stores, 2017"

reg nonfood_simulated_2017 nonfood_observed_2017
local b: di %04.3f = _b[nonfood_observed_2017]
local se: di %04.3f = _se[nonfood_observed_2017]
local r2: di %04.3f = e(r2)

graph set eps fontface "Palatino Linotype"
twoway (scatter nonfood_simulated_2017 nonfood_observed_2017, msymbol(circle)) (lfitci nonfood_simulated_2017 nonfood_observed_2017, fcolor(%20) acolor(%5) lpattern(_)),  graphregion(color(white))  legend(off order( 2 "95% Confidence Interval" 3 "Linear Best Fit")) ytitle("Simulated Non-nonfood stores, 2017") ylabel(, nogrid) subtitle("`corr' correlation", size(medlarge)) text(1500 0 "Slope: `b' (`se')" "R2: `r2'", place(right) justification(left) size(large))  title("Simulated vs. Observed Non-nonfood stores, 2017")

graph export ../../../../output/figures/model_fit/nonfood_2017_scatter.pdf, replace


* Nurseries *
corr nurseries_simulated_2017 nurseries_observed_2017
local corr: di %4.3f `r(rho)'

label variable nurseries_simulated_2017 "Simulated Nurseries, 2017"
label variable nurseries_observed_2017 "Observed Nurseries, 2017"

reg nurseries_simulated_2017 nurseries_observed_2017
local b: di %04.3f = _b[nurseries_observed_2017]
local se: di %04.3f = _se[nurseries_observed_2017]
local r2: di %04.3f = e(r2)

graph set eps fontface "Palatino Linotype"
twoway (scatter nurseries_simulated_2017 nurseries_observed_2017, msymbol(circle)) (lfitci nurseries_simulated_2017 nurseries_observed_2017, fcolor(%20) acolor(%5) lpattern(_)),  graphregion(color(white)) xlabel(250(500)1750)  legend(off order( 2 "95% Confidence Interval" 3 "Linear Best Fit")) ytitle("Simulated Nurseries, 2017") ylabel(, nogrid) subtitle("`corr' correlation", size(medlarge)) text(2500 250 "Slope: `b' (`se')" "R2: `r2'", place(right) justification(left) size(large))  title("Simulated vs. Observed Nurseries, 2017")

graph export ../../../../output/figures/model_fit/nurseries_2017_scatter.pdf, replace