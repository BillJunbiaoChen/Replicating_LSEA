* ------------------------------------------------------------------------------
* Project: Endogenous amenities
* Purpose: prepare data for demand estimation of tourists
* Author: Milena Almagro
* ------------------------------------------------------------------------------

* ------------------------------------------------------------------------------
* Tourist demand estimation at the wijk level
* ------------------------------------------------------------------------------
set type double

cd "$DROPBOX_AMENITIES/data/constructed"
import delimited neighborhood_panel, clear
save neighborhood_panel, replace 

* Set directory
cd "$DROPBOX_AMENITIES/data/airbnb/"
import delimited listings_by_wk.csv, clear 

* Destring variables
replace price_mean = "." if price_mean == "NA"
replace mean_accommodates = "." if mean_accommodates == "NA"
replace review_scores_location = "." if review_scores_location == "NA"
replace airbnb_pop_commercial = "." if airbnb_pop_commercial == "NA"
replace airbnb_pop = "." if airbnb_pop == "NA"

destring price_mean, replace
destring mean_accommodates, replace
destring review_scores_location, replace
destring airbnb_pop_commercial, replace
destring airbnb_pop, replace

* Generate price per guest
gen price_mean_pp = price_mean/mean_accommodates

* Merge with neighborhood characteristics 
cd "$DROPBOX_AMENITIES/data/constructed"
merge 1:1 wk_code year using neighborhood_panel

* Take logs for regressions
foreach var of varlist tourism_offices_net_hotels bars_locations restaurants_locations food_stores nonfood_stores nurseries price_mean_pp review_scores_location {
	bysort year (wk_code): gen log_`var' = log(`var'*(`var'>0) + (`var'==0))
}

* Construct LHS
bysort year (wk_code): egen total_pop_hotels = sum(pop_tourists_hotels)
bysort year (wk_code): gen norm_daily_pop_airbnb = log(airbnb_pop_commercial)-log(total_pop_hotels)

* Establish panel structure
xtset wk_code year
tab year, gen(dyear)

////// Using mean /////

*Restrict to relevant years and observations with price info
keep if year >= 2015 & year <= 2018

eststo reg1: xtreg norm_daily_pop_airbnb dyear* log_price_mean_pp log_touri log_rest log_bars  log_food log_nonfood log_nurseries, fe cluster(wk_code)


eststo reg2: xtreg norm_daily_pop_airbnb dyear* log_price_mean_pp log_touri log_rest log_bars  log_food log_nonfood log_nurseries log_review_scores_location , fe cluster(wk_code)
esttab reg* using "$DROPBOX_AMENITIES/output/tables/tourist_demand_estimates.tex", ///
    replace wide drop(_cons dyear*) ///
    mtitles("Baseline" "Controlling for reviews") ///
    stats(N r2, fmt(%9.0fc %9.3f) labels("N" "R$^2$")) ///
    b(%5.3f) se(%5.3f) star(* 0.10 ** 0.05 *** 0.01) nonumbers ///
    rename(log_price_mean_pp "Log Price Per Guest" ///
           log_tourism_offices_net_hotels "Log Touristic Amenities" ///
           log_restaurants_locations "Log Restaurants" ///
           log_bars_locations "Log Bars" ///
           log_food_stores "Log Food Stores" ///
           log_nonfood_stores "Log Nonfood Stores" ///
           log_nurseries "Log Nurseries" ///
           log_review_scores_location "Log Review Scores") ///
    prehead("\begin{table}[!ht]" ///
            "\centering" ///
            "\caption{Tourist demand across locations.}\label{tab:tourist_demand}" ///
            "\scalebox{.85}{" ///
            "\def\sym#1{\ifmmode^{#1}\else\(^{#1}\)\fi}" ///
            "\begin{tabular}{l*{4}{c}}" ///
            "\toprule" ///
            "& \multicolumn{4}{c}{Dependent Variable: $\log \Prob^{ST}_{jt} - \log \Prob^{H}_t$} \\" ///
            "\cmidrule(l{3pt}r{3pt}){2-5}") ///
    posthead("\midrule") ///
    prefoot("\hline") ///
    postfoot("\bottomrule" ///
             "\end{tabular}" ///
             "}" ///
             "\legend{Table reports estimates of tourists' preference for neighborhood (wijk-level) characteristics for a static model of location choice, using neighborhood-level data for 2015-2018. Construction of Airbnb supply and prices is described in Appendix \ref{oa-sec: appendix data}. Wijk-level clustered standard errors in parenthesis. \sym{*}\(p<0.10\), \sym{**}\(p<0.05\), \sym{***}\(p<0.01\)." ///
             "}" ///
             "\end{table}") substitute("\_{" "_{")








gen y_hat = 0
foreach var of varlist log_price_mean_pp log_touri log_bars log_rest log_food log_nonfood log_nurseries log_review_scores_location {
	replace y_hat = y_hat + _b[`var']*`var'
	gen b_`var' = _b[`var']
}

* Save output
eststo mycoeftable
cd "$DROPBOX_AMENITIES/output/estimates"
esttab mycoeftable using "tourist_demand_estimates.csv", replace nogap cells(b) drop(dyear* _cons) noobs plain nomtitles
cd "$DROPBOX_AMENITIES/data/final/estimates"
esttab mycoeftable using "tourist_demand_estimates.csv", replace nogap cells(b) drop(dyear* _cons) noobs plain nomtitles

predict fe, u
predict res, e
predict u_hat, xbu

* Recover delta_j of j options
keep if year == 2017
gen delta_j = fe + res + _b[_cons] + _b[dyear3]

* Include outside option
insobs 1
replace delta_j = 0 if year == .
replace y_hat = 0 if year == .
replace wk_code = 100 if year == .
replace gb_code = 100 if year == .
replace year = 2017 if year == .

gen pop_tourist = airbnb_pop_commercial
sort gb_code 
replace pop_tourist = total_pop_hotels[_n-1] if gb_code == 100
replace norm_daily_pop_airbnb = 0 if gb_code == 100
replace mean_accommodates = 0 if gb_code == 100
replace price_mean = 0 if gb_code == 100
replace review_scores_location = 0 if gb_code == 100

* Rename all tourist population variables 
rename airbnb_pop pop_airbnb_total
rename airbnb_pop_commercial pop_airbnb_commercial 
rename pop_tourists_hotels pop_hotels 
drop pop_tourists_*

* Save at the gebied level
collapse delta_j b_* price_mean review_scores_location mean_accommodates (sum) commercial pop_* tourism_offices_net_hotels bars_locations restaurants_locations food_stores nonfood_stores nurseries, by(gb_code)

gen price_mean_pp = price_mean/mean_accommodates
replace price_mean_pp = 0 if gb_code == 100

* Take logs 
foreach var of varlist tourism_offices_net_hotels bars_locations restaurants_locations food_stores nonfood_stores nurseries price_mean_pp review_scores_location {

	gen log_`var' = log(`var'*(`var'>0) + (`var'==0))
}

* Recover delta_j at the gebied level 
sort gb_code
gen norm_daily_pop_airbnb = log(pop_tourist)-log(pop_tourist[_N])

gen y_hat_gebied = 0
foreach var of varlist tourism_offices_net_hotels bars_locations restaurants_locations food_stores nonfood_stores nurseries review_scores_location price_mean_pp {
	replace y_hat_gebied = y_hat_gebied + b_log_`var'*log_`var'
	
}

replace y_hat_gebied = 0 if gb_code == 100
gen delta_j_gebied = norm_daily_pop_airbnb-y_hat_gebied

* Generate conversion rate from guests to units
gen str_guests_to_str_units = commercial/pop_tourist
replace str_guests_to_str_units = 0 if gb_code == 100

* Generate conversion rate from commercial guests to total guests 
gen str_guests_to_total_guests = pop_airbnb_total/pop_tourist
replace str_guests_to_total_guests = 0 if gb_code == 100

egen total_tourist_pop = sum(pop_tourist)
gen prob_tourists = pop_tourist/total_tourist_pop

keep gb delta_j_gebied log_* y_hat_gebied *guest* norm_daily_pop_airbnb price_mean price_mean_pp mean_accommodates pop_* total_tourist_pop prob_tourists

cd "$DROPBOX_AMENITIES/data/final/inputs"
export delim gebied_tourist_demand_covariates, replace
