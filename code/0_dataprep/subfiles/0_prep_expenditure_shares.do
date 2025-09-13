* ------------------------------------------------------------------------------
* Project: Endogenous Amenities and Location Sorting
* Purpose: Prepare exported data on expenditure shares at the group-year level
* Author: Milena Almagro
* ------------------------------------------------------------------------------

cd "$DROPBOX_AMENITIES/data"
import excel "cbs/exports/expenditure_shares/Expenditure Shares - April 2022", first clear

keep year gb combined exp_sh_hh

* Add three groups (2 social housing + tourists)
forval g = 5/7{
	forval j = 1/22{
		forval y = 2008/2019{
			set obs `=_N+1'
			replace combined = `g' if combined == .
			replace gb = `j' if gb == .
			replace year = `y' if year == .
		} 
	}
}

* Merge to annual income 
merge m:1 year combined using "constructed/annual_income"

* Replace value of social housing
replace exp_sh_hh = 12*631/disposable_income if combined >= 5 & combined <= 6 & year == 2008
replace exp_sh_hh = 12*647/disposable_income if combined >= 5 & combined <= 6 & year == 2009
replace exp_sh_hh = 12*647/disposable_income if combined >= 5 & combined <= 6 & year == 2010
replace exp_sh_hh = 12*652/disposable_income if combined >= 5 & combined <= 6 & year == 2011
replace exp_sh_hh = 12*664/disposable_income if combined >= 5 & combined <= 6 & year == 2012
replace exp_sh_hh = 12*681/disposable_income if combined >= 5 & combined <= 6 & year == 2013
replace exp_sh_hh = 12*699/disposable_income if combined >= 5 & combined <= 6 & year == 2014
replace exp_sh_hh = 12*710/disposable_income if combined >= 5 & combined <= 6 & year == 2015
replace exp_sh_hh = 12*710/disposable_income if combined >= 5 & combined <= 6 & year == 2016
replace exp_sh_hh = 12*710/disposable_income if combined >= 5 & combined <= 6 & year == 2017
replace exp_sh_hh = 12*710/disposable_income if combined >= 5 & combined <= 6 & year == 2018

* Tourists caibrated from Loon and Rouwendal (2017)
* https://link.springer.com/article/10.1007/s10824-017-9293-1
replace exp_sh_hh = 1-0.5164 if combined == 7

gen exp_sh_c = 1-exp_sh_hh

* Collapse at the group level
collapse exp_sh_hh exp_sh_c, by(combined)
rename exp_sh_hh exp_sh

* Save
save constructed/expenditure_shares, replace
export delimited final/inputs/expenditure_shares, replace
