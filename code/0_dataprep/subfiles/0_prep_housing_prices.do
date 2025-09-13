* ------------------------------------------------------------------------------
* Project: Endogenous Amenities and Location Sorting
* Purpose: Prepare exported data on rents and transaction values
* Author: Milena Almagro
* ------------------------------------------------------------------------------


* Save rents *
cd "$DROPBOX_AMENITIES/data/cbs/exports/housing_prices"
import excel "House and Rent Priced by gebied - February 2022", sheet("Imputed Rent") first clear
keep year gb net_rent* median*
cd "$DROPBOX_AMENITIES/data/constructed/"
save imputed_rent, replace
export delimited imputed_rent, replace

* Save transaction values*
cd "$DROPBOX_AMENITIES/data/cbs/exports/housing_prices"
import excel "House and Rent Priced by gebied - February 2022", sheet("Transaction Values") first clear
keep year gb transaction* median*
cd "$DROPBOX_AMENITIES/data/constructed/"
save transaction_values, replace
