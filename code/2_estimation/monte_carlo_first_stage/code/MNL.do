*-------------------------------------------------------------------------------
* Project: Endogenous Amenities
* Purpose: compute multinomial logit model and predict probabilities on simulated choice data
* Author: Marek Bojko
*-------------------------------------------------------------------------------
* Install required packages

ssc install regsave,replace
ssc install parallel, replace

* Set working directory
cd "../Data/"

local folders : dir . dirs "simulated_choices_full_dynamic_J*"

foreach folder of local folders {
    di "Processing folder: `folder'"
    
    * Get a list of all .csv files in the current folder
    local files : dir "`folder'" files "*.csv"
    
    * Loop over each .csv file
    foreach file of local files {
        di "  Processing file: `file'"
		
		
		capture {
			* Read a file
			import delimited "`folder'/`file'", clear
			*import delimited "Z:\amenities_model\simulated_choices\simulated_choices_J25_Pop1050_iter1.csv", clear
			
			**** Fit a multi-nomial logit model

			* generate tau^2
			gen tau_init_squared = tau_init^2

			* Fit a multinomial logit model
			mlogit j_prime i.j_init i.t tau_init tau_init_squared, noconstant baseoutcome(25)

			* Compute predicted probabilities
			drop i j_prime tau_prime
			duplicates drop
			predict p*
			
			* Store back in the file
			export delimited "`folder'/`file'", replace

			* Store estimated coefficients 
			*regsave using `file', replace
		}
		
        
		
        clear
    }
}