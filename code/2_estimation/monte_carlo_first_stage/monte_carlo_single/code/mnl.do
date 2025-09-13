* Set working directory and create log file
cd "../Data"

* Process each data folder
local folders : dir "." dirs "simulated_choices_full_dynamic_J*"

foreach folder of local folders {
    display _n "Processing `folder'"
    
    * Get the CSV file for iteration 1
    local files : dir "`folder'" files "*iter1.csv"
    
    foreach file of local files {
        capture {
            * Load and prepare data
            import delimited "`folder'/`file'", clear varnames(1)
            
            * Generate squared term for tau
            gen tau_init_squared = tau_init^2
            
            * Estimate MNL model
            mlogit j_prime i.j_init i.t tau_init tau_init_squared, noconstant baseoutcome(25)
            
            * Generate predictions and clean
            drop i j_prime tau_prime
            duplicates drop
            predict p*
            
            * Save results
            export delimited "`folder'/`file'", replace
            
            * Save model results
            estimates save "../Results/mnl_`folder'_`file'", replace
        }
        if _rc {
            display "Error processing `file' in `folder'"
        }
        clear
    }
}

log close