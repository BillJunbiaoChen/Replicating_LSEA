# Guide to MC Parameter Estimation Folder

This folder contains all code needed to run a MC-based parameter estimation  as detailed in the Amenities ECMA Paper. 

The code is organized as follows:

* /code: This contains all the folders and julia / stata files required to run the codebase
* /Data: All generated choice data is stored within this folder.
* /Output: This contains all files that contain the estimated paraemters, as well as the bias of the parameters as calcualted. It also contains the .tex files which store the final tables for display in LaTeX documents.


**/code**

* Generate_data/: Contains all files to run the generation of choice data. Each file is self-contained, and generates choice data for a certain set of parameters as described in its filename. 
* Main_functions/: Contains files needed to estimate parameters from choice data, which is read in from ~/Data/

**The run_all.sh script takes care of the entire process end-to-end and will store the final LaTeX outputs in ~/Output, as descriebd previously.**

**/Data**

* Stores all choice data in various forms:
    * Folders containing **true_param** store the true parameters used to derive the MC
    * Folders containing **true_prob_mat** store the true probability values
    * Folders containing **raw_choices** store the raw simulated choice data
    * Folders containing **freq_probs** store the frequency-based simulation as described in the paper.

**/Output**

* This folder contains all the outputs used in the entire codebase
* The final bias clacualtions are stored in .csv files containing the term **avg_bias_experiments**
* The .tex files contain LaTeX code for generating formatted LaTeX tables with the bias results, for each experiment type across experiment parameters.


**How to Use *run_all.sh***

1. Change line 25 of the script to your correct filepath to the /monte_carlo_first_stage/code folder
2. Open a terminal window and navigate to the directory containing the script.
3. Run the script using `./run_all.sh`.
4. The outputs will be stored in the **/Output** folder as described before.

**Note**

* The *run_all.sh* script assumes the directory structure and script file locations are as described.
* You can modify the script to handle errors or change the order of execution as needed.
