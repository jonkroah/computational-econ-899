clear all
set more off
set type double

cd "/Users/jonkroah/Documents/GitHub/computational-econ-899/PS08"

//------- Copy data to CSV
use "data/Mortgage_performance_data.dta", clear
//export excel "data/Mortgage_performance_data.xlsx", firstrow(var) replace

//------- Estimate probit
/*
Outcome:
	i_close_first_year 	// 	=1 if loan closed during year 0 or 1
Covariates:
	i_large_loan 		// 	large loan
	i_medium_loan 		// 	medium loan
	rate_spread 		// 	mortgage interest rate spread at origination
	i_refinance 		// 	=1 if refinanced?
	age_r 				// 	age at last birthday
	cltv 				// 	combined loan to value (LTV)
	dti 				// 	mortgage debt-to-income
	cu 					// 	credit union 
	first_mort_r 		// 	=1 if first mortgage in credit file
	score_0 			// 	FICO score at origination
	score_1 			// 	FICO score when loan age = 1
	i_FHA 				// 	=1 if FHA loan?
	i_open_year2 		// 	open year == 2014
	i_open_year3 		// 	open year == 2015
	i_open_year4 		// 	open year == 2016
	i_open_year5 		// 	open year == 2017
*/

loc y_var i_close_first_year

loc x_vars i_large_loan i_medium_loan rate_spread i_refinance age_r cltv dti cu first_mort_r score_0 score_1 i_FHA i_open_year2 i_open_year3 i_open_year4 i_open_year5

loc options tex(frag) excel dec(3) alpha(0.01, 0.05, 0.10) label
cd "output"

logit `y_var' `x_vars'
outreg2 using ps08_stata_results, `options' ctitle(Logit) replace
