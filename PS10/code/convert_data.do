clear all

cd "/Users/jonkroah/Documents/GitHub/computational-econ-899/PS10"

use "reference/Car_demand_characteristics_spec1.dta", clear
export delimited "data/Car_demand_characteristics_spec1.csv", delim(",") replace

use "reference/Car_demand_iv_spec1.dta", clear
export delimited "data/Car_demand_iv_spec1.csv", delim(",") replace

use "reference/Simulated_type_distribution.dta", clear
export delimited "data/Simulated_type_distribution.csv", delim(",") replace
