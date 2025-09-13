###############################################
## Project: Endogenous Amenities and Location Sorting
## Purpose: Stylized facts on housing outcomes
## Author: Milena Almagro & Tomas Dominguez-Iino; Tables: Sriram Tolety
###############################################

library(pacman)
library(modelsummary)
packageVersion("modelsummary")
options(warn = -1)
pacman::p_load(rstudioapi, tidyverse, broom, data.table, 
               readstata13, readxl, foreign,
               lfe, fixest, ivreg, kableExtra)


setwd(dirname(getwd()))
print(getwd())
rm(list = setdiff(ls(), "wd"))
`%!in%` <- Negate(`%in%`)
options("modelsummary_format_numeric_latex" = "plain")


################## Import data

#Public neighborhood-level data
df1 = fread("../data/bbga/bbga.csv")
df2 = fread("../data/airbnb/listings_by_wk.csv")
setnames(df2, old=c('total'), new=c('total_listings'))
df3 = fread("../data/bbga/monument_count.csv")
df4 = fread("../data/googletrends/google_trends_annual.csv")
setnames(df4, old=c('mean'), new=c('google'))

dfp = merge(df1,df2, by=c('year','wk_code'))[,.(year, wk_code, sd_code,
                                         total_listings, commercial,
                                         addresses_total, addresses_private_rental,
                                         pop_total, pop_high_skill_p, average_income,
                                         addresses_total_imp, addresses_private_rental_imp,
                                         pop_total_imp, pop_high_skill_p_imp, average_income_imp)]
dfp = merge(dfp,df3, by=c('wk_code'))
dfp = merge(dfp,df4, by=c('year'))

#Exported CBS neighborhood-level data
df1 = setDT(read_excel('../data/cbs/exports/LIGHT_rent_house_transactions_aangepast - 9.23.2021/LIGHT_rent_house_transactions_aangepast.xlsx', sheet=2))
df2 = setDT(read_excel('../data/cbs/exports/LIGHT_rent_house_transactions_aangepast - 9.23.2021/LIGHT_rent_house_transactions_aangepast.xlsx', sheet=3))
df3 = setDT(read_excel('../data/cbs/exports/LIGHT_rent_house_transactions_aangepast - 9.23.2021/LIGHT_rent_house_transactions_aangepast.xlsx', sheet=4))
df =  merge(df1,df2, by=c('year','wk'))
df =  merge(df,df3, by=c('year','wk'))
setnames(df[, c("num_obs","num_obs_rent","num_obs_rent_hat"):=NULL], old=c('wk'), new=c('wk_code'))
                    
#Merge 
dfp = merge(df,dfp, by=c('year','wk_code'))
dfp$year_sd_code = with(dfp, interaction(year,sd_code))


################## Summary statistics 

dfs = df
dfs$rent = dfs$net_rent_meter_hat_rf
dfs = dfs[!is.na(dfs$rent), .(rent_md=median(rent), rent_mn=mean(rent) ), by=year]
dfs2 = dfs %>% mutate(growth_rate_percent_mn = 100*(rent_mn - lag(rent_mn))/rent_mn )
annualized_growth_rate_2_mn = (dfs[year==max(dfs$year), rent_mn]/dfs[year==min(dfs$year), rent_mn])^(1/(max(dfs$year)-min(dfs$year)))-1


################## Regressions

start_year = 2008
end_year = 2019
df = dfp[year>= start_year & year <= end_year,]
df = df[wk_code %!in% c(10,11,50,51),]

df$x = log(df$commercial)
df$x1 = log(df$addresses_total_imp)
df$x2 = log(df$average_income_imp)
df$x3 = log(df$pop_high_skill_p_imp)
df$z = log((df$google+.1)*df$monuments)
df = df[is.finite(x),]


####### Rent

df$y = log(df$net_rent_meter_hat_rf)

#OLS
rent_ols = feols(y ~ x, cluster = c('wk_code'), data = df)
rent_ols_x = feols(y ~ x + x1+x2+x3 , cluster = c('wk_code'), data = df)
rent_ols_yxsd = feols(y ~ x + x1+x2+x3| year_sd_code, cluster = c('wk_code'), data = df)

#IV
rent_iv = feols(y ~ 1 | x~z , cluster = c('wk_code'), data = df)
rent_iv_x = feols(y ~ x1+x2+x3 | x~z, cluster = c('wk_code'), data = df)
rent_iv_yxsd = feols(y ~ x1+x2+x3| year_sd_code | x~z , cluster = c('wk_code'), data = df)


####### Home sale values

df$y = log(df$mean_transaction_value)

#OLS
value_ols = feols(y ~ x, cluster = c('wk_code'), data = df)
value_ols_x = feols(y ~ x + x1+x2+x3, cluster = c('wk_code'), data = df)
value_ols_yxsd = feols(y ~ x + x1+x2+x3| year_sd_code, cluster = c('wk_code'), data = df)

#IV
value_iv = feols(y ~ 1 | x~z , cluster = c('wk_code'), data = df)
value_iv_x = feols(y ~ x1+x2+x3 | x~z, cluster = c('wk_code'), data = df)
value_iv_yxsd = feols(y ~ x1+x2+x3| year_sd_code | x~z , cluster = c('wk_code'), data = df)


####### Export OLS + IV tables

# Rent
F1 = as.character(round(fitstat(rent_iv, "ivf", simplify = T)$stat, digits=2))
F2 = as.character(round(fitstat(rent_iv_x, "ivf", simplify = T)$stat, digits=2))
F3 = as.character(round(fitstat(rent_iv_yxsd, "ivf", simplify = T)$stat, digits=2))

rows <- tribble(~term,                 ~OLS, ~IV, ~OLS, ~IV, ~OLS, ~IV,
                'District-year FE'  ,  '', '', '', '','X' , 'X',
                'First stage F-stat',  '', F1, '', F2,'', F3)
attr(rows, 'position') <- c(9,10,11,12)

gm <- tibble::tribble(
  ~raw,        ~clean,          ~fmt,
  "nobs",      "Observations",  0,
  "adj.r.squared", "R2", 3,
)

tab_iv_rent <- msummary(list('OLS'=rent_ols, 'IV'=rent_iv, 'OLS'=rent_ols_x, 'IV'=rent_iv_x, 'OLS'= rent_ols_yxsd, 'IV'= rent_iv_yxsd),
                        coef_map = c('x'   = 'Ln (commercial Airbnb listings)',
                        'fit_x' ='Ln (commercial Airbnb listings)',
                        'x1'   = 'Ln (housing stock)',
                        'x2'   = 'Ln (average income)',
                        'x3'   = 'Ln (high-skill population share)' ),
                        estimate = "{estimate}{stars}",
                        gof_map = gm, escape=F,
                        add_rows = rows,
                        output='latex')  %>%
  add_header_above(c(" " = 1, "Ln (rent/m2)" = 6)) %>%
  row_spec(1, bold = T)
kableExtra::save_kable(tab_iv_rent, file = "../output/tables/reducedform_iv_rent.tex")


# Home sale values
F1 = as.character(round(fitstat(value_iv, "ivf", simplify = T)$stat, digits=2))
F2 = as.character(round(fitstat(value_iv_x, "ivf", simplify = T)$stat, digits=2))
F3 = as.character(round(fitstat(value_iv_yxsd, "ivf", simplify = T)$stat, digits=2))
rows <- tribble(~term,                 ~OLS, ~IV, ~OLS, ~IV, ~OLS, ~IV,
                'District-year FE'  ,  '', '', '', '','X' , 'X',
                'First stage F-stat',  '', F1, '', F2,'', F3)
attr(rows, 'position') <- c(9,10,11,12)

gm <- tibble::tribble(
  ~raw,        ~clean,          ~fmt,
  "nobs",      "Observations",  0,
  "adj.r.squared", "R2", 3,
)

tab_iv_value <- msummary(list('OLS'=value_ols, 'IV'=value_iv, 'OLS'=value_ols_x, 'IV'=value_iv_x, 'OLS'= value_ols_yxsd, 'IV'= value_iv_yxsd),
                         coef_map = c('x'   = 'Ln (commercial Airbnb listings)',
                                     'fit_x' ='Ln (commercial Airbnb listings)',
                                     'x1'   = 'Ln (housing stock)',
                                     'x2'   = 'Ln (average income)',
                                     'x3'   = 'Ln (high-skill population share)' ),
                        estimate = "{estimate}{stars}",
                        gof_map = gm, escape=F,
                        add_rows = rows,
                        output='latex')  %>%
  add_header_above(c(" " = 1,"Ln (house sale price)" = 6)) %>%
  row_spec(1, bold = T)
kableExtra::save_kable(tab_iv_value, file = "../output/tables/reducedform_iv_salevalue.tex")



process_subtable <- function(model_list, dep_var, tex_file) {
  # Read F-stats from the .tex file
  f_stats <- c("", "", "")
  tex_content <- readLines(tex_file)
  
  # Extract F-stats
  f_stat_line <- grep("First stage F-stat", tex_content, value = TRUE)
  if (length(f_stat_line) > 0) {
    extracted_stats <- stringr::str_extract_all(f_stat_line, "\\d+\\.?\\d*")[[1]]
    if (length(extracted_stats) >= 3) {
      f_stats <- extracted_stats[1:3]
    }
  }
  
  
  rows <- tribble(
    ~term,                 ~`1`, ~`2`, ~`3`, ~`4`, ~`5`, ~`6`,
    'Control variables',   '',   '',   'X',  'X',  'X',  'X',
    'District-year FE',    '',   '',   '',   '',   'X',  'X',
    'First stage F-stat',  '',   as.character(f_stats[1]), '', as.character(f_stats[2]), '', as.character(f_stats[3]),
    'Observations',        as.character(model_list[[1]]$nobs), as.character(model_list[[2]]$nobs), 
                           as.character(model_list[[3]]$nobs), as.character(model_list[[4]]$nobs), 
                           as.character(model_list[[5]]$nobs), as.character(model_list[[6]]$nobs)
  )
  
  subtable <- msummary(model_list,
           coef_map = c('x' = 'Ln (commercial Airbnb listings)',
                        'fit_x' ='Ln (commercial Airbnb listings)'),
           estimate = "{estimate}{stars}",
           statistic = "std.error",
           gof_omit = ".*",
           escape = F,
           output = 'latex_tabular')
  
  # Extract only the body of the table
  body_lines <- strsplit(subtable, "\n")[[1]]
  body_lines <- body_lines[!(grepl("\\\\begin|\\\\end|\\\\toprule|\\\\bottomrule", body_lines))]
  body_lines[4] <- paste(body_lines[4], body_lines[5])
  
  # Add the header
  header <- paste0("\\multicolumn{1}{c}{ } & \\multicolumn{6}{c}{", dep_var, "} \\\\")
  body_lines <- c(header, "\\cmidrule(l{3pt}r{3pt}){2-7}", 
                  " & OLS & IV & OLS & IV & OLS & IV \\\\",
                  "\\midrule",
                  body_lines[4], "\\midrule")
  
  # Make the coefficient row bold and fix asterisks
  coef_line <- which(grepl("Ln \\(commercial Airbnb listings\\)", body_lines))
  body_lines[coef_line] <- gsub("Ln", "\\\\textbf{Ln", body_lines[coef_line])
  body_lines[coef_line] <- gsub("listings)", "listings)}", body_lines[coef_line])
  body_lines[coef_line] <- gsub("\\$", "}", body_lines[coef_line])
  body_lines[coef_line] <- gsub("\\*\\*\\*", "***", body_lines[coef_line])
  body_lines[coef_line] <- gsub("\\*\\*", "**", body_lines[coef_line])
  body_lines[coef_line] <- gsub("\\*", "*", body_lines[coef_line])
  body_lines[coef_line] <- gsub("\\+", "*", body_lines[coef_line])
  body_lines[coef_line] <- gsub("(\\d+\\.\\d+)([*]+)", "\\\\textbf{\\1\\2}", body_lines[coef_line])
  
  # Add additional rows
  for (i in 1:nrow(rows)) {
    row_elements <- as.character(rows[i, -1])
    last_element <- trimws(tail(row_elements, 1))  # Get the last element and remove trailing whitespace
    other_elements <- head(row_elements, -1)  # Get all but the last element
    
    new_row <- paste0(
      rows$term[i], 
      " & ", 
      paste(other_elements, collapse = " & "),
      ifelse(length(other_elements) > 0, " & ", ""),  # Add " & " only if there are other elements
      last_element,
      "\\\\"
    )
    
    body_lines <- c(body_lines, new_row)
  }
  
  return(paste(body_lines, collapse = "\n"))
}

rent_table <- process_subtable(
  list('OLS' = rent_ols, 'IV' = rent_iv, 'OLS' = rent_ols_x, 'IV' = rent_iv_x, 'OLS' = rent_ols_yxsd, 'IV' = rent_iv_yxsd),
  "Ln (rent/m2)",
  "../output/tables/reducedform_iv_rent.tex"
)

# Process value table
value_table <- process_subtable(
  list('OLS' = value_ols, 'IV' = value_iv, 'OLS' = value_ols_x, 'IV' = value_iv_x, 'OLS' = value_ols_yxsd, 'IV' = value_iv_yxsd),
  "Ln (house sale price)",
  "../output/tables/reducedform_iv_salevalue.tex"
)

# Combine tables and add formatting
combined_table <- c(
  "\\begin{table}[!ht]",
  "\\caption{Airbnb intensity and housing market outcomes}\\label{tab: reduced form iv - rent and sale value}",
  "\\centering",
  "\\footnotesize",
  "\\scalebox{1}{",
  "\\begin{tabular}[t]{lcccccc}",
  "\\toprule",
  rent_table,
  "\\bottomrule",
  "\\\\",
  value_table,
  "\\bottomrule",
  "\\end{tabular}",
  "}",
  "\\legend{Observations are at the wijk (neighborhood) level. A ``district'' is a larger spatial unit than a neighborhood. Rent prices are neighborhood-average long-term rental prices constructed from CBS rent surveys. House sale prices are neighborhood average transaction values, constructed from CBS data covering the universe of housing transactions. Commercial Airbnb listings are constructed from the Inside Airbnb data (see Appendix \\ref{oa-sec: appendix data - airbnb} for construction details). Neighborhood-level control variables are: housing stock, average income, high-skill population share, all from \\href{https://data.amsterdam.nl/datasets/rl6-35tFAw2Ljw/basisbestand-gebieden-amsterdam-bbga/}{ACD BBGA}. Standard errors are clustered at the wijk level in parenthesis.}",
  "\\end{table}"
)

# Write the combined table to a file
writeLines(combined_table, "../output/tables/reducedform_iv_rent_and_salevalue.tex")

files_to_delete <- c("../output/tables/reducedform_iv_rent.tex", 
                     "../output/tables/reducedform_iv_salevalue.tex")

for (file in files_to_delete) {
  if (file.exists(file)) {
    tryCatch({
      file.remove(file)
      cat("Deleted file:", file, "\n")
    }, error = function(e) {
      warning("Failed to delete file: ", file, "\n", "Error: ", e$message)
    })
  } else {
    cat("File does not exist:", file, "\n")
  }
}
