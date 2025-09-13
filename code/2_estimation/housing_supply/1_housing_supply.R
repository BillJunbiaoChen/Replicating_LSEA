###############################################
## Project: Endogenous Amenities and Location Sorting
## Purpose: Estimate housing supply regressions
## Author: Milena Almagro & Tomas Dominguez-Iino; Tables: Sriram Tolety
###############################################

library(pacman)
library(modelsummary)
packageVersion("modelsummary")
pacman::p_load(rstudioapi, tidyverse, broom, data.table, 
               readstata13, readxl,foreign,
               lfe, fixest, ivreg, plyr, lubridate,
               hrbrthemes, viridis, RColorBrewer, ggplot2,
               latex2exp,sf,kableExtra)
options(warn = -1)
setwd(dirname(getwd()))
print(getwd())
rm(list = setdiff(ls(), "wd"))
options("modelsummary_format_numeric_latex" = "plain")
`%!in%` <- Negate(`%in%`)


################## Import BBGA and Airbnb data

#Public data (BBGA and Airbnb)
df1 = fread("../data/bbga/bbga.csv")
df2 = fread("../data/airbnb/listings_by_wk.csv")
setnames(df2, old=c('total'), new=c('total_listings'))
df3 = fread("../data/googletrends/google_trends_annual.csv")
setnames(df3, old=c('mean'), new=c('google'))

dfp = merge(df1,df2, by=c('year','wk_code'))[,.(year, wk_code, sd_code, gsd,
                                                total_listings, commercial, ehha, entire_home,
                                                addresses_total, addresses_private_rental, addresses_owner_corporation,
                                                price_mean, price_median,
                                                tourism_offices,pop_total,hotel_beds_imp,
                                                review_scores_location,mean_accommodates)]
dfp = merge(dfp,df3, by=c('year'))


################## Import confidential CBS neighborhood-level data
df1_cbs = setDT(read_excel('../data/cbs/exports/LIGHT_rent_house_transactions_aangepast - 9.23.2021/LIGHT_rent_house_transactions_aangepast.xlsx', sheet=2))
df2_cbs = setDT(read_excel('../data/cbs/exports/LIGHT_rent_house_transactions_aangepast - 9.23.2021/LIGHT_rent_house_transactions_aangepast.xlsx', sheet=3))
df3_cbs = setDT(read_excel('../data/cbs/exports/LIGHT_rent_house_transactions_aangepast - 9.23.2021/LIGHT_rent_house_transactions_aangepast.xlsx', sheet=4))
df =  merge(df1_cbs,df2_cbs, by=c('year','wk'))
df =  merge(df,df3_cbs, by=c('year','wk'))
setnames(df[, c("num_obs","num_obs_rent"):=NULL], old=c('wk'), new=c('wk_code'))
df = df[df$num_obs_rent_hat!='<10',]
df$num_obs_rent_hat = as.numeric(df$num_obs_rent_hat)
df_cbs1=df

df = setDT(read_excel('../data/cbs/exports/Vrijgegeven220328_0500_House and Rent Prices by wijk - March 2022/House and Rent Priced by wijk - March 2022.xlsx', sheet=4))
df = df[df$num_obs_rent_hat!='<10',]
setnames(df, old=c('wk'), new=c('wk_code'))
df$num_obs_rent_hat = as.numeric(df$num_obs_rent_hat)
df_cbs2=df


################## Merge quantity and price data
dfp = merge(df_cbs2,dfp, by=c('year','wk_code'))

################## Data filters and variable definition

#Restrict years and spatial units
start_year = 2015
end_year = 2017
df = dfp[year>= start_year & year <= end_year,]
df = df[wk_code %!in% c(10,11,50,51),]

#Define quantities and prices
df$quantity_ltr = df$addresses_private_rental
df$quantity_str = df$commercial
df$price_ltr = df$net_rent_hat_rf*12
df$price_str = df$price_mean*365

#Export gebied-level variables
dfx = df[year==2017,.(year, gsd, quantity_str, quantity_ltr)][, lapply(.SD, sum, na.rm=TRUE), by=c('year','gsd') ]
dfy = df[year==2017,.(year, gsd, price_str, price_ltr)][, lapply(.SD, mean, na.rm=TRUE), by=c('year','gsd') ]
df_xy = merge(dfx,dfy, by=c('year','gsd') ) 
setnames(df_xy, old=c('gsd'), new=c('gb_code'))
write.csv(df_xy, "../data/final/inputs/str_ltr_gebied.csv", row.names = FALSE)


#Define Y and X variables
df$y = log(df$quantity_ltr) - log(df$quantity_str)
df$x = (df$price_ltr- df$price_str)/10000
df$x_exo = log(df$mean_accommodates)

#Define instruments
dfz = df1[year==2008,  .(wk_code, tourism_offices)]
setnames(dfz, old=c('tourism_offices'), new=c('tourism_offices_start'))
df =   merge(df, dfz, by=c('wk_code'))
df$z = df$google*df$tourism_offices_start
df = df[is.finite(x),]

#Statistics for interpretation
df$price_gap = df$price_ltr-df$price_str
df$share_ltr = df$quantity_ltr/(df$quantity_ltr+df$quantity_str)
alpha_estimate=0.385
mean_share_ltr = mean(df$share_ltr)

sd_effect = (exp(alpha_estimate)-1)*abs(sd(df$price_gap)/mean(df$price_gap))
y = (1+sd_effect)*mean_share_ltr/(1-mean_share_ltr)
x = y/(1+y)
  
#Define spatial unit
df_wk = df
setnames(df_wk, old=c('wk_code'), new=c('spatial_unit'))

#Define spatial unit
df_gd = df
df_gd$spatial_unit=df_gd$gsd
df_gd=df_gd[, c('year','spatial_unit','y','x','x_exo','z')][ , lapply(.SD, mean) , by=c("year", "spatial_unit")]


################## Final regressions

df = df_wk

#FE OLS
m_ols = feols(y ~ x + x_exo, cluster = c('spatial_unit'), data = df)
m_ols_y = feols(y ~ x + x_exo | year, cluster = c('spatial_unit'), data = df)
m_ols_wk = feols(y ~ x + x_exo| spatial_unit, cluster = c('spatial_unit'), data = df)
m_ols_y_wk = feols(y ~ x + x_exo | year + spatial_unit, cluster = c('spatial_unit'), data = df)
msummary(list(m_ols, m_ols_y, m_ols_wk, m_ols_y_wk), 
         estimate = "{estimate}{stars}", coef_omit = 'Intercept', gof_omit = 'R2 Pseudo|AIC|BIC|Log.Lik.')

#IV
m_iv = feols(y ~ 1 + x_exo| x~z, cluster = c('spatial_unit'), data = df)
m_iv_y = feols(y ~ 1 + x_exo | year | x~z , cluster = c('spatial_unit'), data = df)
m_iv_wk = feols(y ~ 1 + x_exo | spatial_unit | x~z, cluster = c('spatial_unit'), data = df)
m_iv_y_wk = feols(y ~ 1 + x_exo | year + spatial_unit | x~z , cluster = c('spatial_unit'), data = df)
msummary(list(m_iv, m_iv_y, m_iv_wk, m_iv_y_wk), 
         estimate = "{estimate}{stars}", coef_omit = 'Intercept', gof_omit = 'R2 Pseudo|AIC|BIC|Log.Lik.')

#First stage F-stat
F1_w = as.character(round(fitstat(m_iv, "ivwald", simplify = T)$stat, digits=2))
F2_w = as.character(round(fitstat(m_iv_y, "ivwald", simplify = T)$stat, digits=2))
F3_w = as.character(round(fitstat(m_iv_wk, "ivwald", simplify = T)$stat, digits=2))
F4_w = as.character(round(fitstat(m_iv_y_wk, "ivwald", simplify = T)$stat, digits=2))


################## Export tables 

rows <- tribble(~term               , ~OLS, ~IV, ~OLS, ~IV, ~OLS, ~IV, ~OLS, ~IV,
                'Year FE'           ,   '',  '', 'X', 'X',  '',  '',  'X','X', 
                'Wijk FE'           ,   '',  '', '',   '', 'X',  'X', 'X','X', 
                'First stage F-stat',   '',  F1_w, '',  F2_w,  '',   F3_w,   '', F4_w)
attr(rows, 'position') <- c(3,4,5)
gm <- tibble::tribble(
  ~raw,        ~clean,          ~fmt,
  "nobs",      "Observations",  0,)

tab_txt <- msummary(list('OLS'=m_ols, 'IV'=m_iv,
              'OLS'=m_ols_y, 'IV'= m_iv_y, 
              'OLS'=m_ols_wk, 'IV'=m_iv_wk,
              'OLS'=m_ols_y_wk, 'IV'=m_iv_y_wk),
         title = 'Long-term (LT) relative to short-term (ST) housing supply elasticities',
         coef_omit = 'Intercept',
         coef_map =  c('x' = 'LT price-ST price','fit_x'= 'LT price-ST price'),
         estimate = "{estimate}{stars}",
         gof_map = gm, 
         escape=F,
         add_rows = rows,
         notes = list('Notes: Standard errors clustered at the wijk level in parenthesis.'))  %>%
  add_header_above(c(" " = 1, "Dependent variable: ln (LT share) - ln (ST share)" = 8)) %>%
  row_spec(1, bold = T)
print(tab_txt)


rows <- tribble(~term               , ~OLS, ~IV, ~IV, ~IV,
                'Year FE'           ,   '',  '', 'X', 'X', 
                'Wijk FE'           ,   '',  '', '' , 'X', 
                'First stage F-stat',   '',  F1_w,  F2_w, F4_w)
attr(rows, 'position') <- c(3,4,5)
gm <- tibble::tribble(
  ~raw,        ~clean,          ~fmt,
  "nobs",      "Observations",  0,)

tab_latex <- msummary(list('OLS'=m_ols, 'IV'=m_iv,'IV'= m_iv_y, 'IV'=m_iv_y_wk),
                      title = '\\label{tab: housing_supply_ols_iv}Long-term (LT) relative to short-term (ST) housing supply elasticities',
                      coef_omit = 'Intercept',
                      coef_map =  c('x' = 'LT price-ST price','fit_x'= 'LT price-ST price'),
                      estimate = "{estimate}{stars}",
                      gof_map = gm, 
                      escape=F,
                      add_rows = rows,
                      output='latex')  %>%
  add_header_above(c(" " = 1, "Dependent variable: ln (LT share) - ln (ST share)" = 4)) %>%
  row_spec(1, bold = T)
kableExtra::save_kable(tab_latex, file = "../output/tables/housing_supply_estimates_table.tex")

tex_content <- readLines("../output/tables/housing_supply_estimates_table.tex")

# Define the caption
caption <- paste0(
  "\\legend{Table reports estimates of landlords' marginal utility of income for a discrete choice model ",
  "between the short- and long-term rental markets. Data are a panel with 92 locations 2015-2017. Prices are instrumented using ",
  "a shift-share instrument \\citep{barron2021effect} that proxies for demand shocks. Wijk-level clustered standard errors in parenthesis. ",
  "\\sym{*}\\(p<0.10\\), \\sym{**}\\(p<0.05\\), \\sym{***}\\(p<0.01\\).}"
)

end_tabular_pos <- which(grepl("\\\\end\\{tabular\\}", tex_content))
tex_content <- c(
  tex_content[1:end_tabular_pos],
  caption,
  tex_content[(end_tabular_pos+1):length(tex_content)]
)

writeLines(tex_content, "../output/tables/housing_supply_estimates_table.tex")




print("Adding table attribute h")

tex_content <- readLines("../output/tables/housing_supply_estimates_table.tex")

# Define the caption
caption <- paste0(
  "[!h]"
)

end_tabular_pos <- which(grepl("\\\\begin\\{table\\}", tex_content))
tex_content[end_tabular_pos] <- paste0(tex_content[end_tabular_pos], "[!h]")
writeLines(tex_content, "../output/tables/housing_supply_estimates_table.tex")


tex_content <- readLines("../output/tables/housing_supply_estimates_table.tex")
centering_pos <- which(grepl("\\\\centering", tex_content))[1]
tex_content <- tex_content[-centering_pos]
writeLines(tex_content, "../output/tables/housing_supply_estimates_table.tex")




print("Adding scalebox")

tex_content <- readLines("../output/tables/housing_supply_estimates_table.tex")
# Define the caption
caption <- paste0(
  "\\scalebox{0.85}{"
)
end_tabular_pos <- which(grepl("\\centering", tex_content))
tex_content <- c(
  tex_content[1:end_tabular_pos],
  caption,
  tex_content[(end_tabular_pos+1):length(tex_content)]
)
writeLines(tex_content, "../output/tables/housing_supply_estimates_table.tex")

tex_content <- readLines("../output/tables/housing_supply_estimates_table.tex")
# Define the caption
caption <- paste0(
  "}"
)
end_tabular_pos <- which(grepl("\\\\end\\{tabular\\}", tex_content))
tex_content <- c(
  tex_content[1:end_tabular_pos],
  caption,
  tex_content[(end_tabular_pos+1):length(tex_content)]
)
writeLines(tex_content, "../output/tables/housing_supply_estimates_table.tex")



################## Export fixed effects

#Shapefile
dts = sf::read_sf(dsn=path.expand("../data/shapefiles/"), layer="wijk")
dts <- st_simplify(dts, dTolerance = 0.1, preserveTopology = T)
dts$wk_code = parse_number(dts$wk_code)
dts = dts[dts$wk_code %!in% c(10,11,50,51),] 

# Time fixed effects
model_spec = m_iv_y_wk
df_te = data.frame(fixef(model_spec)$year)
df_te$year = floor_date(as.Date(as.character(rownames(df_te)), format="%Y"),'year')
names(df_te)[1] <- "kappa_t"
setDT(df_te)
setcolorder(df_te, c("year","kappa_t"))
plot(df_te$year, df_te$kappa_t, type='b')

# Location fixed effects
model_spec = m_iv_y_wk
df_le = data.frame(fixef(model_spec)$spatial_unit)
df_le$wk_code = as.numeric(rownames(df_le))
names(df_le)[1] <- "kappa_j"
setDT(df_le)
setcolorder(df_le, c("wk_code","kappa_j"))
df_le$kappa_j_std = (df_le$kappa_j - mean(df_le$kappa_j))/sd(df_le$kappa_j)

summary(df_le$kappa_j)
hist(df_le$kappa_j)
dt <- merge(dts, df_le, by="wk_code")
figure = ggplot() +
  geom_sf(data=dt, aes(fill=kappa_j_std), linewidth=0.1)+
  scale_fill_fermenter(palette = "RdBu", direction=-1, na.value = "white") +
  theme_void() +
  labs(title = '', fill='') + 
  theme(legend.position = "right",  legend.direction = "vertical", 
        legend.text = element_text(size= 12),
        legend.key.size = unit(1.15, "cm"),
        legend.key.width = unit(0.6,"cm"),
        plot.title = element_text(size = 12, hjust=0))

print(figure)
dev.off()

#Export parameters
df_le$alpha = model_spec$coefficients[1]
df_le$kappa_t_2015 = fixef(model_spec)$year[1]
df_le$kappa_t_2016 = fixef(model_spec)$year[2]
df_le$kappa_t_2017 = fixef(model_spec)$year[3]
df_estimates = df_le[,.(wk_code, kappa_j,kappa_t_2015,kappa_t_2016,kappa_t_2017,alpha)]
write.csv(df_estimates, "../output/estimates/housing_supply_estimates.csv", row.names = FALSE)
write.csv(df_estimates, "../data/final/estimates/housing_supply_estimates.csv", row.names = FALSE)


#Export data at the gebied-level 
df_wk_2017 = df[year==2017,.(year, sd_code, gsd,
                         quantity_ltr,quantity_str,
                         price_ltr,price_str,
                         pop_total)]
df_wk_2017$gb = df_wk_2017$gsd
df_wk_2017$gsd = NULL
df_gb_2017_quantities = df_wk_2017[,.(gb, quantity_ltr, quantity_str)][ , lapply(.SD, sum) , by=c( "gb")]
df_gb_2017_prices = df_wk_2017[,.(gb, price_str)][ , lapply(.SD, mean) , by=c( "gb")]
df_gb_2017 = merge(df_gb_2017_quantities, df_gb_2017_prices, by=c('gb'))

df_gb_prices = fread("../data/constructed/imputed_rent.csv")
df_gb_prices = df_gb_prices[year==2017,.(year, gb, net_rent_hat_rf)]

df_gb_final = merge(df_gb_2017, df_gb_prices, by=c('gb'))
df_gb_final$price_ltr = 12*df_gb_final$net_rent_hat_rf
df_gb_final$net_rent_hat_rf = NULL

df_gb_final$x = (df_gb_final$price_ltr- df_gb_final$price_str)/10000
df_gb_final$share_quantity_str = df_gb_final$quantity_str/(df_gb_final$quantity_str+df_gb_final$quantity_ltr)
df_gb_final$lhs = log(1-df_gb_final$share_quantity_str)-log(df_gb_final$share_quantity_str)
df_gb_final$delta_j = df_gb_final$lhs- model_spec$coefficients[1]*df_gb_final$x
df_gb_final$lhs = NULL
df_gb_final$x = NULL
write.csv(df_gb_final, "../data/final/inputs/gebied_housing_supply.csv", row.names = FALSE)


