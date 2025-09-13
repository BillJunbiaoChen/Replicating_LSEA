###############################################
## Project: Endogenous Amenities and Location Sorting
## Purpose: Hetergeneity CF Tables
## Author: Milena Almagro & Tomas Dominguez-Iino
###############################################

library(pacman)
pacman::p_load(tidyverse, broom, data.table, 
               readstata13, readxl,
               lubridate,
               sf,
               viridis, RColorBrewer,ggplot2,latex2exp,ggpubr,
               latex2exp,extrafont,kableExtra)
options(warn = -1)
setwd(dirname(getwd()))
rm(list=ls())

################################################################################

#### Choose specifications
spec_list = c('152')
bootstrap_options =  c('y')
spec_folder = paste0('../data/simulation_results/')


#### Choose final specification
final_spec = 'y'
spec_final = '152'
bootstrap_final = 'y'

for(bootstrap in bootstrap_options ){
  for(spec in spec_list ){
    
    if(bootstrap=='y'){
      spec_gamma = paste0('B_',spec)
    }else{
      spec_gamma = spec }
    
    spec_folder_eq = paste0(spec_folder,'gamma_',spec_gamma,'/equilibrium_objects/')
    spec_folder_cf = paste0(spec_folder,'gamma_',spec_gamma,'/counterfactuals/')
  
    if(final_spec=='y' & bootstrap==bootstrap_final & spec==spec_final){
      output_folder = paste0('../output/figures/counterfactual_heterogeneity')
      output_folder_tables = paste0('../output/tables')
    }else{
      output_folder = paste0(spec_folder,'gamma_',spec_gamma,'/figures/counterfactual_heterogeneity')
      output_folder_tables = paste0(spec_folder,'gamma_',spec_gamma,'/figures/counterfactual_heterogeneity')
    }
    
    
    #Heterogenous vs homogenous preferences
    for(cf in c('hetero','homo')){
    
      #Endogenous vs exogenous amenities
      for(spec_a in c('endo','exo')){
        
        #Rent
        if(cf=='hetero' & spec_a=='endo'){
          simulation_results_rent = paste0(spec_folder_eq,'r_endo.csv') }
        if(cf=='hetero' & spec_a=='exo'){
          simulation_results_rent = paste0(spec_folder_eq,'r_exo.csv') }
        if(cf=='homo' & spec_a=='endo'){
          simulation_results_rent = paste0(spec_folder_eq,'r_endo_homogeneous.csv')}
        if(cf=='homo' & spec_a=='exo'){
          simulation_results_rent = paste0(spec_folder_eq,'r_exo_homogeneous.csv')}
        
        #Rent
        dfx = read.csv(simulation_results_rent, header = FALSE, col.names = c('r'))
        dfx$id = as.integer(1:22)
        rent_mean = mean(dfx$r)
        
        #Amenities
        if(cf=='hetero' & spec_a=='endo'){
          simulation_results_amenities = paste0(spec_folder_eq,'a_endo.csv')
          dfy = read.csv(simulation_results_amenities, header = FALSE, col.names = c('a1','a2','a3','a4','a5','a6'))
          dfy$id = as.integer(1:22)}
        if(cf=='homo' & spec_a=='endo'){
          simulation_results_amenities = paste0(spec_folder_eq,'a_endo_homogeneous.csv')
          dfy = read.csv(simulation_results_amenities, header = FALSE, col.names = c('a1','a2','a3','a4','a5','a6'))
          dfy$id = as.integer(1:22)}
        if(spec_a=='exo'){
          dfy = fread("../data/constructed/gebied_covariates_panel.csv")
          dfy = dfy[year==2017,list(amenity_1,amenity_2,amenity_3,amenity_4,amenity_5,amenity_6)] %>% rename("a1"="amenity_1","a2"="amenity_2","a3"="amenity_3","a4"="amenity_4","a5"="amenity_5","a6"="amenity_6")
          dfy$id = as.integer(1:22)}
        
        #Compute gini for each variable
        df_m = merge(dfx,dfy, by='id')
        N=nrow(df_m)
        gini_row = df_m[1,c('r','a1','a2','a3','a4','a5','a6')]*0
        for(var_name in c('r','a1','a2','a3','a4','a5','a6') ){
          df = df_m[c(var_name)]
          names(df)[names(df) == var_name] <- 'y_n'
          df = arrange(df, y_n)
          df$n = as.integer(1:N)
          df$ny_n = df$n*df$y_n
          gini = 2*sum(df$ny_n)/(N*sum(df$y_n))-((N+1)/N)
          
          if(var_name=='r'){gini_row$r= gini}
          if(var_name=='a1'){gini_row$a1= gini}
          if(var_name=='a2'){gini_row$a2= gini}
          if(var_name=='a3'){gini_row$a3= gini}
          if(var_name=='a4'){gini_row$a4= gini}
          if(var_name=='a5'){gini_row$a5= gini}
          if(var_name=='a6'){gini_row$a6= gini}
        }
        
        #Format table 
        gini_col = transpose(gini_row)
        if(spec_a=='endo'){
          colnames(gini_col) <- 'gini_endo'
          gini_col$amenity = colnames(gini_row)
          gini_endo = gini_col}
        else{
          colnames(gini_col) <- 'gini_exo'
          gini_col$amenity = colnames(gini_row)
          gini_exo = gini_col
        }
        
      }
        
      gini_table = merge(gini_exo,gini_endo,by='amenity')
      gini_table$delta = gini_table$gini_endo-gini_table$gini_exo
      gini_table = gini_table %>% mutate_if(is.numeric, round, digits=2)
      
      if(cf=='hetero'){gini_hetero = gini_table}
      if(cf=='homo'){gini_homo = gini_table}
      
    }
    
    
    #### TABLES - NEIGHBORHOOD DIFFERENTIATION
    
    #Export .csv table
    tab = merge(gini_homo, gini_hetero,by='amenity')
    tab$amenity = c('Touristic amenities','Restaurants','Bars','Food stores','Non-food stores','Nurseries','Rent')
    setnames(tab, c('Amenity','Exogenous','Endogenous','Delta','Exogenous','Endogenous','Delta'))
    tab_name = "counterfactual_heterogeneity_differentiation"
    write.csv(tab, paste0(output_folder_tables,"/",tab_name,".csv"), row.names = FALSE)
  
    #Export .tex table
    export_tab = knitr::kable(tab, booktabs = T, align = "lcccccc", format = 'latex',
                              caption='Neighborhood differentiation as spatial dispersion of amenities.', 
                              label='heterogeneous vs homogenous - neighborhood differentiation',
                              format.args = list(big.mark = ",")) %>%
      kable_styling(latex_options = c("hold_position")) %>%
      add_header_above(c(" " = 1,"Homogenous"=3,"Heterogenous"=3) ) %>%
      footnote(general_title = "\\\\footnotesize{Notes:}", general = "", escape = F)
    kableExtra::save_kable(export_tab, file = paste0(output_folder_tables, "/",tab_name,".tex"))
    
    #### TABLES - NEIGHBORHOOD DIFFERENTIATION - ONLY ENDOGENOUS AMENITIES
    tab = setDT(merge(gini_homo, gini_hetero,by='amenity'))
    tab = tab[amenity!='r',list(amenity,gini_endo.x,gini_endo.y)][, delta:=gini_endo.y-gini_endo.x]
    tab$amenity = c('Touristic amenities','Restaurants','Bars','Food stores','Non-food stores','Nurseries')
    setnames(tab, c('Amenity','Homogenous (HO)','Heterogenous (HE)','HE-HO'))
    export_tab = knitr::kable(tab, booktabs = T, align = "lccc", format = 'latex',
                              caption='Neighborhood differentiation as spatial dispersion of amenities.', 
                              label='heterogeneous vs homogenous - neighborhood differentiation',
                              format.args = list(big.mark = ",")) %>%
      kable_styling(latex_options = c("hold_position")) %>%
      add_header_above(c(" " = 1,"Gini index for each preference specification"=2, " " = 1) )
    kableExtra::save_kable(export_tab, file = paste0(output_folder_tables, "/",tab_name,"_endo.tex"))
    
    
    #### FIGURES - SORTING AND INEQUALITY
    
    # Residential sorting
    for(pref in c('hetero','homo')){
      
      for(spec_a in c('exo','endo')){
        
        if(pref=='hetero' & spec_a=='endo'){filename=paste0('DL_endo.csv')}
        if(pref=='hetero' & spec_a=='exo'){filename=paste0('DL_exo.csv')}
        if(pref=='homo' & spec_a=='endo'){filename=paste0('DL_endo_homogeneous.csv')}
        if(pref=='homo' & spec_a=='exo'){filename=paste0('DL_exo_homogeneous.csv')}
            
        
        df = read.csv(paste0(spec_folder_eq,filename), header = F, col.names = c('D_j1','D_j2','D_j3'))
        df = df %>% mutate(D_j = rowSums(across(where(is.numeric))))
        df = df %>% mutate_at(vars(D_j1:D_j3) , funs(share = ./D_j))
        setnames(df,c("D_j1_share","D_j2_share","D_j3_share"),c("d_j1","d_j2","d_j3"))
        df$v_j = -df$d_j1*log(df$d_j1)-df$d_j2*log(df$d_j2)-df$d_j3*log(df$d_j3)
          
        D_1 = sum(df$D_j1)
        D_2 = sum(df$D_j2)
        D_3 = sum(df$D_j3)
        D = sum(df$D_j)
        
        v_hat = -(D_1/D)*log(D_1/D)-(D_2/D)*log(D_2/D)-(D_3/D)*log(D_3/D)
        df$v_bar_j = df$v_j*df$D_j/D
        v_bar = sum(df$v_bar_j)
        entropy = (v_hat-v_bar)/v_hat
        
        if(pref=='hetero' & spec_a=='endo'){entropy_hetero_endo=entropy}
        if(pref=='hetero' & spec_a=='exo'){entropy_hetero_exo=entropy}
        if(pref=='homo' & spec_a=='endo'){entropy_homo_endo=entropy}
        if(pref=='homo' & spec_a=='exo'){entropy_homo_exo=entropy}
      }
    }
    
    df = data.frame('entropy'=c(entropy_hetero_exo,entropy_hetero_endo,entropy_homo_exo,entropy_homo_endo))
    df$pref = c('Heterogenous','Heterogenous','Homogenous','Homogenous')
    df$spec_a = c('Exogenous','Endogenous','Exogenous','Endogenous')
    df$spec_a = factor(df$spec_a, levels = c('Exogenous', 'Endogenous'))
    figure_entropy = ggplot(df, aes(x=pref, y=entropy, fill=spec_a)) +
      geom_bar(position='dodge', stat='identity', linewidth=0.4) +  
      labs(title = 'Residential sorting', x='', y='Entropy index') +
      theme_minimal() +
      theme(aspect.ratio = 1, 
            legend.position = c(0.2,0.9),
            legend.title = element_blank(),
            legend.text = element_text(size=7),
            plot.title = element_text(hjust = 0.5),
            text=element_text(family="Palatino"),
            panel.grid = element_blank(),
            axis.line = element_line(linewidth=0.25),
            axis.ticks = element_line(linewidth=0.25) ) +
      scale_fill_manual(values=c("gray30", "gray60"))
    
    df = data.frame('entropy'=c(entropy_hetero_endo,entropy_homo_endo))
    df$pref = c('Heterogenous','Homogenous')
    figure_entropy_endo = ggplot(df, aes(x=pref, y=entropy)) +
      geom_bar(position='dodge', stat='identity', linewidth=0.4) +  
      labs(title = 'Residential sorting', x='', y='Entropy index') +
      theme_minimal() +
      theme(aspect.ratio = 1, 
            legend.position = c(0.2,0.9),
            legend.title = element_blank(),
            legend.text = element_text(size=7),
            plot.title = element_text(hjust = 0.5),
            text=element_text(family="Palatino"),
            panel.grid = element_blank(),
            axis.line = element_line(linewidth=0.25),
            axis.ticks = element_line(linewidth=0.25) )
    
    
    # Welfare inequality
    df_het = read.csv(paste0(spec_folder_cf,'CS_endo.csv'), header = F, col.names = c('Heterogenous'))
    df_hom = read.csv(paste0(spec_folder_cf,'CS_endo_homogeneous.csv'), header = F, col.names = c('Homogenous'))
    df_endo = merge(df_het,df_hom, by= "row.names")
    
    df_het = read.csv(paste0(spec_folder_cf,'CS_exo.csv'), header = F, col.names = c('Heterogenous'))
    df_hom = read.csv(paste0(spec_folder_cf,'CS_exo_homogeneous.csv'), header = F, col.names = c('Homogenous'))
    df_exo = merge(df_het,df_hom, by= "row.names")
    
    # Top-bottom welfare ratio
    df = data.frame('welfare_gap'=c(max(df_exo$Heterogenous)/min(df_exo$Heterogenous), 
                                    max(df_endo$Heterogenous)/min(df_endo$Heterogenous),
                                    max(df_exo$Homogenous)/min(df_exo$Homogenous),
                                    max(df_endo$Homogenous)/min(df_endo$Homogenous)))
    df$pref = c('Heterogenous','Heterogenous','Homogenous','Homogenous')
    df$spec_a = c('Exogenous','Endogenous','Exogenous','Endogenous')
    df$spec_a = factor(df$spec_a, levels = c('Exogenous', 'Endogenous'))
    df_welfare_ratio = setDT(df)
    
    # Standard deviation of welfare
    df = data.frame('welfare_gap'=c(sd(df_exo$Heterogenous),sd(df_endo$Heterogenous),sd(df_exo$Homogenous),sd(df_endo$Homogenous)))
    df$pref = c('Heterogenous','Heterogenous','Homogenous','Homogenous')
    df$spec_a = c('Exogenous','Endogenous','Exogenous','Endogenous')
    df$spec_a = factor(df$spec_a, levels = c('Exogenous', 'Endogenous'))
    df_welfare_sd = setDT(df)
    
    figure_wi = ggplot(df_welfare_ratio, aes(x=pref, y=welfare_gap, fill=spec_a)) +
      geom_bar(position='dodge', stat='identity', size=0.4) +  
      geom_text(aes(label = round(welfare_gap,2)), size=3, vjust = -0.3,  position = position_dodge(.9))+
      labs(title = 'Welfare inequality', x='', y='Welfare Gap') +
      theme_minimal() +
      theme(aspect.ratio = 1, 
            legend.position = c(0.2,0.9),
            legend.title = element_blank(),
            legend.text = element_text(size=7),
            plot.title = element_text(hjust = 0.5),
            text=element_text(family="Palatino"),
            panel.grid = element_blank(),
            axis.line = element_line(linewidth=0.25),
            axis.ticks = element_line(linewidth=0.25) ) +
      scale_fill_manual(values=c("gray30", "gray60"))
    
    figure_wi_endo = ggplot(df_welfare_ratio[spec_a=='Endogenous',], aes(x=pref, y=welfare_gap)) +
      geom_bar(position='dodge', stat='identity', size=0.4) +  
      labs(title = 'Welfare inequality', x='', y='Welfare Gap') +
      theme_minimal() +
      theme(aspect.ratio = 1, 
            legend.position = c(0.2,0.9),
            legend.title = element_blank(),
            legend.text = element_text(size=7),
            plot.title = element_text(hjust = 0.5),
            text=element_text(family="Palatino"),
            panel.grid = element_blank(),
            axis.line = element_line(linewidth=0.25),
            axis.ticks = element_line(linewidth=0.25) )
    
    
    df = data.frame('endogeneity_role'=c((max(df_endo$Heterogenous)/min(df_endo$Heterogenous))/(max(df_exo$Heterogenous)/min(df_exo$Heterogenous)), 
                                    (max(df_endo$Homogenous)/min(df_endo$Homogenous))/(max(df_exo$Homogenous)/min(df_exo$Homogenous)) ))
    df$pref = c('Heterogenous','Homogenous')
    figure_wi_ea = ggplot(df, aes(x=pref, y=endogeneity_role)) +
      geom_bar(position='dodge', stat='identity', linewidth=0.4) +  
      labs(title = 'Role of endogenous amenities for inequality', x='', y='Endogenous/exogenous welfare gap') +
      theme_minimal() +
      theme(aspect.ratio = 1, 
            legend.position = c(0.2,0.9),
            legend.title = element_blank(),
            legend.text = element_text(size=7),
            plot.title = element_text(hjust = 0.5),
            text=element_text(family="Palatino"),
            panel.grid = element_blank(),
            axis.line = element_line(linewidth=0.25),
            axis.ticks = element_line(linewidth=0.25) )
    
    figure = ggarrange(figure_entropy, figure_wi, figure_wi_ea, nrow=1, ncol=3)
    figname = "bar_sorting_inequality"
    pdf(paste0(output_folder,"/",figname,".pdf"), width = 12, height = 3.75)
    print(figure)
    dev.off()
    
    figname = "bar_sorting_inequality_endo"
    figure = ggarrange(figure_entropy_endo, figure_wi_endo, nrow=1, ncol=2)
    pdf(paste0(output_folder,"/",figname,".pdf"), width = 6, height = 3)
    print(figure)
    dev.off()
    
  
  }
}
