###############################################
## Project: Endogenous Amenities and Location Sorting
## Purpose: Construct distance matrix between gebied centroids
## Author: Sriram Tolety
###############################################
require(sf)
library(dplyr)

options(warn = -1)

setwd(dirname(getwd()))
print(getwd())
base <- read_sf("../data/shapefiles/wijk.shp")
data <- read.csv("../data/shapefiles/wijk_to_gb.csv")

wk_codes <- base$wk_code
base$gb <- as.integer(length(wk_codes))
for (i in 1:length(wk_codes)) {
    base$gb[i] <- data$gb[data$wk==i-1]
}

base <- base %>% 
    arrange(gb) %>%
    group_by(gb) %>%
    summarise()
base <- st_transform(base, crs = 3857)
centroids <- st_centroid(base) 
names(centroids) <- c('gb', 'geometry')
dist_matrix <- (st_distance(centroids))
dist_matrix <- dist_matrix/1000
dist_matrix <- dist_matrix[1:22, 1:22]
colnames(dist_matrix) <- as.integer(colnames(dist_matrix))
print(dist_matrix)
write.csv(dist_matrix, "../data/final/inputs/dist_mat_centroids.csv", row.names = FALSE)
