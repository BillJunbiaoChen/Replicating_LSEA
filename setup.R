if (!require("pacman")) install.packages("pacman")
library(pacman)
options(warn = -1)
packages <- readLines("r_requirements.txt")

packages <- packages[packages != "pacman"]

p_load(char = packages)

install.packages("remotes")
remotes::install_github("skgrange/threadr")

if (!require("modelsummary")) {
  install.packages(
    "https://cran.r-project.org/src/contrib/Archive/modelsummary/modelsummary_1.4.5.tar.gz",
    repos = NULL, type = "source"
  )
} else if (packageVersion("modelsummary") != "1.4.5") {
  remove.packages("modelsummary")
  install.packages(
    "https://cran.r-project.org/src/contrib/Archive/modelsummary/modelsummary_1.4.5.tar.gz",
    repos = NULL, type = "source"
  )
}
library(modelsummary)
packageVersion("modelsummary")
