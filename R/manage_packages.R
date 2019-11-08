# Install packages to a docker image with packrat

# Install packrat
install.packages("packrat", repos = "https://cran.rstudio.com/")

# Initialize packrat, but don't let it try to find packages to install itself.
packrat::init(
  infer.dependencies = FALSE,
  enter = TRUE,
  restart = FALSE
)

# Install CRAN packages
list.of.packages <- c(
  "caret",
  "mlbench",
  "fscaret",
  "dplyr",
  "psych",
  "corrplot",
  "xgboost",
  "ROCR",
  "pROC",
  "gridExtra",
  "gbm",
  "e1071",
  "roxygen2",
  "devtools"
)

new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[, "Package"])]
if (length(new.packages)) install.packages(new.packages, repos = "https://cran.rstudio.com/", dependencies = T)

usethis::use_package("caret")
usethis::use_package("mlbench")
usethis::use_package("fscaret")
usethis::use_package("dplyr")
usethis::use_package("psych")
usethis::use_package("corrplot")
usethis::use_package("xgboost")
usethis::use_package("ROCR")
usethis::use_package("pROC")
usethis::use_package("gridExtra")
usethis::use_package("gbm")
usethis::use_package("e1071")

# Take snapshot

packrat::snapshot(
  snapshot.sources = FALSE,
  ignore.stale = TRUE,
  infer.dependencies = FALSE
)
