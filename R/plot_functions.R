#' Side by Side Histogram and Box Plots
#'
#' @param dat The dataset
#' @param x The variable in the dataset
#'
#' @return nothing
#' @export
#'
#' @examples
#' nothing
hist_box_plots <- function(dat, x){
  hist(dat[, x], main = paste0("Histogram of ",x), xlab=x, ylab="Frequency", col = "blue")
  boxplot(dat[, x], main = paste0("Boxplot of ",x), xlab=x, col = "blue")
}

#' Stacked Histograms
#'
#' @param dat_a The dataset for stack 1
#' @param dat_b The dataset for stack 2
#' @param x The variable in the dataset
#'
#' @return nothing
#' @export
#'
#' @examples
#' nothing
stacked_bar_plots <- function(dat_a, dat_b, x){
  hist(dat_a[, x], main = paste0("Stacked Histogram of ",x), xlab=x, ylab="Frequency", col=rgb(.5,.8,1,0.5))
  hist(dat_b[, x], col = rgb(1,.5,.4,.5), add=T)
  legend("topright", c("Alert", "Not Alert"), col=c(rgb(.5,.8,1,0.5), rgb(1,.5,.4,.5)), lwd=10)
  box()
}

