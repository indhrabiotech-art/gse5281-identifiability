#!/usr/bin/env Rscript
# ==============================================================================
# figure_scan_timeline.R
#
# Saves GSE5281 scan dates to CSV, then draws the two-panel scan timeline:
# one row per sample, coloured by diagnosis, GSE48350 above and GSE5281 below.
#
# Usage:  cd ~/project_ml && Rscript R_scripts/figure_scan_timeline.R
# ==============================================================================

rm(list = ls()); options(stringsAsFactors = FALSE)
setwd(path.expand("~/project_ml"))

OUTD <- "04_ML/Discovery_Audit";            dir.create(OUTD, showWarnings = FALSE, recursive = TRUE)
OUTF <- "06_Manuscript/Submission_Figures"; dir.create(OUTF, showWarnings = FALSE, recursive = TRUE)

BLUE <- "#5b7ba8"; RED <- "#c1272d"; GREY <- "#4a4a4a"

# ---------------------------------------------- 1. GSE5281 scan dates to CSV
f5281 <- file.path(OUTD, "GSE5281_scan_dates.csv")
if (!file.exists(f5281)) {
  dirs <- list.dirs(".", recursive = TRUE); dirs <- dirs[grepl("5281", dirs)]
  cels <- unlist(lapply(dirs, function(d)
    list.files(d, pattern = "\\.CEL(\\.gz)?$", full.names = TRUE, ignore.case = TRUE)))
  g  <- sub("\\.CEL.*$", "", sub("_.*$", "", basename(cels)))
  sd <- sapply(cels, function(x)
    tryCatch(affyio::read.celfile.header(x, info = "full")$ScanDate,
             error = function(e) NA_character_))
  m  <- read.csv("02_Metadata/GSE5281_hippocampus_metadata.csv")
  write.csv(data.frame(GSM = g, diagnosis = m$diagnosis[match(g, m$GSM)],
                       scan_date = unname(sd)), f5281, row.names = FALSE)
  cat("written:", f5281, "\n")
}

d52 <- read.csv(f5281)
d48 <- read.csv(file.path(OUTD, "GSE48350_batch_metadata.csv"))

prep <- function(x) {
  x$date <- as.Date(sub(" .*$", "", x$scan_date), format = "%m/%d/%y")
  x$ad   <- grepl("^ad$|alzh|affect", x$diagnosis, ignore.case = TRUE)
  x[!is.na(x$date), ]
}
d48 <- prep(d48); d52 <- prep(d52)

both <- function(x) sum(tapply(x$ad, x$date, function(v) length(unique(v)) == 2))
cat(sprintf("GSE48350: %d samples, %d dates, %d carrying both classes\n",
            nrow(d48), length(unique(d48$date)), both(d48)))
cat(sprintf("GSE5281 : %d samples, %d dates, %d carrying both classes\n",
            nrow(d52), length(unique(d52$date)), both(d52)))

xr <- range(c(d48$date, d52$date))
xr <- c(xr[1] - 40, xr[2] + 40)

panel <- function(x, ttl, lab) {
  x <- x[order(x$date, x$ad), ]
  # stack samples sharing a date
  y <- ave(seq_len(nrow(x)), as.numeric(x$date), FUN = seq_along)
  plot(NA, xlim = xr, ylim = c(0, max(y) + 1.4), axes = FALSE,
       xlab = "", ylab = "")
  ticks <- seq(as.Date("2004-07-01"), as.Date("2007-10-01"), by = "6 months")
  axis(1, at = as.numeric(ticks), labels = format(ticks, "%b\n%Y"),
       cex.axis = 0.68, padj = 0.4, tck = -0.05)
  # vertical guide per date
  for (dt in unique(as.numeric(x$date)))
    segments(dt, 0.25, dt, max(y[as.numeric(x$date) == dt]) + 0.45,
             col = "grey85", lwd = 0.7)
  points(as.numeric(x$date), y, pch = 21, cex = 1.25, lwd = 0.8,
         bg = ifelse(x$ad, RED, BLUE), col = "black")
  mtext(lab, side = 3, line = 0.5, adj = 0, font = 2, cex = 1.05)
  mtext(ttl, side = 3, line = 0.5, cex = 0.8)
  b <- both(x); n <- length(unique(x$date))
  mtext(sprintf("%d of %d scan dates carry both classes", b, n),
        side = 1, line = 2.6, cex = 0.68, font = 3,
        col = if (b == 0) RED else GREY)
}

draw <- function() {
  par(mfrow = c(2, 1), mar = c(4.2, 1.2, 2.6, 1.2), xpd = NA)
  panel(d48, "GSE48350  (discovery, bulk tissue)", "a")
  legend("topright", c("Case", "Control"), pch = 21, pt.bg = c(RED, BLUE),
         col = "black", bty = "n", cex = 0.8, pt.cex = 1.2, horiz = TRUE)
  panel(d52, "GSE5281  (transfer, LCM neurons)", "b")
}

png(file.path(OUTF, "Figure_7_scan_timeline.png"), width = 2000, height = 1500, res = 300)
draw(); dev.off()
pdf(file.path(OUTF, "Figure_7_scan_timeline.pdf"), width = 6.7, height = 5.0)
draw(); dev.off()

cat("\nwritten: Figure_7_scan_timeline.png and .pdf in", OUTF, "\n")
