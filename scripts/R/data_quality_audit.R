#!/usr/bin/env Rscript
# ==============================================================================
# data_quality_audit.R
#   1. exhaustive duplicate-array detection, both datasets
#   2. cross-block individual-ID collisions in GSE48350 (all 253 samples)
#   3. whether the six sex mismatches are explained by duplication
#   4. impact of the duplicate on every internal estimate
#   5. sensitivity analysis with the duplicate removed
# Usage: cd ~/project_ml && Rscript R_scripts/data_quality_audit.R
# ==============================================================================
rm(list=ls()); options(stringsAsFactors=FALSE, warn=1)
setwd(path.expand("~/project_ml"))
OUT <- "04_ML/Data_Quality"; dir.create(OUT, showWarnings=FALSE, recursive=TRUE)
sink(file.path(OUT,"data_quality_log.txt"), split=TRUE)
if(!requireNamespace("pROC",quietly=TRUE)) install.packages("pROC",repos="https://cloud.r-project.org")
key <- function(x) sub("\\.CEL.*$","",sub("_.*$","",x))
cat("=========== DATA QUALITY AUDIT ===========\n", format(Sys.time()), "\n")

## ---- 1. duplicate arrays -------------------------------------------------
cat("\n--------- 1. DUPLICATE ARRAYS ---------\n")
dups <- list()
for (f in c("03_Preprocessing/GSE48350_RMA_genelevel.csv",
            "03_Preprocessing/GSE5281_RMA_genelevel.csv")) {
  E <- as.matrix(read.csv(f, row.names=1, check.names=FALSE))
  colnames(E) <- key(colnames(E))
  C <- cor(E, method="spearman"); diag(C) <- NA
  h <- which(C > 0.999, arr.ind=TRUE); h <- h[h[,1] < h[,2], , drop=FALSE]
  cat("\n", basename(f), " n =", ncol(E), "  pairs above 0.999:", nrow(h), "\n")
  if (nrow(h)) for (i in seq_len(nrow(h))) {
    A <- colnames(E)[h[i,1]]; B <- colnames(E)[h[i,2]]
    ident <- identical(E[,A], E[,B])
    cat(sprintf("   %s / %s  rho=%.6f  bit-identical=%s\n", A, B, C[h[i,1],h[i,2]], ident))
    dups[[length(dups)+1]] <- data.frame(dataset=basename(f), s1=A, s2=B,
                                         rho=C[h[i,1],h[i,2]], identical=ident)
  }
}
if (length(dups)) write.csv(do.call(rbind,dups),
  file.path(OUT,"duplicate_arrays.csv"), row.names=FALSE)

## ---- 2. cross-block individual collisions, full series -------------------
cat("\n--------- 2. INDIVIDUAL-ID COLLISIONS (all 253) ---------\n")
m <- read.csv("02_Metadata/GSE48350_metadata_grouped.csv")
m$blk <- ifelse(as.numeric(sub("^GSM","",m$GSM)) > 1e6, "AD_block","Ctrl_block")
tb <- table(m$Individual, m$blk)
both <- rownames(tb)[tb[,1] > 0 & tb[,2] > 0]
cat("individual IDs in BOTH blocks:", if(length(both)) paste(both,collapse=", ") else "none", "\n\n")
if (length(both)) {
  sub <- m[m$Individual %in% both, c("GSM","Individual","Brain_region","Sex","Age","Diagnosis","blk")]
  print(sub[order(sub$Individual, sub$Brain_region, sub$blk),])
  write.csv(sub, file.path(OUT,"crossblock_individuals.csv"), row.names=FALSE)
  cat("\npaired by region:\n")
  for (id in both) for (r in unique(sub$Brain_region[sub$Individual==id])) {
    p <- sub[sub$Individual==id & sub$Brain_region==r,]
    if (nrow(p)==2) cat(sprintf("  ind %s %-24s %s (%s) <-> %s (%s)\n", id, r,
      p$GSM[1], p$Diagnosis[1], p$GSM[2], p$Diagnosis[2]))
  }
}

## ---- 3. are the sex mismatches duplicates? -------------------------------
cat("\n--------- 3. SEX MISMATCHES vs DUPLICATION ---------\n")
sf <- file.path("04_ML/Critique_Runs","R5_GSE48350_sex_check.csv")
if (file.exists(sf)) {
  s <- read.csv(sf); mm <- s$GSM[!s$agree]
  cat("mismatched samples:", paste(mm, collapse=", "), "\n\n")
  E <- as.matrix(read.csv("03_Preprocessing/GSE48350_RMA_genelevel.csv",
                          row.names=1, check.names=FALSE))
  colnames(E) <- key(colnames(E))
  C <- cor(E, method="spearman"); diag(C) <- NA
  for (g in mm) if (g %in% colnames(E)) {
    v <- C[g,]; b <- names(which.max(v))
    cat(sprintf("  %-12s closest: %-12s rho=%.5f  %s\n", g, b, max(v,na.rm=TRUE),
      ifelse(max(v,na.rm=TRUE)>0.999,"<-- DUPLICATE","")))
  }
  ind <- m$Individual[match(mm, m$GSM)]
  cat("\nindividual IDs of mismatched samples:", paste(ind, collapse=", "), "\n")
  cat("of which appear in both blocks:", paste(intersect(ind, both), collapse=", "), "\n")
}

## ---- 4/5. impact and sensitivity -----------------------------------------
cat("\n--------- 4. IMPACT ON INTERNAL ESTIMATES ---------\n")
tr <- read.csv("04_ML/GSE48350_RMA_train_top25var_age_adjusted.csv",
               row.names=1, check.names=FALSE)
if (nrow(tr) > ncol(tr)) tr <- as.data.frame(t(tr))
rownames(tr) <- key(rownames(tr))
mt <- read.csv("04_ML/GSE48350_RMA_train_meta.csv")
gc <- names(mt)[which(sapply(mt,function(c) any(grepl("^GSM",as.character(c)))))[1]]
dc <- grep("diagnos|group|status", names(mt), ignore.case=TRUE, value=TRUE)[1]
mt[[gc]] <- key(as.character(mt[[gc]])); mt <- mt[match(rownames(tr), mt[[gc]]),]
y <- as.integer(grepl("^ad$|alzh|affect", mt[[dc]], ignore.case=TRUE))
SIG <- c("ABCA6","CRLF1","TNFRSF11B")
ev <- function(idx) {
  X <- scale(tr[idx,SIG]); fit <- glm(y[idx] ~ X, family=binomial)
  ap <- as.numeric(pROC::auc(pROC::roc(y[idx], predict(fit,type="response"), quiet=TRUE)))
  set.seed(42); k <- sample(rep(1:5, length.out=length(idx)))
  pr <- numeric(length(idx))
  for (i in 1:5) { f <- glm(y[idx][k!=i] ~ X[k!=i,], family=binomial)
    pr[k==i] <- as.numeric(plogis(cbind(1,X[k==i,,drop=FALSE]) %*% coef(f))) }
  cv <- as.numeric(pROC::auc(pROC::roc(y[idx], pr, quiet=TRUE)))
  c(apparent=ap, cv5=cv, n=length(idx), cases=sum(y[idx])) }
allx <- seq_len(nrow(tr))
res <- rbind(as_deposited = ev(allx))
drop <- "GSM1176215"
if (drop %in% rownames(tr)) {
  res <- rbind(res, duplicate_removed = ev(setdiff(allx, match(drop, rownames(tr)))))
}
print(round(res,4))
write.csv(res, file.path(OUT,"duplicate_sensitivity.csv"))
cat("\nApparent AUC changes by",
    round(res["duplicate_removed","apparent"]-res["as_deposited","apparent"],4),
    "when the mislabelled duplicate is removed.\n")
cat("\nOutputs in", OUT, "\n"); sink()
