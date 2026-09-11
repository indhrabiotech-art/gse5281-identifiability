#!/usr/bin/env Rscript
rm(list=ls()); options(stringsAsFactors=FALSE, warn=1)
setwd(path.expand("~/project_ml"))
OUT <- "04_ML/Critique_Runs"; dir.create(OUT, showWarnings=FALSE, recursive=TRUE)
sink(file.path(OUT,"critique_runs_R_log.txt"), split=TRUE)
cat("=========== CRITIQUE RUNS (R) ===========\nRun:",format(Sys.time()),"\n\n")

need <- function(p, bioc=TRUE){ if(!requireNamespace(p, quietly=TRUE)){
  if(bioc){ if(!requireNamespace("BiocManager",quietly=TRUE))
      install.packages("BiocManager", repos="https://cloud.r-project.org")
    BiocManager::install(p, ask=FALSE, update=FALSE)
  } else install.packages(p, repos="https://cloud.r-project.org") } }
need("sva"); need("limma"); need("pROC", bioc=FALSE)

key <- function(x) sub("\\.CEL.*$","",sub("_.*$","",x))
e5  <- as.matrix(read.csv("03_Preprocessing/GSE5281_RMA_genelevel.csv",
                          row.names=1, check.names=FALSE))
g5  <- key(colnames(e5))
m5  <- read.csv("02_Metadata/GSE5281_hippocampus_metadata.csv")
dx5 <- factor(ifelse(grepl("^ad$|alzh|affect", m5$diagnosis[match(g5,m5$GSM)],
                     ignore.case=TRUE),"AD","Control"), levels=c("Control","AD"))
coh <- factor(ifelse(as.numeric(sub("^GSM","",g5)) < 200000,"Jul2006","Oct2007"))
cat("GSE5281:",ncol(e5),"samples\n"); print(table(dx5, coh)); cat("\n")

## ---- R1  ComBat demonstration ------------------------------------------
cat("\n---------- R1  ComBat ----------\n")
degs <- function(x){ f <- limma::eBayes(limma::lmFit(x, model.matrix(~dx5)))
  sum(limma::topTable(f, coef=2, number=Inf)$adj.P.Val < 0.05) }
n_before <- degs(e5); cat("DEGs before ComBat (FDR<0.05):", n_before, "\n")
cat("\nComBat WITH diagnosis as covariate:\n")
r <- tryCatch({ sva::ComBat(e5, batch=coh, mod=model.matrix(~dx5)); "RAN" },
              error=function(z) paste("ERROR:", conditionMessage(z)))
cat("  ", r, "\n")
cat("\nComBat WITHOUT diagnosis:\n")
e2 <- tryCatch(sva::ComBat(e5, batch=coh, mod=NULL), error=function(z) NULL)
if(!is.null(e2)){ n_after <- degs(e2)
  cat("  ran. DEGs after ComBat (FDR<0.05):", n_after, "\n")
  cat("  collapse:", n_before, "->", n_after, "\n")
  write.csv(data.frame(stage=c("before","after"), n_DEG=c(n_before,n_after)),
            file.path(OUT,"R1_combat_deg_counts.csv"), row.names=FALSE)
} else cat("  ComBat failed without covariate too\n")

## ---- R5  sex inference --------------------------------------------------
cat("\n---------- R5  sex inference ----------\n")
sexcall <- function(E, ids, lab){
  y <- c("RPS4Y1","KDM5D","DDX3Y","UTY"); y <- y[y %in% rownames(E)]
  hasX <- "XIST" %in% rownames(E)
  if(!length(y)) { cat(lab,": no Y genes found\n"); return(NULL) }
  ys <- colMeans(E[y,,drop=FALSE]); xs <- if(hasX) E["XIST",] else rep(NA,ncol(E))
  sex <- ifelse(ys > mean(range(ys)), "M", "F")
  cat("\n", lab, "  Y genes:", paste(y,collapse=","), "\n"); print(table(sex))
  data.frame(GSM=ids, Y_mean=as.numeric(ys), XIST=as.numeric(xs), inferred_sex=sex)
}
s5 <- sexcall(e5, g5, "GSE5281")
if(!is.null(s5)){ s5$diagnosis <- as.character(dx5)
  print(table(s5$inferred_sex, s5$diagnosis))
  write.csv(s5, file.path(OUT,"R5_GSE5281_inferred_sex.csv"), row.names=FALSE) }

e4 <- as.matrix(read.csv("03_Preprocessing/GSE48350_RMA_genelevel.csv",
                         row.names=1, check.names=FALSE))
g4 <- key(colnames(e4))
s4 <- sexcall(e4, g4, "GSE48350")
if(!is.null(s4)){
  m4 <- read.csv("02_Metadata/GSE48350_metadata_grouped.csv")
  s4$recorded_sex <- m4$Sex[match(g4, m4$GSM)]
  s4$agree <- toupper(substr(s4$recorded_sex,1,1)) == s4$inferred_sex
  cat("\nGSE48350 inferred vs recorded:\n"); print(table(s4$inferred_sex, s4$recorded_sex))
  cat("mismatches:", sum(!s4$agree, na.rm=TRUE), "of", nrow(s4), "\n")
  write.csv(s4, file.path(OUT,"R5_GSE48350_sex_check.csv"), row.names=FALSE) }

## ---- R10  distributions by cohort --------------------------------------
cat("\n---------- R10  GSE5281 distribution by cohort ----------\n")
md <- apply(e5, 2, median)
cat("median log2 by cohort:\n"); print(tapply(md, coh, function(v)
  sprintf("mean %.3f  range %.3f-%.3f", mean(v), min(v), max(v))))
cat("Wilcoxon p =", signif(wilcox.test(md ~ coh)$p.value,4), "\n")
cat("cohort ROC-AUC on median alone:",
    sprintf("%.3f", as.numeric(pROC::auc(pROC::roc(coh, md, quiet=TRUE)))), "\n")
write.csv(data.frame(GSM=g5, cohort=as.character(coh),
                     diagnosis=as.character(dx5), median_log2=md),
          file.path(OUT,"R10_GSE5281_sample_medians.csv"), row.names=FALSE)

cat("\n=========== R BLOCK DONE ===========\n"); sink()
