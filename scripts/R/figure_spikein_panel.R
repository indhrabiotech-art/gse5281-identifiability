rm(list=ls()); setwd(path.expand("~/project_ml"))
suppressPackageStartupMessages({library(affy); library(pROC)})
OUTF <- "06_Manuscript/Submission_Figures"; dir.create(OUTF, showWarnings=FALSE, recursive=TRUE)
GREY <- "#b8c4d4"; RED <- "#c1272d"; GREYT <- "#4a4a4a"
key <- function(x) sub("\\.CEL.*$","",sub("_.*$","",x))
dirs <- list.dirs(".", recursive=TRUE); dirs <- dirs[grepl("5281", dirs)]
cels <- unlist(lapply(dirs, function(d) list.files(d, pattern="\\.CEL(\\.gz)?$",
                 full.names=TRUE, ignore.case=TRUE)))
cels <- cels[!duplicated(basename(cels))]
g <- key(basename(cels))
m <- read.csv("02_Metadata/GSE5281_hippocampus_metadata.csv")
dx <- ifelse(grepl("^ad$|alzh|affect", m$diagnosis[match(g,m$GSM)], ignore.case=TRUE),"AD","Control")
ok <- !is.na(dx); cels <- cels[ok]; g <- g[ok]; dx <- dx[ok]
E <- exprs(rma(ReadAffy(filenames=cels), verbose=FALSE)); colnames(E) <- g
pick <- function(p) grep(p, rownames(E), value=TRUE)
hyb <- unique(c(pick("^AFFX-BioB"),pick("^AFFX-BioC"),pick("^AFFX-BioDn"),pick("^AFFX-CreX")))
pol <- unique(c(pick("AFFX.*LysX"),pick("AFFX.*PheX"),pick("AFFX.*ThrX"),pick("AFFX.*DapX")))
sc <- function(ids) colMeans(E[ids,,drop=FALSE])
draw <- function(){
  par(mfrow=c(1,2), mar=c(4.0,4.4,3.4,1), mgp=c(2.6,0.7,0))
  panel <- function(v,ttl,lab){
    a <- as.numeric(auc(roc(dx, v, quiet=TRUE)))
    p <- wilcox.test(v ~ dx, exact=FALSE)$p.value
    boxplot(v ~ factor(dx, levels=c("Control","AD")), col=c(GREY,RED),
            border="grey25", ylab=expression(log[2]*" intensity"), xlab="",
            ylim=range(c(sc(hyb),sc(pol)))*c(0.95,1.05), las=1)
    points(jitter(as.numeric(factor(dx, levels=c("Control","AD"))),0.6), v,
           pch=21, bg="white", cex=0.8)
    mtext(lab, side=3, line=2.0, adj=0, font=2, cex=1.1)
    mtext(ttl, side=3, line=2.0, cex=0.85)
    mtext(sprintf("cohort ROC-AUC %.3f   Wilcoxon p = %.3g", a, p),
          side=3, line=0.4, cex=0.66, col=if(a>0.9) RED else GREYT)
  }
  panel(sc(pol), "Poly-A controls (pre-RT)", "a")
  panel(sc(hyb), "Hybridisation controls (post-label)", "b")
}
png(file.path(OUTF,"Figure_1d_spikein.png"), width=2000, height=1100, res=300); draw(); dev.off()
pdf(file.path(OUTF,"Figure_1d_spikein.pdf"), width=6.7, height=3.7); draw(); dev.off()
cat("written: Figure_1d_spikein\n")
cat(sprintf("poly-A  AUC %.3f\nhyb     AUC %.3f\n",
  as.numeric(auc(roc(dx, sc(pol), quiet=TRUE))),
  as.numeric(auc(roc(dx, sc(hyb), quiet=TRUE)))))
