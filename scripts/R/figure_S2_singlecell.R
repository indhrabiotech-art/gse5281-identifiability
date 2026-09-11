rm(list=ls()); setwd(path.expand("~/project_ml"))
OUTF <- "06_Manuscript/Submission_Figures"; dir.create(OUTF, showWarnings=FALSE, recursive=TRUE)
GREY <- "#b8c4d4"; RED <- "#c1272d"

ps <- read.csv("MANUSCRIPT_DATA/SC_GSE138852_ABCA6_CRLF1_pseudobulk_statistics.csv")
cat("pseudobulk columns:\n"); print(names(ps))
has <- function(df,pat) apply(df,1,function(r) any(grepl(pat,r,ignore.case=TRUE)))
row <- ps[has(ps,"ABCA6") & has(ps,"oligo") & !has(ps,"opc|precursor"), , drop=FALSE]
cat("\nABCA6 oligodendrocyte row:\n"); print(row)
grab <- function(pat){ k <- grep(pat, names(row), ignore.case=TRUE)
  if(!length(k)) stop("no column matching: ", pat); as.numeric(row[[k[1]]])[1] }
ad_expr <- grab("^ad.*mean|mean.*ad|ad_expr|AD_mean")
ct_expr <- grab("^(ct|control).*mean|mean.*(ct|control)|control_expr")
fold <- ad_expr/ct_expr
cat(sprintf("\nAD %.4f  control %.4f  fold %.2f\n", ad_expr, ct_expr, fold))

lib <- read.csv("MANUSCRIPT_DATA/SC_GSE138852_composition_expression_library_table.csv")
is_ad <- grepl("^ad", lib$condition, ignore.case=TRUE)

draw <- function(){
  par(mfrow=c(1,2), mar=c(4.4,4.4,3.6,1), mgp=c(2.6,0.7,0))
  bp <- barplot(c(ct_expr, ad_expr), names.arg=c("Control","AD"),
                col=c(GREY,RED), border="black", las=1,
                ylim=c(0, max(ad_expr,ct_expr)*1.40), ylab="mean expression")
  text(bp, c(ct_expr,ad_expr), sprintf("%.3f", c(ct_expr,ad_expr)), pos=3, cex=0.8)
  mtext("a", side=3, line=2.2, adj=0, font=2, cex=1.1)
  mtext("ABCA6 in oligodendrocytes", side=3, line=2.2, cex=0.85)
  mtext(sprintf("%.2f-fold, Cohen's d = 1.59\npermutation p = 0.2, FDR = 0.80  (3 vs 3 libraries)",
        fold), side=3, line=0.1, cex=0.64, col=RED)
  barplot(lib$Oligodendrocyte_percentage, names.arg=lib$sample_group,
          col=ifelse(is_ad,RED,GREY), border="black", las=2, ylim=c(0,100),
          cex.names=0.7, ylab="oligodendrocyte percentage")
  mtext("b", side=3, line=2.2, adj=0, font=2, cex=1.1)
  mtext("Cell-type composition by library", side=3, line=2.2, cex=0.85)
  mtext("Composition differs in the same direction as the signal",
        side=3, line=0.4, cex=0.64, col=RED)
  legend("topright", c("Control","AD"), fill=c(GREY,RED), border="black",
         bty="n", cex=0.75)
}
png(file.path(OUTF,"Figure_S2.png"), width=2100, height=1100, res=300); draw(); dev.off()
pdf(file.path(OUTF,"Figure_S2.pdf"), width=7.0, height=3.7); draw(); dev.off()
cat("\nwritten: Figure_S2.png and .pdf\n")
cat(sprintf("CHECK: fold %.2f  (manuscript states 6.06)\n", fold))
