library(phytools)

squamate.tree <- read.nexus(file="http://www.phytools.org/Rbook/6/squamate.tre")
squamate.data <- read.csv(
  file="http://www.phytools.org/Rbook/6/squamate-data.csv",
  row.names=1
)

chk <- geiger::name.check(squamate.tree, squamate.data)


## prune tree
squamate.tree <- drop.tip(squamate.tree, chk$tree_not_data)

## subsample data
squamate.data <- squamate.data[squamate.tree$tip.label,, drop=FALSE]

## check again
geiger::name.check(squamate.tree, squamate.data)


squamate.toes <- setNames(squamate.data$rear.toes, rownames(squamate.data))

ordered_model <- matrix(
  c(
    0,1,0,0,0,0,
    2,0,3,0,0,0,
    0,4,0,5,0,0,
    0,0,6,0,7,0,
    0,0,0,8,0,9,
    0,0,0,0,10,0
  ),
  6,
  6,
  byrow=TRUE,
  dimnames=list(0:5,0:5)
)

library(foreach)
library(doParallel)
niter<-10 ## set iterations
## set ncores and open cluster
ncores<-min(niter,detectCores()-1)
mc<-makeCluster(ncores,type="PSOCK")
registerDoParallel(cl=mc)
all_fits<-foreach(i=1:niter)%dopar%{ 
  obj<-list()
  class(obj)<-"try-error"
  while(inherits(obj,"try-error")){
    obj<-try(phytools::fitMk(squamate.tree,
                             squamate.toes,model=ordered_model,pi="fitzjohn",
                             logscale=sample(c(FALSE,TRUE),1),
                             opt.method=sample(c("nlminb","optim"),1),
                             rand.start=TRUE))
  }
  obj
}
stopCluster(mc) ## stop cluster
lnL<-sapply(all_fits,logLik)
lnL


