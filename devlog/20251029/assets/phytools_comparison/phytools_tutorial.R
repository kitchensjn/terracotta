library(phytools)
library(expm)

pruning<-function(q,tree,x,model=NULL,...){
  if(hasArg(return)) return<-list(...)$return
  else return<-"likelihood"
  pw<-reorder(tree,"postorder")
  k<-length(levels(x))
  if(is.null(model)){
    model<-matrix(1,k,k)
    diag(model)<-0
  }
  if(hasArg(pi)) pi<-list(...)$pi
  else pi<-rep(1/k,k)
  Q<-matrix(0,k,k)
  Q[]<-c(0,q)[model+1]
  diag(Q)<--rowSums(Q)
  L<-rbind(to.matrix(x[pw$tip.label],levels(x)),
           matrix(0,tree$Nnode,k,dimnames=
                    list(1:tree$Nnode+Ntip(tree))))
  nn<-unique(pw$edge[,1])
  for(i in 1:length(nn)){
    ee<-which(pw$edge[,1]==nn[i])
    PP<-matrix(NA,length(ee),k)
    for(j in 1:length(ee)){
      P<-expm(Q*pw$edge.length[ee[j]])
      PP[j,]<-P%*%L[pw$edge[ee[j],2],]
    }
    L[nn[i],]<-apply(PP,2,prod)
  }
  prob<-log(sum(pi*L[nn[i],]))
  if(return=="likelihood") prob
  else if(return=="conditional") L
}


data(eel.data)
data(eel.tree)

write.tree(eel.tree, file="/Users/jameskitchens/Documents/GitHub/terracotta/devlog/20251029/assets/phytools_comparison/tutorial_tree.newick")

feeding_mode<-setNames(eel.data$feed_mode,rownames(eel.data))
plotTree(eel.tree,lwd=1,ftype="i",direction="upwards",offset=1,
         fsize=0.7)
tiplabels(pie=to.matrix(feeding_mode[eel.tree$tip.label],
                        levels(feeding_mode)),cex=0.3,
          piecol=viridisLite::viridis(n=2))
legend("topleft",levels(feeding_mode),pch=21,pt.cex=2,
       pt.bg=viridisLite::viridis(n=2),bty="n",cex=0.8)


model<-matrix(c(0,1,2,0),2,2,byrow=TRUE)
fitted<-optim(c(1,1),pruning,tree=eel.tree,x=feeding_mode,
              model=model,method="L-BFGS-B",lower=1e-12,
              control=list(fnscale=-1))
fitted


fitMk(eel.tree,feeding_mode,model="ARD")



