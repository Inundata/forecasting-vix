runrf=function(Y,indice,lag){
  #comp=princomp(scale(Y,scale=FALSE))
  #Y2=cbind(Y,comp$scores[,1:4])

  X=embed(as.matrix(Y),4)
  
  Xin=X[-c((nrow(X)-lag+1):nrow(X)),]
  Xout=X[nrow(X),]
  
  y=tail(Y[,1],nrow(Xin))
  X = Xin 
  X.out = Xout
  
  model <- foreach(ntree = rep(167, 3), .combine=randomForest::combine, .multicombine=TRUE, .packages = "randomForest") %dopar% randomForest(X,y,ntree=ntree, importance=TRUE)
  pred=predict(model,X.out)
  
  return(list("model"=model,"pred"=pred))
}


rf.rolling.window=function(Y,npred,indice=1,lag=1){
  
  save.importance=list()
  save.pred=matrix(NA,npred,1)
  for(i in npred:1){
    Y.window=Y[(1+npred-i):(nrow(Y)-i),]
    lasso=runrf(Y.window,indice,lag)
    save.pred[(1+npred-i),]=lasso$pred
    save.importance[[i]]=importance(lasso$model)
    cat("iteration",(1+npred-i),"\n")
  }
  
  real=Y[,indice]
  plot(real,type="l")
  lines(c(rep(NA,length(real)-npred+lag-1),save.pred),col="red")
  
  rmse=sqrt(mean((tail(real,npred-lag+1)-save.pred)^2))
  mae=mean(abs(tail(real,npred-lag+1)-save.pred))
  errors=c("rmse"=rmse,"mae"=mae)
  
  return(list("pred"=save.pred,"errors"=errors,"save.importance"=save.importance))
  
}
