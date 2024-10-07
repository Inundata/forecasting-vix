runrf=function(Y,indice,lag,selected){
  #comp=princomp(scale(Y,scale=FALSE))
  #Y2=cbind(Y,comp$scores[,1:4])

  X=embed(as.matrix(Y),4)
  
  Xin=X[-c((nrow(X)-lag+1):nrow(X)),]
  Xout=X[nrow(X),]
  
  y=tail(Y[,1],nrow(Xin))
  X = Xin 
  X.out = Xout
  
  model <- foreach(ntree = rep(167, 3), .combine=randomForest::combine, .multicombine=TRUE, .packages = "randomForest") %dopar% randomForest(X[,selected],y,ntree=ntree, importance=TRUE)
  pred=predict(model,X.out[selected])
  
  return(list("model"=model,"pred"=pred))
}


rf.rolling.window=function(Y,nprev,indice=1,lag=1,selected){
  
  save.importance=list()
  save.pred=matrix(NA,nprev,1)
  for(i in nprev:1){
    Y.window=Y[(1+nprev-i):(nrow(Y)-i),]
    lasso=runrf(Y.window,indice,lag,selected)
    save.pred[(1+nprev-i),]=lasso$pred
    save.importance[[i]]=importance(lasso$model)
    cat("iteration",(1+nprev-i),"\n")
  }
  
  real=Y[,indice]
  plot(real,type="l")
  lines(c(rep(NA,length(real)-npred+lag-1),save.pred),col="red")
  
  rmse=sqrt(mean((tail(real,npred-lag+1)-save.pred)^2))
  mae=mean(abs(tail(real,npred-lag+1)-save.pred))
  errors=c("rmse"=rmse,"mae"=mae)
  
  return(list("pred"=save.pred,"errors"=errors,"save.importance"=save.importance))
}
