runcsr=function(Y,indice,lag){
  #comp=princomp(scale(Y,scale=FALSE))
  
  X=embed(as.matrix(Y),4)
  
  Xin=X[-c((nrow(X)-lag+1):nrow(X)),]
  Xout=X[nrow(X),]
  
  y=tail(Y[,1],nrow(Xin))
  X = Xin 
  X.out = Xout
  
  f.seq=seq(indice,ncol(X),ncol(Y))
  
  model=HDeconometrics::csr(x=X,y,fixed.controls = f.seq)
  pred=predict(model,X.out)
  
  return(list("model"=model,"pred"=pred))
}


csr.rolling.window=function(Y,npred,indice=1,lag=1){
  
  save.pred=matrix(NA,npred,1)
  for(i in npred:1){
    Y.window=Y[(1+npred-i):(nrow(Y)-i),]
    cs=runcsr(Y.window,indice,lag)
    save.pred[(1+npred-i),]=cs$pred
    cat("iteration",(1+npred-i),"\n")
  }
  
  real=Y[,indice]
  plot(real,type="l")
  lines(c(rep(NA,length(real)-npred+lag-1),save.pred),col="red")
  
  rmse=sqrt(mean((tail(real,npred-lag+1)-save.pred)^2))
  mae=mean(abs(tail(real,npred-lag+1)-save.pred))
  errors=c("rmse"=rmse,"mae"=mae)
  
  return(list("pred"=save.pred,"coef"=save.coef,"errors"=errors))
  
}
