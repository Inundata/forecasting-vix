
rw.rolling.window=function(Y,npred,indice=1,lag=1){
  
  YY=as.matrix(Y[,indice])
  
  save.pred=matrix(NA,npred,1)
  
  for(i in npred:1){
    pred=YY[(1+nrow(YY)-i-lag)]
    save.pred[(1+npred-i),]=pred
    cat("iteration",(1+npred-i),"\n")
  }
  
  real=Y[,indice]
  plot(real,type="l")
  lines(c(rep(NA,length(real)-npred+lag-1),save.pred),col="red")
  
  rmse=sqrt(mean((tail(real,npred-lag+1)-save.pred)^2))
  mae=mean(abs(tail(real,npred-lag+1)-save.pred))
  errors=c("rmse"=rmse,"mae"=mae)
  
  return(list("pred"=save.pred, "errors"=errors))
  
}
