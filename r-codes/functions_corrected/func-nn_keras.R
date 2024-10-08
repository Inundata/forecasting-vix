runnn=function(Y,indice,lag){
  
  #dum=Y[,ncol(Y)]
  #Y=Y[,-ncol(Y)]
  #comp=princomp(scale(Y,scale=FALSE))
  #Y2=cbind(Y,comp$scores[,1:4])
  
  X=embed(as.matrix(Y),4)
  
  Xin=X[-c((nrow(X)-lag+1):nrow(X)),]
  Xout=X[nrow(X),]
  
  y=tail(Y[,1],nrow(Xin))
  X = Xin 
  X.out = Xout
  
  mean <- apply(X, 2, mean)
  sd <- apply(X, 2, sd)
  
  X <- scale(X, center=mean, scale=sd)
  X.out <- scale(t(X.out), center=mean, scale=sd)

  build_model <- function(){
    
  model <- keras_model_sequential() %>%
    layer_dense(units=32, activation='relu',input_shape = dim(X)[[2]]) %>%
    layer_dropout(rate=0.2) %>% 
    layer_dense(units=16, activation='relu') %>%
    layer_dropout(rate=0.2) %>%
    layer_dense(units=1)
    
    model %>% compile(optimizer = 'rmsprop',
                      loss ='mse',
                      metric = c('mae'))
  }
  
  model <- build_model()
  
  model %>% fit(X, y, epochs = 100, batch_size=50)
  
  pred = model %>% predict(X.out)
  
  return(list("model"=model,"pred"=pred))
}


nn.rolling.window=function(Y,npred,indice=1,lag=1){
  
  save.pred=matrix(NA,npred,1)
  for(i in npred:1){
    Y.window=Y[(1+npred-i):(nrow(Y)-i),]
    lasso=runnn(Y.window,indice,lag)
    save.pred[(1+npred-i),]=lasso$pred
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

