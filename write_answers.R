answers = rep("A", 20)
  
pml_write_files = function(x){
    n = length(x)
    for(i in 1:n){
      filename = paste0(".answers/problem_id_",i,".txt")
      write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}

pml_write_files(answers)

# load("./data/gbm.RData")
gbm
answers <- predict(gbm, test_scaled)
names(train_scaled)
names(test_scaled)
