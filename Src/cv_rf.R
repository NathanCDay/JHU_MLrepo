# randomForest Cv
doParallel::registerDoParallel(cores = 5)
fin_train <- readRDS("Objects/fin_train.RDS")

cv_fits_rf <- list()
cv_predicts_rf <- list()

time_k10_rf <- system.time(
  for (i in unique(fin_train$opt_folds)) {
    
    sub_train <- dplyr::filter(fin_train, opt_folds != i)
    sub_test <- dplyr::filter(fin_train, opt_folds == i)
    
    fit <- caret::train(classe ~ ., method = "rf", data = sub_train)
    cv_fits_rf[[i]] <- fit
    
    
    sub_test$pred <- predict(fit, sub_test)
    cv_predicts_rf[[i]] <- sub_test
  }
)

saveRDS(cv_fits_rf, "Objects/cv_fits_rf.RDS")
saveRDS(cv_predicts_rf, "Objects/cv_predicts_rf.RDS")