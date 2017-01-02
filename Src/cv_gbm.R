# Stochastic Gradient Boost Cv
doParallel::registerDoParallel(cores = 5)
fin_train <- readRDS("Objects/fin_train.RDS")

cv_fits_gbm <- list()
cv_predicts_gbm <- list()

time_k10_gbm <- system.time(
  for (i in unique(fin_train$opt_folds)) {
    
    sub_train <- dplyr::filter(fin_train, opt_folds != i)
    sub_test <- dplyr::filter(fin_train, opt_folds == i)
    
    fit <- caret::train(classe ~ ., method = "gbm", data = sub_train)
    cv_fits_gbm[[i]] <- fit
    
    
    sub_test$pred <- predict(fit, sub_test)
    cv_predicts_gbm[[i]] <- sub_test
  }
)

saveRDS(cv_fits_gbm, "Objects/cv_fits_gbm.RDS")
saveRDS(cv_predicts_gbm, "Objects/cv_predicts_gbm.RDS")