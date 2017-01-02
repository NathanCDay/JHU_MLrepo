doParallel::registerDoParallel(cores = 5)
fin_train <- readRDS("~/Projects/mlProject/Objects/fin_train.RDS")

cv_fits_ada <- list()
cv_predicts_ada <- list()

time_k10_ada <- system.time(
  for (i in unique(fin_train$opt_folds)) {
    
    sub_train <- dplyr::filter(fin_train, opt_folds != i)
    sub_test <- dplyr::filter(fin_train, opt_folds == i)
    
    fit <- caret::train(classe ~ ., method = "AdaBoost.M1", data = sub_train)
    cv_fits_ada[[i]] <- fit
    
    
    sub_test$pred <- predict(fit, sub_test)
    cv_predicts_ada[[i]] <- sub_test
  }
)

saveRDS(cv_fits_ada, "~/Projects/mlProject/Objects/cv_fits_ada.RDS")
saveRDS(cv_predicts_ada, "~/Projects/mlProject/Objects/cv_predicts_ada.RDS")
