# LDA Cv

fin_train <- readRDS("Objects/fin_train.RDS")

cv_fits_lda <- list()
cv_predicts_lda <- list()

time_k10_lda <- system.time(
  for (i in unique(fin_train$opt_folds)) {
    
    sub_train <- dplyr::filter(fin_train, opt_folds != i)
    sub_test <- dplyr::filter(fin_train, opt_folds == i)
    
    fit <- caret::train(classe ~ ., method = "lda", data = sub_train)
    cv_fits_lda[[i]] <- fit
    
    
    sub_test$pred <- predict(fit, sub_test)
    cv_predicts_lda[[i]] <- sub_test
  }
)

saveRDS(cv_fits_lda, "Objects/cv_fits_lda.RDS")
saveRDS(cv_predicts_lda, "Objects/cv_predicts_lda.RDS")