
doParallel::registerDoParallel(cores = 5)

fin_train <- readRDS("~/Projects/mlProject/Objects/fin_train.RDS")

ada_time <- system.time(ada_fit <- caret::train(classe ~ ., method = "AdaBoost.M1", data = fin_train))
saveRDS(ada_fit, "~/Projects/mlProject/Objects/ada_fit.RDS")