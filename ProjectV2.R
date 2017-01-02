#### Weigth Lifting Project
library(caret)
library(magrittr)
library(tidyverse)
library(randomForest)
library(MASS)
library(adabag)
library(e1071)
library(cowplot)
library(ggbeeswarm)
library(doParallel)
registerDoParallel(cores = 5)

#### ins ####
all <- data.table::fread("~/Projects/mlProject/pml-training.csv",
                            na.strings = c("NA", " ", "", "#DIV/0!")) # lots of empty string columns
quiz <- data.table::fread("~/Projects/mlProject/pml-testing.csv",
                         na.strings = c("NA", " ", "", "#DIV/0!"))

# drop non predictors
all %<>% select(-c(1:7))

#### split ####
index <- createDataPartition(all$classe, p = .8, list = F)

trainy <- all[index, ]
ids <- select(trainy, 153)
ids %<>% map_df(as.factor)
preds <- select(trainy, -153)


#### CLEAN ####
# want all numeric
preds %<>% map_df(as.numeric)
# lots of mostly  NA columns
num_NAs <- apply(preds, 2, is.na) %>% apply(2, sum)
good_cols <- grep(T, sapply(num_NAs, function(x){x < 15000}))
preds %<>% select(good_cols)




#looks for low variablity predictors
#remove near zero covars (no variablity)
near_zero <- nearZeroVar(preds, saveMetrics = T) %>% rownames_to_column()
preds %<>% select(grep(F, near_zero$nzv))

fin_train <- cbind(ids, preds)

fin_train$opt_folds <- createFolds(ids$classe, list = F)
saveRDS(fin_train, "Objects/fin_train.RDS")

cv_time <- system.time(crossv_rf <- rfcv(select(fin_train, -classe), fin_train$class, cv.fold = 10))

fold_vis_cart <- data.frame()
fold_fits_cart <- list()
time_k10_rf <- system.time(
for (i in unique(fin_train$opt_folds)) {
  sub_df <- filter(fin_train, opt_folds == i)
  fit <- train(classe ~ ., method = "rpart", data = sub_df)
  fold_fits_cart[[i]] <- fit
  vi <- varImp(fit)$importance %>% rownames_to_column("pred") %>% mutate(fold = i)
  fold_vis_cart %<>% rbind(vi)
}
)
fold_vis_cart %<>% filter(pred != "opt_folds")
ggplot(fold_vis_cart, aes(x = pred, y = Overall)) +
  geom_text(aes(label = fold))




fold_vis_gbm <- data.frame()
fold_fits_gbm <- list()
time_k10_rf <- system.time(
  for (i in unique(fin_train$opt_folds)) {
    sub_df <- filter(fin_train, opt_folds == i)
    fit <- train(classe ~ ., method = "rpart", data = sub_df)
    fold_fits_gbm[[i]] <- fit
    vi <- varImp(fit)$importance %>% rownames_to_column("pred") %>% mutate(fold = i)
    fold_vis_gbm %<>% rbind(vi)
  }
)
fold_vis_gbm %<>% filter(pred != "opt_folds")

fv1 <- ggplot(fold_vis_gbm, aes(x = pred, y = Overall)) +
  geom_text(aes(label = fold))


# plot CV results
cf_plots <- list()

# lda cv vis
lda_folds <- readRDS("~/Projects/mlProject/Objects/cv_fits_lda.RDS")
lda_times <- lda_folds %>% map(~ use_series(., times)) %>% map(~ use_series(., everything)) %>%
  do.call(rbind, .) %>% as.data.frame() %>% mutate(method = "LDA")
lda_vis <- map(lda_folds, ~varImp(.)$importance) %>% map(~rownames_to_column(.,"pred")) %>% bind_rows()
lda_vis$fold <- rep(1:10, each = 53)
lda_vis %<>% filter(pred != "opt_fold")
lda_vis_avg <- lda_vis %>% group_by(pred) %>% dplyr::summarise(imp_avg = mean(Overall)) %>% arrange(-imp_avg)

ggplot(lda_vis, aes(x = factor(pred, levels = lda_vis_avg$pred), y = Overall)) +
  geom_quasirandom(alpha = .5) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90)) +
  labs(title = "Linear Discriminant Analysis's Variable Importance",
       subtitle = "10 Fold Cross Validation",
       y = "Variable Importance (Scaled 0-100)",
       x = "Variable")

lda_preds <- readRDS("~/Projects/mlProject/Objects/cv_predicts_lda.RDS")
lda_cfs <- lda_preds %>% map(~ confusionMatrix(.$pred, .$class))
lda_accs <- lda_cfs %>% map(~use_series(.,overall)[1]) %>% do.call(rbind, .) %>%
  as.data.frame() %>% dplyr::rename(LDA = Accuracy)
lda_plot_cfs <- lda_cfs %>% map(~use_series(., table)) %>% map(as.data.frame) %>% do.call(rbind,.) %>%
  mutate(fold = rep(1:10, each = 25))
lda_plot_cfs$Prediction %<>% factor(levels = LETTERS[5:1])

cf_plots[[1]] <- ggplot(data = lda_plot_cfs, aes(x = Reference, y = Prediction, fill = log(Freq+1))) +
  geom_tile() +
  geom_text(aes(label = Freq), size = 3) +
  theme_classic() +
  scale_fill_gradient(low = "white", high = "#7570B3", guide = "none") +
  facet_wrap(~fold, nrow = 5, labeller = "label_both") +
  labs(title = "LDA's Confusion Matrices",
       subtitle = "10 Fold Cross Validation ")

# rf cv viz

rf_folds <- readRDS("~/Projects/mlProject/Objects/cv_fits_rf.RDS")
rf_times <- rf_folds %>% map(~ use_series(., times)) %>% map(~ use_series(., everything)) %>%
  do.call(rbind, .) %>% as.data.frame() %>% mutate(method = "RF")
rf_vis <- map(rf_folds, ~varImp(.)$importance) %>% map(~rownames_to_column(.,"pred")) %>% bind_rows()
rf_vis$fold <- rep(1:10, each = 53)
rf_vis %<>% filter(pred != "opt_fold")
rf_vis$method <- "RF"
rf_vis_avg <- rf_vis %>% group_by(pred) %>% dplyr::summarise(imp_avg = mean(Overall)) %>% arrange(-imp_avg)

ggplot(rf_vis, aes(x = factor(pred, levels = rf_vis_avg$pred), y = Overall)) +
  geom_quasirandom(alpha = .5) +
  theme_classic() +
  theme(axis.text.x = element_text(angle = 90)) +
  labs(title = "RF's Variable Importance",
       subtitle = "10 Fold Cross Validation",
       y = "Variable Importance (Scaled 0-100)",
       x = "Variable")

rf_preds <- readRDS("~/Projects/mlProject/Objects/cv_predicts_rf.RDS")
rf_cfs <- rf_preds %>% map(~ confusionMatrix(.$pred, .$class))
rf_accs <- rf_cfs %>% map(~use_series(.,overall)[1]) %>% do.call(rbind, .) %>%
  as.data.frame() %>% dplyr::rename(RF = Accuracy)
rf_plot_cfs <- rf_cfs %>% map(~use_series(., table)) %>% map(as.data.frame) %>% do.call(rbind,.) %>%
  mutate(fold = rep(1:10, each = 25))
rf_plot_cfs$Prediction %<>% factor(levels = LETTERS[5:1])

cf_plots[[2]] <- ggplot(data = rf_plot_cfs, aes(x = Reference, y = Prediction, fill = log(Freq+1))) +
  geom_tile() +
  geom_text(aes(label = Freq), size = 3) +
  theme_classic() +
  scale_fill_gradient(low = "white", high = "#D95F02", guide = "none") +
  facet_wrap(~fold, nrow = 5, labeller = "label_both") +
  labs(title = "RF's Confusion Matrices",
       subtitle = "10 Fold Cross Validation ")


# gbm cv vis
gbm_folds <- readRDS("~/Projects/mlProject/Objects/cv_fits_gbm.RDS")
gbm_times <- gbm_folds %>% map(~ use_series(., times)) %>% map(~ use_series(., everything)) %>%
  do.call(rbind, .) %>% as.data.frame() %>% mutate(method = "GBM")

gbm_vis <- map(gbm_folds, ~varImp(.)$importance) %>% map(~rownames_to_column(.,"pred")) %>% bind_rows()
gbm_vis$fold <- rep(1:10, each = 53)
gbm_vis %<>% filter(pred != "opt_fold")
gbm_vis$method <- "GBM"
gbm_vis_avg <- gbm_vis %>% group_by(pred) %>% dplyr::summarise(imp_avg = mean(Overall)) %>% arrange(-imp_avg)
saveRDS(gbm_vis_avg, "~/Projects/mlProject/Objects/gbm_vis_avg.RDS")

ggplot(gbm_vis, aes(x = factor(pred, levels = gbm_vis_avg$pred), y = Overall)) +
  geom_quasirandom(alpha = .5) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90)) +
  labs(title = "GBM's Variable Importance",
       subtitle = "10 Fold Cross Validation",
       y = "Variable Importance (Scaled 0-100)",
       x = "Variable")

gbm_preds <- readRDS("~/Projects/mlProject/Objects/cv_predicts_gbm.RDS")
gbm_cfs <- gbm_preds %>% map(~ confusionMatrix(.$pred, .$class))
gbm_accs <- gbm_cfs %>% map(~use_series(.,overall)[1]) %>% do.call(rbind, .) %>%
  as.data.frame() %>% dplyr::rename(GBM = Accuracy)
gbm_plot_cfs <- gbm_cfs %>% map(~use_series(., table)) %>% map(as.data.frame) %>% do.call(rbind,.) %>%
  mutate(fold = rep(1:10, each = 25))
gbm_plot_cfs$Prediction %<>% factor(levels = LETTERS[5:1])

cf_plots[[3]] <- ggplot(data = gbm_plot_cfs, aes(x = Reference, y = Prediction, fill = log(Freq+1)))  +
  geom_tile() +
  geom_text(aes(label = Freq), size = 3) +
  scale_fill_gradient(low = "white", high = "#1B9E77", guide = "none") +
  facet_wrap(~fold, nrow = 5, labeller = "label_both") +
  theme_classic() +
  labs(title = "GBM's Confusion Matrices",
       subtitle = "10 Fold Cross Validation ")

#### final confustion matrixies
saveRDS(cf_plots, "~/Projects/mlProject/Objects/cf_plots.RDS")
plot_grid(plotlist = cf_plots, ncol = 3)


# panel acc and importance
panel2 <- list()

all_accs <- cbind(lda_accs, rf_accs, gbm_accs) %>% mutate(fold = 1:10)
all_accs %<>% map_df(~round(.,3))
all_accs %<>% gather(method, acc, -fold)
all_accs$method %<>% factor(levels = c("RF", "GBM", "LDA"))

panel2[[2]] <- ggplot(all_accs, aes(x = fold, y = acc, color = method)) +
  
  geom_point() +
  theme_minimal() +
  theme(legend.position = "bottom") +
  geom_smooth(method = "lm", se = F) +
  geom_hline(yintercept = .5, linetype = 3) +
  annotate("text", label = "Futility", y = .52, x = 9) +
  scale_y_continuous(breaks = seq(.1, 10, .1)) +
  scale_x_continuous(breaks = 1:10) +
  scale_color_brewer(palette = "Dark2") +
  labs(title = "Accuracy Comparison",
       subtitle = "10 fold Cross Validation",
       y = "Accuracy",
       x = "Fold")


#### fit times #####



all_times <- rbind(lda_times, rf_times, gbm_times) %>% as.data.frame() %>% mutate(fold = 1:10)
  
all_times$method %<>% factor(levels = c("RF", "GBM", "LDA"))

panel2[[1]] <- ggplot(all_times, aes(x = fold, y = elapsed, color = method)) +
  geom_point() +
  theme_minimal() +
  theme(legend.position = "bottom") +
  geom_smooth(method = "lm", se = F) +
  scale_y_log10(breaks = c(10, 100, 1000)) + annotation_logticks(sides = "l") +
  scale_x_continuous(breaks = 1:10) +
  scale_color_brewer(palette = "Dark2") +
  labs(title = "System Time Comparison",
       subtitle = "10 fold Cross Validation",
       y = "Elapsed Execution Time (sec)",
       x = "Fold",
       caption = "cores = 5")
  
saveRDS(panel2, "~/Projects/mlProject/Objects/panel2.RDS")
plot_grid(plotlist = panel2, ncol = 2, align = "h")


all_vis <- rbind(rf_vis, gbm_vis)
all_vis$method %<>% factor(levels = c("RF", "GBM", "LDA"))
saveRDS(all_vis, "~/Projects/mlProject/Objects/all_vis.RDS")


ggplot(all_vis, aes(x = factor(all_vis$pred, levels = gbm_vis_avg$pred), y = Overall, color = method)) +
  geom_quasirandom(alpha = .5) +
  theme_minimal() +
  theme(legend.position = "bottom") +
  theme(axis.text.x = element_text(angle = 90)) +
  scale_color_brewer(palette = "Dark2") +
  labs(title = "Variable Importance Comparison",
       subtitle = "10 Fold Cross Validation",
       y = "Variable Importance (Scaled 0-100)",
       x = "Variable")



fin_train2 <- select(fin_train, -opt_folds)
saveRDS(fin_train2, "Objects/fin_train2.RDS")

fit_control <- trainControl(method = "boot")

cart_time <- system.time(cart_fit <- train(classe ~ ., method = "rpart", data = fin_train))

lda_time <- system.time(lda_fit <- train(classe ~ ., method = "lda", data = fin_train))
saveRDS(lda_fit, "Objects/lda_fit.RDS")

knn_time <- system.time(knn_fit <- train(classe ~ ., method = "knn", data = fin_train))
saveRDS(knn_fit, "~/Projects/mlProject/knn_fit.RDS")

rf_time <- system.time(rf_fit <- train(classe ~ ., method = "rf", data = fin_train))
saveRDS(rf_fit, "~/Projects/mlProject/rf_fit.RDS")

gbm_time <- system.time(gbm_fit <- train(classe ~ .,method = "gbm", data = fin_train))
saveRDS(gbm_fit, "~/Projects/mlProject/gbm_fit.RDS")

ada_time <- system.time(ada_fit <- train(classe ~ ., method = "adaboost", data = fin_train))
saveRDS(ada_fit, "~/Projects/mlProject/ada_fit.RDS")


testy <- all[-index, ]
test_ids <- dplyr::select(testy, 153)
test_ids %<>% map_df(as.factor)
test_preds <- dplyr::select(testy, -153)
test_preds %<>% map_df(as.numeric)
test_preds %<>% dplyr::select(good_cols)
test_preds %<>% dplyr::select(grep(F, near_zero$nzv))
test_preds <- predict(knn_imputer, test_preds)
test_fin <- cbind(test_ids, test_preds)
saveRDS(test_preds, "Objects/test_preds.RDS")
saveRDS(test_fin, "Objects/test_fin.RDS")

# test cart
test_fin$cart <- predict(cart_fit, test_preds)
cf <- confusionMatrix(test_fin$cart, test_fin$classe)
vi <- varImp(cart_fit)$importance %>% rownames_to_column("pred")


# test lda
test_fin$lda <- predict(lda_fit, test_preds)
cf <- confusionMatrix(test_fin$lda, test_fin$classe)
vi <- varImp(lda_fit)$importance %>% rownames_to_column("pred")

# test rf
test_fin$rf <- predict(rf_fit, test_preds)
confusionMatrix(test_fin$rf, test_fin$classe)
rf_vi <- varImp(rf_fit)

# test gbm
test_fin$gbm <- predict(gbm_fit, test_preds)
confusionMatrix(test_fin$gbm, test_fin$classe)
varImp(gbm_fit)

# test ada
test_fin$ada <- predict(ada_fit, test_preds)
confusionMatrix(test_fin$ada, test_fin$classe)
varImp(ada_fit)
