library(gamlr)
library(kableExtra)
library(randomForest)
library(gbm)
library(pdp)
library(mosaic)
library(dummies)
library(haven)
library(pander)
library(flextable)

texas = read.csv("https://raw.githubusercontent.com/bmagalhaes/ECO395M-Final-Project/master/texas.csv")

texas$post = ifelse(texas$statefip == 48 & texas$year >= 1993, 1, 0)
texas = texas %>%
  mutate(year_fixed = as_factor(year)) %>%
  mutate(statefip = as_factor(statefip))

texas_train = subset(texas, post == 0)
texas_lasso <- dummy.data.frame(texas_train, names = c("statefip", "year_fixed"))
texas_lasso$statefip = texas_train$statefip
texas_lasso$year_fixed = texas_train$year_fixed

base_model = lm(bmprison ~ crack + alcohol + income + ur + poverty + black 
                + perc1519 + aidscapita + year_fixed + statefip, data=texas_train)


### LASSO

texas_x = sparse.model.matrix(bmprison ~ (crack + alcohol + income + ur + poverty + black 
                                          + perc1519 + aidscapita + year_fixed + statefip)^2, data=texas_lasso)[, -1]
texas_y = texas_lasso$bmprison
reg_lasso = gamlr(texas_x, texas_y)
plot(reg_lasso)

plot(reg_lasso$lambda, AICc(reg_lasso))
plot(log(reg_lasso$lambda), AICc(reg_lasso))

texas_lasso_beta = coef(reg_lasso) 
texas_lasso_variables = as.data.frame(texas_lasso_beta@Dimnames[[1]])
var_position = c(texas_lasso_beta@i + 1)
texas_result = texas_lasso_variables[var_position,1]
texas_result = texas_result[, drop=TRUE]
var_coef = as.data.frame(texas_result)
var_coef[] <- lapply(var_coef, as.character)
var_coef = cbind(var_coef, c(texas_lasso_beta@x))
colnames(var_coef) = c("Variable","Coefficient")
kable(var_coef) %>% kable_styling("striped")

log(reg_lasso$lambda[which.min(AICc(reg_lasso))])
sum(texas_lasso_beta!=0)

p1 <- dimnames(texas_lasso_beta)[[1]]
p1
p2 <- c()
p2
for (i in c(1:length(texas_lasso_beta))){
  p2 <- c(p2, as.list(texas_lasso_beta)[[i]])
}
model_lasso = c("bmprison ~ ")
for (i in c(2:length(texas_lasso_beta))){
  if (p2[i] != 0){
    if (model_lasso == "bmprison ~ "){
      model_lasso = paste(model_lasso, p1[i])
    }
    else{
      model_lasso = paste(model_lasso,"+", p1[i])
    }
  }
}
model_lasso <- as.formula(model_lasso)
model_lasso = lm(model_lasso, data=texas_lasso)


N=nrow(texas_train)
K=10
fold_id = rep_len(1:K, N)
fold_id = sample(fold_id, replace = FALSE)
err_save_lasso = rep(0, K)
for(i in 1:K){
  train_set = which(fold_id != i)
  y_test = texas_lasso$bmprison[-train_set]
  yhat_test = predict(model_lasso, newdata = texas_lasso[-train_set,])
  err_save_lasso[i] = mean((y_test-yhat_test)^2)
}

sqrt(mean(err_save_lasso))

texas_general <- dummy.data.frame(texas, names = c("statefip", "year_fixed"))
texas_general$statefip = texas$statefip
texas_general$year_fixed = texas$year_fixed

model_lasso_test = lm(model_lasso, data=texas_lasso)
texas_general$yhat = predict(model_lasso_test, newdata = texas_general)

random_state = sample(as.numeric(unique(texas_general$statefip)), size=5)

subset(texas_general, statefip == 48 | statefip == random_state[1] | 
         statefip == random_state[2] | statefip == random_state[3] |
         statefip == random_state[4] | statefip == random_state[5]) %>%
  ggplot(aes(x = year, colour=state)) +
  geom_line(aes(y = bmprison)) +
  geom_line(aes(y = yhat), linetype = "dashed") +
  theme_bw() +
  xlab("Year") +
  ylab("Black Male Incarceration") +
  geom_vline(xintercept = 1993, color = "dark grey", size = 0.8) +
  theme(panel.grid.minor = element_blank(), panel.grid.major.x = element_blank())

### RANDOM FOREST

err_save2 = rep(0, K)
for(i in 1:K){
  train_set2 = which(fold_id != i)
  y_test2 = texas_train$bmprison[-train_set2]
  model2_train = randomForest(bmprison ~ (crack + alcohol + income + ur + poverty + black 
                                          + perc1519 + aidscapita + year_fixed + statefip), data=texas_train[train_set2,], ntree=200)
  yhat_test2 = predict(model2_train, newdata = texas_train[-train_set2,])
  err_save2[i] = mean((y_test2-yhat_test2)^2)
}
plot(model2_train)
sqrt(mean(err_save2))


# BOOSTING

err_save3 = rep(0, K)
for(i in 1:K){
  train_set3 = which(fold_id != i)
  y_test3 = texas_train$bmprison[-train_set3]
  model3_train <- gbm(bmprison ~ (crack + alcohol + income + ur + poverty + black 
                                  + perc1519 + aidscapita + year_fixed + statefip), data=texas_train[train_set3,], interaction.depth=4, n.trees = 200, shrinkage =.05)
  yhat_test3 = predict(model3_train, newdata = texas_train[-train_set3,], n.trees =200)
  err_save3[i] = mean((y_test3-yhat_test3)^2)
}

sqrt(mean(err_save3))

model3_test <- gbm(bmprison ~ (crack + alcohol + income + ur + poverty + black 
                                + perc1519 + aidscapita + year_fixed + statefip), data=texas_train, interaction.depth=4, n.trees = 200, shrinkage =.05)

predict.gbm <- function (object, newdata, n.trees, type = "link", single.tree = FALSE, ...) {
  if (missing(n.trees)) {
    if (object$train.fraction < 1) {
      n.trees <- gbm.perf(object, method = "test", plot.it = FALSE)
    }
    else if (!is.null(object$cv.error)) {
      n.trees <- gbm.perf(object, method = "cv", plot.it = FALSE)
    }
    else {
      n.trees <- length(object$train.error)
    }
    cat(paste("Using", n.trees, "trees...\n"))
    gbm::predict.gbm(object, newdata, n.trees, type, single.tree, ...)
  }
}

texas_general$yhatRF = predict.gbm(model3_test, newdata = texas_general)

subset(texas_general, statefip == 48 | statefip == random_state[1] | 
         statefip == random_state[2] | statefip == random_state[3] |
         statefip == random_state[4] | statefip == random_state[5]) %>%
  ggplot(aes(x = year, colour=state)) +
  geom_line(aes(y = bmprison)) +
  geom_line(aes(y = yhatRF), linetype = "dashed") +
  theme_bw() +
  xlab("Year") +
  ylab("Black Male Incarceration") +
  geom_vline(xintercept = 1993, color = "dark grey", size = 0.8) +
  theme(panel.grid.minor = element_blank(), panel.grid.major.x = element_blank())

## RMSE comparison
Model <- c('Lasso','Randomforest','Boosting')
RMSE <- c(round(sqrt(mean(err_save_lasso)),2), round(sqrt(mean(err_save2)),2),round(sqrt(mean(err_save3)),2))
RMSE_result = data.frame(Model,RMSE)
RMSE_result = t(RMSE_result)
[Table 2] RMSE results of each model
pandoc.table(RMSE_result)
?pandoc.table
##

texas_pred = c(1:nrow(texas))
texas_pred = as.data.frame(texas_pred)

for(i in 1:1000) {
  train_resampling = resample(texas_train)
  boot_train <- gbm(bmprison ~ (crack + alcohol + income + ur + poverty + black 
                                + perc1519 + aidscapita + year_fixed + statefip), data=train_resampling, interaction.depth=4, n.trees = 300, shrinkage =.05)
  yhat = predict.gbm(boot_train, newdata = texas_general)
  texas_pred = cbind(texas_pred,yhat)
}

texas_pred$texas_pred = NULL
texas_pred$mean = rowMeans(texas_pred, na.rm = FALSE, dims = 1)
texas_pred$se = apply(texas_pred[,-1001], 1, sd)
texas_general$yhat_RF_mean = texas_pred$mean
texas_general$yhat_RF_se = texas_pred$se

subset(texas_general, statefip == 48) %>%
  ggplot(aes(x = year, ymin = yhat_RF_mean-1.96*yhat_RF_se, ymax = yhat_RF_mean+1.96*yhat_RF_se)) +
  geom_line(aes(y = bmprison)) +
  geom_line(aes(y = yhat_RF_mean), color = "dark blue", linetype = "dashed") +
  geom_line(aes(y = yhat_RF_mean+1.96*yhat_RF_se), color = 'light blue',linetype = "dashed") +
  geom_line(aes(y = yhat_RF_mean-1.96*yhat_RF_se), color = 'light blue',linetype = "dashed")+
  geom_ribbon(alpha = 0.4, fill = "light blue") +
  theme_bw() +
  xlab("Year") +
  ylab("Black Male Incarceration") +
  geom_vline(xintercept = 1993, color = "dark grey", size = 0.8) +
  theme(panel.grid.minor = element_blank(), panel.grid.major.x = element_blank())

### I WILL WORK OUT LATER
# 
# texas_train_new = dummy.data.frame(texas_train, names = c("statefip", "year_fixed"))
# texas_train_new$statefip = texas_train$statefip
# texas_train_new$year_fixed = texas_train$year_fixed
# 
# texas_train_new$bmprison[texas_train_new$statefip == random_state[1] & texas_train_new$year >= 1993] = ""
# texas_train_new$bmprison[texas_train_new$statefip == random_state[2] & texas_train_new$year >= 1993] = ""
# texas_train_new$bmprison[texas_train_new$statefip == random_state[3] & texas_train_new$year >= 1993] = ""
# texas_train_new$bmprison[texas_train_new$statefip == random_state[4] & texas_train_new$year >= 1993] = ""
# texas_train_new$bmprison[texas_train_new$statefip == random_state[5] & texas_train_new$year >= 1993] = ""
# 
# texas_pred_comp = c(1:nrow(texas))
# texas_pred_comp = as.data.frame(texas_pred_comp)
# 
# for(i in 1:1000) {
#   train_resampling = resample(texas_train_new)
#   boot_train <- gbm(bmprison ~ (crack + alcohol + income + ur + poverty + black 
#                                 + perc1519 + aidscapita + year_fixed + statefip), data=train_resampling, interaction.depth=4, n.trees = 300, shrinkage =.05)
#   yhat = predict.gbm(boot_train, newdata = texas_general)
#   texas_pred_comp = cbind(texas_pred_comp,yhat)
# }
# 
# texas_pred_comp$texas_pred_comp = NULL
# texas_pred_comp$mean = rowMeans(texas_pred, na.rm = FALSE, dims = 1)
# texas_pred_comp$se = apply(texas_pred[,-1001], 1, sd)
# texas_general$yhat_RF_mean_comp = texas_pred_comp$mean
# texas_general$yhat_RF_se_comp = texas_pred_comp$se
# 
# subset(texas_general, statefip == random_state[1]) %>%
#   ggplot(aes(x = year, ymin = yhat_RF_mean-1.96*yhat_RF_se, ymax = yhat_RF_mean+1.96*yhat_RF_se)) +
#   geom_line(aes(y = bmprison)) +
#   geom_line(aes(y = yhat_RF_mean), color = "dark blue", linetype = "dashed") +
#   geom_line(aes(y = yhat_RF_mean+1.96*yhat_RF_se), color = 'light blue',linetype = "dashed") +
#   geom_line(aes(y = yhat_RF_mean-1.96*yhat_RF_se), color = 'light blue',linetype = "dashed")+
#   geom_ribbon(alpha = 0.4, fill = "light blue") +
#   theme_bw() +
#   xlab("Year") +
#   ylab("Black Male Incarceration") +
#   geom_vline(xintercept = 1993, color = "dark grey", size = 0.8) +
#   theme(panel.grid.minor = element_blank(), panel.grid.major.x = element_blank())
