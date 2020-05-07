library(gamlr)
library(kableExtra)
library(randomForest)
library(gbm)
library(pdp)
library(mosaic)

texas$trend = texas$year - 1985

texas_train = subset(texas, post == 0)
texas_lasso <- dummy.data.frame(texas_train, names = c("statefip", "year_fixed"))
texas_lasso$statefip = texas_train$statefip
texas_lasso$year_fixed = texas_train$year_fixed

### STEPWISE

base_model = lm(bmprison ~ crack + alcohol + income + ur + poverty + black 
                + perc1519 + aidscapita + year_fixed + statefip,
                data=texas_train)
full = lm(bmprison ~ (crack + alcohol + income + ur + poverty + black 
                + perc1519 + aidscapita + year_fixed + statefip)^2,
                data=texas_train)
null = lm(bmprison~1, data=texas_train)

step_select = step(base_model, scope= list(lower=null, upper=full), dir='both')

step_select[["call"]]

model1 = lm(formula = bmprison ~ crack + alcohol + income + ur + poverty + 
              black + perc1519 + year_fixed + statefip + year_fixed:statefip, 
            data = texas_train)

N=nrow(texas_train)
K=10
fold_id = rep_len(1:K, N)
fold_id = sample(fold_id, replace = FALSE)
err_save = rep(0, K)
err_save2 = err_save
for(i in 1:K){
  train_set = which(fold_id != i)
  y_test = texas_train$bmprison[-train_set]
  model1_train = lm(model1, data=texas_train[train_set,])
  yhat_test = predict(model1_train, newdata = texas_train[-train_set,])
  base_model_train = lm(base_model, data=texas_train[train_set,])
  yhat_test2 = predict(base_model_train, newdata = texas_train[-train_set,])
  err_save[i] = mean((y_test-yhat_test)^2)
  err_save2[i] = mean((y_test-yhat_test2)^2)
}

sqrt(mean(err_save))
sqrt(mean(err_save2))

### LASSO

texas_x = sparse.model.matrix(bmprison ~ (crack + alcohol + income + ur + poverty + black 
                                          + perc1519 + aidscapita + year_fixed + statefip)^2, data=texas_lasso)[, -1]
texas_y = texas_lasso$bmprison
reg_lasso = gamlr(texas_x, texas_y)
plot(reg_lasso)

min(AICc(reg_lasso))
which.min(AICc(reg_lasso))

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
log(texas_lasso$lambda[which.min(AICc(reg_lasso))])
sum(texas_lasso_beta!=0)

p1 <- dimnames(texas_lasso_beta)[[1]]
p2 <- c()
for (i in c(1:length(texas_lasso_beta))){
  p2 <- c(p2, as.list(texas_lasso_beta)[[i]])
}
model2 = c("bmprison ~ ")
for (i in c(2:length(texas_lasso_beta))){
  if (p2[i] != 0){
    if (model2 == "bmprison ~ "){
      model2 = paste(model2, p1[i])
    }
    else{
      model2 = paste(model2,"+", p1[i])
    }
  }
}
model2 <- as.formula(model2)
model2 = lm(model2, data=texas_lasso)
err_save_lasso = rep(0, K)
for(i in 1:K){
  train_set2 = which(fold_id != i)
  y_test2 = texas_lasso$bmprison[-train_set2]
  model2_train = lm(model2, data=texas_lasso[train_set2,])
  yhat_test2 = predict(model2_train, newdata = texas_lasso[-train_set2,])
  err_save_lasso[i] = mean((y_test2-yhat_test2)^2)
}

sqrt(mean(err_save_lasso))

texas_general <- dummy.data.frame(texas, names = c("statefip", "year_fixed"))
texas_general$statefip = texas$statefip
texas_general$year_fixed = texas$year_fixed

model2_test = lm(model2, data=texas_lasso)
texas_general$yhat = predict(model2_test, newdata = texas_general)

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

err_save4 = rep(0, K)
system.time(for(i in 1:K){
  train_set4 = which(fold_id != i)
  y_test4 = texas_train$bmprison[-train_set4]
  model4_train = randomForest(bmprison ~ (crack + alcohol + income + ur + poverty + black 
                                          + perc1519 + aidscapita + year_fixed + statefip), data=texas_train[train_set4,], ntree=300)
  yhat_test4 = predict(model4_train, newdata = texas_train[-train_set4,])
  err_save4[i] = mean((y_test4-yhat_test4)^2)
})

sqrt(mean(err_save4))

# BOOSTING

err_save5 = rep(0, K)
system.time(for(i in 1:K){
  train_set5 = which(fold_id != i)
  y_test5 = texas_train$bmprison[-train_set5]
  model5_train <- gbm(bmprison ~ (crack + alcohol + income + ur + poverty + black 
                                  + perc1519 + aidscapita + year_fixed + statefip), data=texas_train[train_set5,], interaction.depth=4, n.trees = 300, shrinkage =.05)
  yhat_test5 = predict(model5_train, newdata = texas_train[-train_set5,], n.trees =300)
  err_save5[i] = mean((y_test5-yhat_test5)^2)
})

sqrt(mean(err_save5))

model5_test <- gbm(bmprison ~ (crack + alcohol + income + ur + poverty + black 
                                + perc1519 + aidscapita + year_fixed + statefip), data=texas_train, interaction.depth=4, n.trees = 300, shrinkage =.05)

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

texas_general$yhatRF = predict.gbm(model5_test, newdata = texas_general)

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
