library(gamlr)
library(kableExtra)
library(randomForest)
library(gbm)
library(pdp)

texas$trend = texas$year - 1985

texas_train = subset(texas, post == 0)
texas_lasso <- dummy.data.frame(texas_train, names = c("statefip", "year_fixed"))
texas_lasso$statefip = texas_train$statefip
texas_lasso$year_fixed = texas_train$year_fixed

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
K=100
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
  err_save2[i] = mean((y_test2-yhat_test2)^2)
}

sqrt(mean(err_save2))

texas_general <- dummy.data.frame(texas, names = c("statefip", "year_fixed"))
texas_general$statefip = texas$statefip
texas_general$year_fixed = texas$year_fixed

model2_test = lm(model2, data=texas_lasso)
texas_general$yhat = predict(model2_test, newdata = texas_general)

subset(texas_general, statefip == 48) %>%
  ggplot(aes(x = year)) +
  geom_line(aes(y = bmprison), color = "dark blue") +
  geom_line(aes(y = yhat), color = "dark blue", linetype = "dashed") 
  theme_bw() +
  xlab("Years Relative to Prison Expansion") +
  ylab("Black Male Prison") +
  geom_hline(yintercept = 1993, color = "dark grey", size = 0.8) +
  geom_vline(xintercept = 0, color = "dark grey", size = 0.8) +
  theme(panel.grid.minor = element_blank(), panel.grid.major.x = element_blank())

random_state = sample(as.numeric(unique(texas_general$statefip)), size=5)
  
subset(texas_general, statefip == 48 | statefip == random_state[1] | 
         statefip == random_state[2] | statefip == random_state[3] |
         statefip == random_state[4] | statefip == random_state[5]) %>%
    ggplot(aes(x = year, colour=statefip)) +
    geom_line(aes(y = bmprison)) +
    geom_line(aes(y = yhat), linetype = "dashed")


