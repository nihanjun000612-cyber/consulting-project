# GVHD Pre-transplant Binary Classification Pipeline
# Target construction from agvhd24/agvhd34 and statistically rigorous modeling

suppressPackageStartupMessages({
  library(dplyr)
  library(glmnet)
  library(car)
  library(pROC)
  library(ResourceSelection)
  library(caret)
  library(rms)
  library(broom)
})

set.seed(2026)

# -----------------------------
# 0) Input data
# -----------------------------
# Expected objects:
#   - df_clean: cleaned dataset from EDA (recommended)
#   - or df_raw: original dataset (then user should clean first)
# Required columns:
#   - agvhd24, agvhd34 (indicators)
#   - pre-transplant predictors

if (!exists("df_clean")) stop("Please provide df_clean in workspace before running.")

# -----------------------------
# 1) Outcome engineering
# -----------------------------
# 3-level severity from teacher's rule:
# severe     : agvhd34 == 1
# moderate   : agvhd24 == 1 & agvhd34 == 0
# mild       : otherwise

analysis_df <- df_clean %>%
  mutate(
    agvhd24 = as.integer(agvhd24),
    agvhd34 = as.integer(agvhd34),
    y3 = case_when(
      agvhd34 == 1 ~ 3L,
      agvhd24 == 1 & agvhd34 == 0 ~ 2L,
      TRUE ~ 1L
    ),
    # Final binary target (recommended): severe vs non-severe
    y_bin = if_else(y3 == 3L, 1L, 0L)
  )

# -----------------------------
# 2) Predictor set (pre-transplant only)
# -----------------------------
# IMPORTANT: replace with your exact pre-transplant variables from EDA.
pre_tx_vars <- c(
  "age", "sex", "disease_risk", "conditioning_intensity", "donor_type",
  "hla_match", "stem_cell_source", "cmv_status", "kps", "comorbidity_index"
)

missing_vars <- setdiff(pre_tx_vars, names(analysis_df))
if (length(missing_vars) > 0) {
  stop(paste("Missing predictors:", paste(missing_vars, collapse = ", ")))
}

model_df <- analysis_df %>%
  select(y_bin, all_of(pre_tx_vars)) %>%
  mutate(across(where(is.character), as.factor)) %>%
  mutate(across(where(is.logical), as.integer)) %>%
  tidyr::drop_na()

# -----------------------------
# 3) Train/test split (stratified)
# -----------------------------
idx <- createDataPartition(model_df$y_bin, p = 0.8, list = FALSE)
train_df <- model_df[idx, ]
test_df  <- model_df[-idx, ]

# -----------------------------
# 4) LASSO logistic for variable screening
# -----------------------------
x_train <- model.matrix(y_bin ~ ., train_df)[, -1]
y_train <- train_df$y_bin

cvfit <- cv.glmnet(
  x = x_train,
  y = y_train,
  family = "binomial",
  alpha = 1,
  nfolds = 10,
  type.measure = "deviance"
)

coef_1se <- coef(cvfit, s = "lambda.1se")
selected <- rownames(coef_1se)[as.vector(coef_1se != 0)]
selected <- setdiff(selected, "(Intercept)")

if (length(selected) == 0) stop("No predictors selected by LASSO under lambda.1se")

# -----------------------------
# 5) Refit standard logistic model + inference
# -----------------------------
selected_formula <- as.formula(
  paste("y_bin ~", paste(selected, collapse = " + "))
)

fit <- glm(selected_formula, data = train_df, family = binomial())

# Multicollinearity check (VIF > 5 suggests concern)
vif_table <- car::vif(fit)

# Odds ratios with 95% CI and p-values
or_table <- broom::tidy(fit, conf.int = TRUE, exponentiate = TRUE)

# -----------------------------
# 6) Discrimination + calibration
# -----------------------------
# Test probabilities
p_test <- predict(fit, newdata = test_df, type = "response")

# ROC/AUC
roc_obj <- pROC::roc(test_df$y_bin, p_test, quiet = TRUE)
auc_val <- as.numeric(pROC::auc(roc_obj))
auc_ci <- pROC::ci.auc(roc_obj)

# Hosmer-Lemeshow (on train set; grouping=10)
hl <- ResourceSelection::hoslem.test(train_df$y_bin, fitted(fit), g = 10)

# Brier score (test)
brier <- mean((p_test - test_df$y_bin)^2)

# Bootstrap optimism-corrected C-index (optional but recommended)
dd <- datadist(train_df)
options(datadist = "dd")
fit_lrm <- rms::lrm(selected_formula, data = train_df, x = TRUE, y = TRUE)
val <- rms::validate(fit_lrm, B = 200)

# -----------------------------
# 7) Output key results
# -----------------------------
cat("\n=== Selected predictors (LASSO 1SE) ===\n")
print(selected)

cat("\n=== Logistic regression OR table ===\n")
print(or_table)

cat("\n=== VIF ===\n")
print(vif_table)

cat("\n=== Test AUC and 95% CI ===\n")
cat(sprintf("AUC = %.3f, 95%%CI [%.3f, %.3f]\n", auc_val, auc_ci[1], auc_ci[3]))

cat("\n=== Hosmer-Lemeshow test (train) ===\n")
print(hl)

cat("\n=== Test Brier score ===\n")
print(brier)

cat("\n=== Bootstrap validation (rms::validate) ===\n")
print(val)

# Decision threshold example: Youden index
best <- pROC::coords(roc_obj, x = "best", best.method = "youden", transpose = FALSE)
cat("\n=== Suggested threshold (Youden) ===\n")
print(best)
