options(stringsAsFactors = FALSE)

args <- commandArgs(trailingOnly = TRUE)
input <- if (length(args) >= 1) args[[1]] else "./results/tables/analysis_dataset.csv"
out <- if (length(args) >= 2) args[[2]] else "./results/tables"
sims <- if (length(args) >= 3) as.integer(args[[3]]) else 10000L
seed <- if (length(args) >= 4) as.integer(args[[4]]) else 42L

x <- "node_contrib__mode1__Posterior_PHG"
m <- "negative_Lmax_mean_10_15_20"
y <- "AutomaticThought"
covariates <- c("n_sentences", "negative_sentence_prop", "age", "sex_M")
columns <- c("subject", x, m, y, covariates)

data <- read.csv(input, check.names = FALSE, fileEncoding = "UTF-8-BOM")
data <- data[complete.cases(data[, columns]), columns]
continuous <- c(x, m, y, setdiff(covariates, "sex_M"))
for (name in continuous) data[[name]] <- as.numeric(scale(data[[name]]))
data$sex_M <- as.numeric(data$sex_M)

fit_lm <- function(formula, data) {
  model <- lm(formula, data)
  model$call$formula <- formula
  model
}

mediator_model <- fit_lm(reformulate(c(x, covariates), response = m), data)
outcome_model <- fit_lm(reformulate(c(m, x, covariates), response = y), data)
total_model <- fit_lm(reformulate(c(x, covariates), response = y), data)

set.seed(seed)
med <- mediation::mediate(
  model.m = mediator_model,
  model.y = outcome_model,
  treat = x,
  mediator = m,
  boot = TRUE,
  sims = sims,
  long = TRUE
)

coef_p <- function(model, term) summary(model)$coefficients[term, "Pr(>|t|)"]
summary_row <- data.frame(
  n = nrow(data),
  path_a = unname(coef(mediator_model)[x]),
  path_a_p = coef_p(mediator_model, x),
  path_b = unname(coef(outcome_model)[m]),
  path_b_p = coef_p(outcome_model, m),
  direct_path_c_prime = unname(coef(outcome_model)[x]),
  direct_path_c_prime_p = coef_p(outcome_model, x),
  total_path_c_lm = unname(coef(total_model)[x]),
  total_path_c_lm_p = coef_p(total_model, x),
  acme = med$d.avg,
  acme_ci95_low = med$d.avg.ci[[1]],
  acme_ci95_high = med$d.avg.ci[[2]],
  acme_p = med$d.avg.p,
  ade = med$z.avg,
  ade_ci95_low = med$z.avg.ci[[1]],
  ade_ci95_high = med$z.avg.ci[[2]],
  bootstrap_samples = sims,
  seed = seed
)

dir.create(out, recursive = TRUE, showWarnings = FALSE)
write.csv(summary_row, file.path(out, "indirect_association.csv"), row.names = FALSE)
write.csv(data, file.path(out, "mediation_standardized_data.csv"), row.names = FALSE)
capture.output(summary(mediator_model), summary(outcome_model), summary(total_model),
               summary(med), sessionInfo(), file = file.path(out, "indirect_association_output.txt"))
print(summary_row[, c("n", "path_a", "path_b", "direct_path_c_prime", "acme", "acme_ci95_low", "acme_ci95_high", "acme_p")])
