// KLD-Bayesian Gaussian Likelihood Model (multiplying the scale parameter sigma2 by a constant sigma2_adj)

data {
   // Inputs for the sampler: data and prior hyperparameters
   int<lower=0> n;
   matrix[n,1] y;
   real mu_m;
   real<lower=0> mu_s;
   real<lower=0> sig_p1;
   real<lower=0> sig_p2;
   real<lower=0> w;
   real<lower=0> sigma2_adj;
}

parameters 
{
   // Parameters for which we do inference
   real mu;
   real<lower=0> sigma2;

}

model {
   // The prior
   //sigma2 ~ inv_gamma(sig_p1,sig_p2);
   //mu ~ normal(mu_m,sqrt(sigma2*mu_s));
   target += inv_gamma_lpdf(sigma2 | sig_p1, sig_p2);
   target += normal_lpdf(mu | mu_m, sqrt(sigma2*mu_s));
   
   // The likelihood 
   //y[,1] ~ normal(mu,sqrt(sigma2*sigma2_adj));
   target += normal_lpdf(y[,1] | mu, sqrt(sigma2*sigma2_adj));
}

generated quantities {
   // Sampling from the posterior predictive
   real y_predict;
   y_predict = normal_rng(mu, sqrt(sigma2*sigma2_adj));

}
