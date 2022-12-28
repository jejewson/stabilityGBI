
data {
   // Inputs for the sampler: data and prior hyperparameters
   int<lower=0> n;
   matrix[n,1] y;
   real mu_m;
   real<lower=0> mu_s;
   real<lower=0> sig_p1;
   real<lower=0> sig_p2;
   real<lower=0> w;
   real gamma;
   real<lower=0> sigma2_adj;
}

parameters 
{
   // Parameters for which we do inference
   real mu;
   real<lower=0> sigma2;

}

transformed parameters
{
  // Calculates the integral term int f(z;theta)^(gamma+1) dz
  real int_term;
  
  int_term = (1/((2.0*pi())^(gamma/2.0)*(1 + gamma)^0.5*((sigma2*sigma2_adj)^(gamma/2))));
  
  
}

model {
   // The prior
   //sigma2 ~ inv_gamma(sig_p1,sig_p2);
   //mu ~ normal(mu_m,sqrt(sigma2*mu_s));
   target += inv_gamma_lpdf(sigma2 | sig_p1, sig_p2);
   target += normal_lpdf(mu | mu_m, sqrt(sigma2*mu_s));
  
   // The general Bayesian loss function
   for(i in 1:n){
     target += w*((1/gamma)*exp(normal_lpdf(y[i,1] | mu, sqrt(sigma2_adj*sigma2)))^(gamma) * 1/(gamma + 1) * 1/(int_term^(gamma/(gamma + 1))));
   }
}

generated quantities {
   // Sampling from the posterior predictive
   real y_predict;
   y_predict = normal_rng(mu, sqrt(sigma2*sigma2_adj));

}
