// gammaD-Bayesian Student's-t Likelihood Model

data {
   // Inputs for the sampler: data and prior hyperparameters
   int<lower=0> n;
   matrix[n,1] y;
   real mu_m;
   real<lower=0> mu_s;
   real<lower=0> sig_p1;
   real<lower=0> sig_p2;
   real<lower =0> df;
   real<lower=0> w;
   real gamma;
}

parameters 
{
   // Parameters for which we do inference
   real mu;
   real<lower=0> sigma2;

}

transformed parameters
{
  // Calculates the integral term int f(z;theta)^(beta+1) dz
  real int_term;
  
  int_term = (tgamma((df + 1.0)/2.0)^(gamma + 1.0)*tgamma((gamma*df + gamma + df)/2.0))/
             (tgamma(df/2)^(gamma + 1)*tgamma((gamma*df + gamma + df + 1.0)/2.0)*(df)^((gamma)/2.0)*pi()^((gamma)/2.0)*sigma2^(gamma/2.0));
  
  
}

model {
  
   // The prior
   target += inv_gamma_lpdf(sigma2 | sig_p1, sig_p2);
   target += normal_lpdf(mu | mu_m, sqrt(sigma2*mu_s));
  
   // The general Bayesian loss function
   for(i in 1:n){
    target += w*((1.0/gamma)*exp(student_t_lpdf(y[i,1] | df, mu, sqrt(sigma2)))^(gamma) * 1/(gamma + 1) * 1/(int_term^(gamma/(gamma + 1))));
  }
}

generated quantities {
  // Sampling from the posterior predictive
  real y_predict;
  y_predict = student_t_rng(df, mu, sqrt(sigma2));

}
