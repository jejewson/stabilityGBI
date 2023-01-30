// KL-Bayesian Student-t likelihood model

// Inputs for the sampler: data and prior hyperparameters
data {
   
   int<lower=0> n;// number of obs
   matrix[n,1] y;// the observations
   real mu_m;// location param of the prior for the location
   real<lower=0> mu_s;// scale param of the prior fo the location
   real<lower=0> sig_p1;// shape param of prior for the scale 
   real<lower=0> sig_p2;// scale param of prior for the scale 
   real<lower =0> df;// Student-t degrees of freedom
   real<lower=0> w;// General bayes calibration weight
}

// Parameters for which we do inference  
parameters 
{
    
   real mu;// the location parameter 
   real<lower=0> sigma2;// the scale parameter 

}

// Computations for the log-postrior 
model {
   // The prior
   target += inv_gamma_lpdf(sigma2 | sig_p1, sig_p2);
   target += normal_lpdf(mu | mu_m, sqrt(sigma2*mu_s));
  
   // The likelihood 
   target += student_t_lpdf(y[,1] | df, mu, sqrt(sigma2));
}

// Sampling from the posterior predictive
generated quantities {
  
  real y_predict;
  y_predict = student_t_rng(df, mu, sqrt(sigma2));

}
