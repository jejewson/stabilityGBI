// betaD-Bayesian Linear Model (multiplying the scale parameter sigma2 by a constant sigma2_adj)

data {
   
   int<lower=0> n;
   int<lower=0> p;
   matrix[n,1] y;
   matrix[n,p] X;
   real mu_beta;
   real<lower=0> beta_s;
   real<lower=0> sig_p1;
   real<lower=0> sig_p2;
   real<lower=0> w;
   real<lower=0> sigma2_adj;
   real beta_p;
}

parameters 
{
   
   vector[p] beta;
   real<lower=0> sigma2;
}

transformed parameters
{
  
   real int_term;
   matrix[n,1] lin_pred;
   lin_pred[,1] = X*beta;
  
   int_term = (1 / ((2.0*pi())^(beta_p / 2.0) * (1 + beta_p)^1.5*((sigma2*sigma2_adj)^(beta_p / 2))));
  
  
}

model {
  
   target += inv_gamma_lpdf(sigma2 | sig_p1, sig_p2);
   target += normal_lpdf(beta | mu_beta,  sqrt((sigma2*beta_s)));


   for(i in 1:n){
      target += (w*((1/beta_p) * exp(normal_lpdf(y[i,1] | lin_pred[i, 1], sqrt(sigma2_adj*sigma2)))^(beta_p) - int_term));
   }
}

generated quantities {
   // Sampling from the posterior predictive
   matrix[n,1] y_predict;
   for(i in 1:n){
     y_predict[i, 1] = normal_rng(lin_pred[i,1], sqrt(sigma2*sigma2_adj));
   }
   
}
