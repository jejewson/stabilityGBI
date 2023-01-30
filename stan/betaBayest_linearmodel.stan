// betaD-Bayesian Student's-t Linear Model

data {
   
   int<lower=0> n;
   int<lower=0> p;
   matrix[n,1] y;
   matrix[n,p] X;
   real mu_beta;
   real<lower=0> beta_s;
   real<lower=0> sig_p1;
   real<lower=0> sig_p2;
   real<lower =0> df;
   real<lower=0> w;
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
  
   int_term = (tgamma((df + 1.0)/2.0)^(beta_p + 1.0)*tgamma((beta_p*df + beta_p + df)/2.0))/
             ((1.0 + beta_p)*tgamma(df/2)^(beta_p + 1)*tgamma((beta_p*df + beta_p + df + 1.0)/2.0)*(df)^((beta_p)/2.0)*pi()^((beta_p)/2.0)*sigma2^(beta_p/2.0));
  
  
}

model {
  
   target += inv_gamma_lpdf(sigma2 | sig_p1, sig_p2);
   target += normal_lpdf(beta | mu_beta,  sqrt((sigma2*beta_s)));


   for(i in 1:n){
      target += (w*((1/beta_p) * exp(student_t_lpdf(y[i,1] | df, lin_pred[i, 1], sqrt(sigma2)))^(beta_p) - int_term));
   }
}

generated quantities {
   // Sampling from the posterior predictive
   matrix[n,1] y_predict;
   for(i in 1:n){
     y_predict[i, 1] = student_t_rng(df, lin_pred[i,1], sqrt(sigma2));
   }
   
}
