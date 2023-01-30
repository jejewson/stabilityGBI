// KL-Bayesian Linear Model with Student-t errors

// Inputs for the sampler: data and prior hyperparameters
data {
   
   int<lower=0> n;// number of obs
   int<lower=0> p;// number of predictors
   matrix[n,1] y;// response variable
   matrix[n,p] X;// predictors
   real mu_beta;// prior mean for regression params
   real<lower=0> beta_s;// prior sd for regression params
   real<lower=0> sig_p1;// shape param of prior for residuals scale 
   real<lower=0> sig_p2;// scale param of prior for residuals scale 
   real<lower =0> df;// Student-t degrees of freedom
   real<lower=0> w;// General bayes calibration weight
}

// Parameters for which we do inference 
parameters 
{
   
   vector[p] beta;// regression parameters
   real<lower = 0> sigma2;// residuals scale
}

transformed parameters
{
   matrix[n,1] lin_pred;
   lin_pred[,1] = X*beta;

}

// Computations for the log-postrior 
model {
   // the prior
   target += inv_gamma_lpdf(sigma2 | sig_p1, sig_p2);
   target += normal_lpdf(beta | mu_beta, sqrt(sigma2*beta_s));

   // The likelihood
   target += w * student_t_lpdf(y[,1] | df, lin_pred[,1], sqrt(sigma2));
   
}

// Sampling from the posterior predictive
generated quantities {
   
   matrix[n,1] y_predict;
   for(i in 1:n){
     y_predict[i, 1] = student_t_rng(df, lin_pred[i,1], sqrt(sigma2));
   }
   

}

