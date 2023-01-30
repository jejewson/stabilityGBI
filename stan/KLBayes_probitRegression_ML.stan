// KLD-Bayesian Probit Regression 

data {
   
   int<lower=0> n;
   int<lower=0> p;
   int<lower=-1,upper=1> y[n,1];
   matrix[n,p] X;
   real mu_beta;
   real<lower=0> beta_s;
   real<lower=0> w;

}

parameters 
{
   
  vector[p] beta;

}

transformed parameters
{
  
  matrix[n,1] lin_pred;
  lin_pred[,1] = X*beta;
  
  
}

model {
   real log_p_probit;
   
   target += normal_lpdf(beta | mu_beta,sqrt(beta_s));


   for(i in 1:n){
      if(y[i,1] == 1){
         log_p_probit = normal_lcdf(lin_pred[i,1]| 0, 1);
      }
      if(y[i,1] == -1){
         log_p_probit = log(1 - normal_cdf(lin_pred[i,1], 0, 1));
      }
     
      target += log_p_probit;
   }
}

