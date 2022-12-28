
data {
   
   int<lower=0> n;
   int<lower=0> p;
   int<lower=-1,upper=1> y[n,1];
   matrix[n,p] X;
   real mu_beta;
   real<lower=0> beta_s;
   real<lower=0> w;
   real<lower=0> w_beta;

}

parameters 
{
   
vector[p] beta;

}

transformed parameters
{
  
  matrix[n,1] lin_pred;
  lin_pred[,1] = w_beta*X*beta;
  
  
}

model {
  
beta ~ normal(mu_beta,sqrt(beta_s));


  for(i in 1:n){
    target += 0.5*y[i,1]*lin_pred[i,1]-log(exp(0.5*lin_pred[i,1])+exp(-0.5*lin_pred[i,1]));
  }
}

