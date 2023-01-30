// KLD-Bayesian Mislabelled Logistic Regression (multiplying the beta parameters by a constant w_beta)


data {
   
   int<lower=0> n;
   int<lower=0> p;
   int<lower=-1,upper=1> y[n,1];
   matrix[n,p] X;
   real mu_beta;
   real<lower=0> beta_s;
   real<lower = 0> eta_bound[2];
   real<lower=0> w;
   real<lower=0> w_beta;

}

parameters 
{
   
   vector[p] beta;
   vector<lower=eta_bound[1], upper = eta_bound[2]>[2] eta;

}

transformed parameters
{
  
  matrix[n,1] lin_pred;
  lin_pred[,1] = w_beta*X*beta;
  
  
}

model {
  
   target += normal_lpdf(beta | mu_beta,sqrt(beta_s));
   target += beta_lpdf(eta | 1, 1);

  for(i in 1:n){
     
      target += log((1-eta[(y[i,1]+1)/2 + 1])*exp(0.5*y[i,1]*lin_pred[i,1])/(exp(0.5*lin_pred[i,1])+exp(-0.5*lin_pred[i,1])) + 
                       eta[(-y[i,1]+1)/2 + 1]*exp(-0.5*y[i,1]*lin_pred[i,1])/(exp(0.5*lin_pred[i,1])+exp(-0.5*lin_pred[i,1])));
    
  }
}

