
data {
   
   int<lower=0> n;
   int<lower=0> p;
   int<lower=-1,upper=1> y[n,1];
   matrix[n,p] X;
   real mu_beta;
   real<lower=0> beta_s;
   real<lower=0> w;
   real beta_p;

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
   real p_probit;
   beta ~ normal(mu_beta,sqrt(beta_s));

   for(i in 1:n){
      if(y[i,1] == 1){
         p_probit = normal_cdf(lin_pred[i,1], 0, 1);
      }
      if(y[i,1] == -1){
         p_probit = (1 - normal_cdf(lin_pred[i,1], 0, 1));
      }
     
      target += 1/beta_p*p_probit^beta_p - 
              1/(beta_p+1)*(p_probit^(beta_p+1)+(1-p_probit)^(beta_p+1));
   }
}

