// betaD-Bayesian t-Logistic Regression


functions {
   real exp_t(real x, real t){
      real out;
      if(t==1){ 
         out = exp(x);
      }
      else{
         out = fmax(0,(1+(1-t)*x)^(1/(1-t)));
      }
      return out;
   }
   
   real log_t(real x, real t){
      
      real out;
      if(t==1){ 
         out = log(x);
      }
      else{
         out = (x^(1-t)-1)/(1-t);
      }
      return out;
   }
   
   real G_t(real a_hat, real t){
      real tol = 10e-10;
      real a_hat_abs = a_hat;
      real a_tilde;
      real conv = 1e5;
      real Z_a_tilde;
      real a_tilde_new;
      if(a_hat < 0){ 
        a_hat_abs  = - a_hat;
      }
      a_tilde = a_hat_abs;
      while(conv > tol){
         Z_a_tilde = 1+exp_t(-a_tilde,t);
         a_tilde_new = Z_a_tilde^(1-t)*a_hat_abs;
         conv = fabs(a_tilde_new-a_tilde);
         a_tilde = a_tilde_new;
      }
      return -log_t(1/Z_a_tilde,t)+a_hat_abs/2.0;
   }

}

data {
   
   int<lower=0> n;
   int<lower=0> p;
   int<lower=-1,upper=1> y[n,1];
   matrix[n,p] X;
   real mu_beta;
   real<lower=0> beta_s;
   real<lower=0> w;
   real<lower=0> t;
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
   real p_logistic;
   beta ~ normal(mu_beta,sqrt(beta_s));

   for(i in 1:n){
     p_logistic = exp_t((0.5*y[i,1]*lin_pred[i,1]-G_t(lin_pred[i,1],t)),t);
     
      target += 1/beta_p*p_logistic^beta_p - 
              1/(beta_p+1)*(p_logistic^(beta_p+1)+(1-p_logistic)^(beta_p+1));
   }
}

