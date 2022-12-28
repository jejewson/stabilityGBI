
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
}

parameters 
{
   
   vector[p] beta;
   real<lower = 0> sigma2;
}

transformed parameters
{
   matrix[n,1] lin_pred;
   lin_pred[,1] = X*beta;

}

model {

   target += inv_gamma_lpdf(sigma2 | sig_p1, sig_p2);
   target += normal_lpdf(beta | mu_beta, sqrt(sigma2*beta_s));
   /*
   for(i in 1:n){
      y[i] ~ normal(mu, sqrt(sigma2));
   }
   */
   target += w * normal_lpdf(y[,1] | lin_pred[,1], sqrt(sigma2*sigma2_adj));
   
}

generated quantities {
   // Sampling from the posterior predictive
   matrix[n,1] y_predict;
   for(i in 1:n){
     y_predict[i, 1] = normal_rng(lin_pred[i,1], sqrt(sigma2*sigma2_adj));
   }
   

}

