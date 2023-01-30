// betaD-Bayesian Finite Mixture of Gaussians Model - using NormalInverseWishart priors 
// This is univariate so IW(Sigma_j; nu0, S0) is inverse-gamma distribution. 
// With alpha0 = nu0/2 and \beta0 = S0/2 

functions {
   
  // Creates one simplex
  vector simplex_create(vector theta_raw, int m){
    vector[m+1] theta_simplex;
    real stick_len = 1;
    real prop;
    for(j in 1:m){
      prop = inv_logit(theta_raw[j] - log(m - j + 1));
      theta_simplex[j] = stick_len * prop;
      stick_len = stick_len - theta_simplex[j];
    }
    theta_simplex[m + 1] = stick_len;
    
    return theta_simplex;
  }
  
  // log absolute Jacobian determinant of the simplex_create
  real simplex_create_lj(vector theta_raw, int m){
    real lj = 0;
    real stick_len = 1;
    real adj_theta_raw_j;
    for (j in 1:m) {
      adj_theta_raw_j = theta_raw[j] - log(m - j + 1);
      lj = lj + log(stick_len) - log1p_exp(-adj_theta_raw_j) - log1p_exp(adj_theta_raw_j);
      stick_len = stick_len * (1.0 - inv_logit(adj_theta_raw_j));
    }
    
    return lj;
  }
   
   // Finite Gaussian Mixture Model Likelihood 
   real norm_mix_lpdf (real y, int K, vector mu, vector sigma2, vector omega){
      real log_lik = negative_infinity();
      for(k in 1:K){
         log_lik = log_sum_exp(log_lik, log(omega[k]) + normal_lpdf(y | mu[k], sqrt(sigma2[k])));
      }
     return log_lik;
   }
   
   // exp(-ell(x, theta)) for the betaD loss applied to the Finite Gaussian Mixture Model Likelihood
   real norm_mix_pdf_beta(real x, real xc, real[] theta,
                      real[] x_r, int[] x_i){
      int K = x_i[1];
      real beta_p = x_r[1];
      real mu[K] = segment(theta, 1, K);
      real sigma2[K] = segment(theta, (K + 1), K);
      real omega[K] = segment(theta, (2*K + 1), K);
      
      real log_lik = negative_infinity();
      for(k in 1:K){
         log_lik = log_sum_exp(log_lik, log(omega[k]) + normal_lpdf(x | mu[k], sqrt(sigma2[k])));
      }
     return (exp(log_lik)^(beta_p + 1));
   }
   




}


data {
   
   int<lower=0> n;
   int<lower=1> K;
   matrix[n,1] y;
   vector[K] mu_0;
   real<lower=0> kappa;
   real<lower=0> nu_0;
   real<lower=0> S_0;
   real<lower=0> alpha_0;
   real<lower=0> w;
   real beta_p;

}

parameters 
{
   
   ordered[K] mu;
   vector<lower=0>[K] sigma2;
   vector[K - 1] omega_raw; // Each K simplex only has K - 1 degrees of freedom

}

model {
   
   vector[K] omega_simplex;
   real int_term;
   real theta[3*K];
   // Turn raw omega into simplexes
   omega_simplex = simplex_create(omega_raw, K - 1);
   
   for(k in 1:K){
     theta[k] = mu[k];
     theta[k + K] = sigma2[k];
     theta[k + 2*K] = omega_simplex[k];
   }
   
   
   int_term = 1/(1.0 + beta_p)*integrate_1d(norm_mix_pdf_beta, negative_infinity(), positive_infinity(),
   theta, {beta_p}, {K}, 0.00000001);


   target += dirichlet_lpdf(omega_simplex | rep_vector(alpha_0, K)) + 
               simplex_create_lj(omega_raw, K - 1);
   
   target += inv_gamma_lpdf(sigma2 | 0.5*nu_0, 0.5*S_0);
   target += normal_lpdf(mu | mu_0, sqrt(kappa*sigma2));
   
   for(i in 1:n){
      target += (w*((1/beta_p) * exp(norm_mix_lpdf(y[i, 1] | K, mu , sigma2, omega_simplex))^(beta_p) - int_term));
   }
}

generated quantities {
   vector[K] omega_simplex;
   omega_simplex = simplex_create(omega_raw, K - 1);
   
}

