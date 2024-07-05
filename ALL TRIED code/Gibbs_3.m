function x = rdirichlet(n, alpha) % ARGUMENTS: the number of desired samples, the Dirichlet parameter vector % VALUE: a draw from the resulting Dirichlet distribution 
    l = length(alpha); x = zeros(n, l); 
    for i = 1:n x(i, :) = rgamma(l, alpha, 1) / sum(rgamma(l, alpha, 1)); 
end 