% Jake Zhao
% Department of Economics
% University of Wisconsin-Madison
% Fall 2011
% Introduction to Matlab

clc

max_integer = 10^5;
prime_vec = false(1,max_integer);

%%%%%%%%%%
% Serial %
%%%%%%%%%%
tic

for i=2:max_integer
    still_prime = true;
    if(mod(i,2) == 0 && i ~= 2)
        %disp([num2str(i), ' is composite']);
        still_prime = false;
    end

    j = 3;
    while(j <= sqrt(i) && still_prime)
        if(mod(i,j) == 0)
            %disp([num2str(i), ' is composite']);
            still_prime = false;
        end
        j = j+2;
    end

    if(still_prime)
        prime_vec(i) = true;
        %disp([num2str(i), ' is prime']);
    end
end

toc

%%%%%%%%%%%%%%
% Vectorized %
%%%%%%%%%%%%%%
tic

integer_vec = 1:max_integer;
divisors_vec = [2, 3:2:sqrt(max_integer)];

integer_mat = ones(length(divisors_vec),1)*integer_vec;
divisors_mat = divisors_vec'*ones(1,max_integer);

divisors_mat(integer_mat<=divisors_mat) = 0;

remainder = mod(integer_mat, divisors_mat);
remainder = min(remainder); % The prime numbers will have non-zero value while the composites will have zero value
remainder(1) = 0; % Special case since one is not prime

toc

%%%%%%%%%%%%
% Built-in %
%%%%%%%%%%%%
tic
isprime(1:max_integer);
toc