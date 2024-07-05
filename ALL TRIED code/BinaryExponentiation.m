clear
close all
clc 

% Your code here
output = (pow(5, 5));
disp(output)

% Function definitions
function result = pow(a, x)
% Definition of the pow function
if x == 0
    result = 1; % for a^0
    return;
end

result = 1;
while x > 0
    if bitand(x, 1) == 1 % update the ans = ans*a when LSB = 1
        result = result * a;
    end
    x = bitshift(x, -1); % right shift the power
    a = a * a; % square the base after each right shift
end
end

