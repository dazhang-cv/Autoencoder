function output = normalization(input)

[m,n] = size(input);

maxval = max(input);
minval = min(input);

for i = 1:n
    output(:,i) = (input(:,i) - minval(i))/(maxval(i) - minval(i));
end
