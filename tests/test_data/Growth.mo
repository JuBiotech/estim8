model Growth
Real X(start = X0, fixed = true);
parameter Real X0 = 0.1;
parameter Real mu_max = 0.5;
equation
der(X) = mu_max*X;
end Growth;
