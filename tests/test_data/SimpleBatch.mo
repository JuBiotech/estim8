model SimpleBatch

  // define real states
  Real X(start = X0, fixed = true);
  Real S(start = S0, fixed = true);
  Real mu;

  // define variables
  parameter Real X0 = 0.1;
  parameter Real S0 = 10;
  parameter Real mu_max = 0.5;
  parameter Real Ks = 0.01;
  parameter Real Y_XS = 0.5;

equation
  der(X) = mu*X;
  der(S) = -mu*X/Y_XS;
  mu = mu_max*S/(Ks + S);

// prevent negative substrate concentrations
 if (S<=0) then
  S=0;
 end if;

end SimpleBatch;
