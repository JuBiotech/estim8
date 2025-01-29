model SimpleBatch

  // define real states
  Real X(start = X0, fixed = true);
  Real S(start = S0, fixed = true);
  Real mu;

  // define variables
  parameter Real X0 = 0.2;
  parameter Real S0 = 10;
  parameter Real mu_max = 0.4;
  parameter Real Ks = 0.01;
  parameter Real Y_XS = 0.35;

equation
  der(X) = mu*X;
  der(S) = -mu*X/Y_XS;
  mu = mu_max*S/(Ks + S);

// prevent negative substrate concentrations
 when (S<=0) then
  reinit(S, 0);
 end when;

end SimpleBatch;
