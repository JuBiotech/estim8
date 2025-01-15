---
title: 'estim8 - A python toolbox for bioprocess modeling and parameter estimation'
tags:
  - Functional Mock-up Units
  - Functional Mock-up Interface
  - Bioprocess modeling
  - Parameter estimation
  - Python
  - Uncertainty quantification


authors:
  - name: Tobias Latour
    orcid:
    affiliation: 1
  - name: Daniel Strohmeier
    orcid:
    affiliation: 2
  - name: Wolfgang Wiechert
    orcid: 0000-0001-8501-0694
    affiliation: "1, 3"
  - name: Stephan Noack
    orcid: 0000-0001-9784-3626
    affiliation: 1

affiliations:
 - name: Institute for Bio- and Geosciences (IBG-1), Forschungszentrum Jülich GmbH, Jülich, Germany
   index: 1
 - name: Institute for Sustainable Hydrogen Economy (INW), Forschungszentrum Jülich GmbH, Jülich, Germany
   index: 2
 - name: Computational Systems Biotechnology, RWTH Aachen University, Aachen, Germany
   index: 3

date:
bibliography:
  - references.bib
---

# Summary

# Statement of Need
Mathematical modeling has become a pivotal tool in biotechnological research [@RN29] and industrial bioprocess development [@RN27; @RN28] by parametrizing the information gained from experimental data and creating digital twins. While ordinary differential equations (ODEs) are commonly used to describe continuous biological systems, many biotechnological applications require differential algebraic equation (DAE) systems to handle discontinuities, discrete events, physical constraints, and embedded optimization criteria [@RN26].

Parameter estimation is crucial in this context, as many model parameters cannot be determined a priori and must be fitted to experimental data. Although several general-purpose software tools for simulation and parameter estimation exist, they currently have significant limitations: many only support ODE systems [@RN30; @RN33; @RN34; @RN31], require substantial workarounds for biological problems [@RN35; @RN36], or depend on proprietary MATLAB licenses for DAE simulation [@RN32].

To address these limitations, we present $\texttt{estim8}$: a Python-based toolbox for simulation and parameter estimation of dynamic models. Built on the Functional Mock-up Interface (FMI) standard, $\texttt{estim8}$ provides specialized functionality for biotechnological applications, particularly in handling experimental replicates. By supporting model definition and simulation export from various FMI compliant third-party software, including the open-source OpenModelica platform [@RN22], $\texttt{estim8}$ enables comprehensive DAE support and convenient event handling.



# Materials and Methods
$\texttt{estim8}$ is an open source Python package compatible and tested with Windows and Linux/Unix platforms. It comprises 10 modules: `estimator`, `models`, `datatypes`, `error_models`, `workers`, `generalized_islands`, `objective`, `optimizers`, `utils` and `visualization`. The modular, object-oriented architecture allows for easy expansion by new implementations, like e.g. custom simulators or cost functions. At the very core, $\texttt{estim8}$ currently features interactive simulation of $\textit{Functional Mock-up Units}$ (FMUs) in Python utilizing FMPy [@RN14] via ModelExchange and CoSimulation. Parameter estimation and functionality for uncertainty quantification rely on the packages SciPy [@RN15] and pygmo [@RN21].

# Results and discussion
## Workflow
- mathematical modeling in 3rd party software that is part of the FMI standard
  - e.g. OpenModelica [@RN22], which is open source and offers a interactive modeling environment with graphical
  - model export to FMU, both CoSimulation and ModelExchange. Special solvers can be packed to fmu if needed
  - enables DAE, events and discontinities
- loading the fmu in estim8's `FmuModel`
- datastructure: experiment objects, consisting of measurements, with individual error model, observation mappings
- replicate modeling:
    - best practice to run biological experiments in replicates to account for variability and heterogeinity, essential for enhancing the statistical quality [@RN19]
    - differing conditions between replicates, e.g. different biomass concentrations in reactor vessels need to be accounted by the modeler.
    - $\texttt{estim8}$ defines model replicates, which can be connected by global parameters or may have replicate specific local parameters

- heart of $\texttt{estim8}$ is the `estimator` class, which stores all necessary information and manages parameter estimation tasks. Different solvers can be used for optimization tasks.
- re-estimation based methods for parameter identifiability and uncertainty quantification by means of profile likelihoods and Monte carlo sampling.
- a thourough set of visualization functions for analysis of simulations, compared to data and parameter estimations


## Architecture and Scalability
A set of the population-based solvers enable parallel evaluations of cost functions, especially the pygmo allows fo highly parallelizable setups. With respect to the costly evaluation of expperimental replicates, $\texttt{estim8}$ introduces a second layer of parallelization of model replicate simulations. This setup utilizes a federated computing principle using

# Conclusions



### Author contributions

### Acknowledgements

### Competing interests
No competing interest is declared.



# Bibliography
