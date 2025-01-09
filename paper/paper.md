title: 'estim8 - A python toolbox for bioprocess modeling and parameter estimation'
tags:
  - Functional Mockup Units
  - Functional Mockup Interface
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
- DAE modeling, events & discontinuitees
- no open-source tool with flavours for biotechnological applications that --> State of the field

# Materials and Methods
$\texttt{estim8}$ is an open source Python package compatible and tested with Windows and Linux/Unix platforms. It comprises 10 modules: `estimator`, `models`, `datatypes`, `error_models`, `workers`, `generalized_islands`, `objective`, `optimizers`, `utils` and `visualization`. The modular, object-oriented architecture allows for easy expansion by new implementations, like e.g. custom simulators or cost functions. At the very core, $\texttt{estim8}$ currently features interactive simulation of $\textit{Functional Mockup Units}$ (FMUs) in Python utilizing FMPy [@RN14]. Parameter estimation and functionality for uncertainty quantification rely on the packages SciPy [@RN16] and pygmo [@BISCANI2020].

# Results and discussion
## Workflow


## Architecture and Scalability

# Conclusions



### Author contributions

### Acknowledgements

### Competing interests
No competing interest is declared.



# Bibliography
