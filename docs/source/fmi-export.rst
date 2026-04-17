FMI export with OpenModelica
============================

estim8 uses models distributed as `Functional Mock-up Units (FMUs) <https://fmi-standard.org/>`_ and executes them via the `FMPy <https://github.com/CATIA-Systems/FMPy>`_ library.
Any tool that exports FMI 2.0-compliant FMUs can be used; `OpenModelica <https://openmodelica.org/>`_ is the recommended open-source option.

FMU compilation
---------------

1. Download `OpenModelica <https://openmodelica.org/free-and-open-source-software/download/>`_ and follow the installation instructions
2. Open `OMEdit`
3. Load a Modelica class via **File → Open Model/Library File(s)**, or implement a new one

   .. figure:: images/OMEdit_openmodel.png
      :alt: Opening a model file in OMEdit

4. Optional: For ``CoSimulation``, open **Tools → Options → FMI** and set the solver to ``CVODE``

   .. figure:: images/OMEdit_fmi_settings.png
      :alt: FMI export settings in OMEdit showing platform selection and solver options

5. Right-click the model class in the model browser and select **Export → FMU**

   .. figure:: images/OM_fmi_export.png
      :alt: Exporting a model class as FMU via right-click context menu

Alternatively, compilation can be scripted with `OMPython <https://github.com/OpenModelica/OMPython>`_:

.. code-block:: python

    from OMPython import ModelicaSystem

    model = ModelicaSystem("MyModel.mo", "MyModel")
    model.convertMo2Fmu()   # writes MyModel.fmu to the working directory

.. important::

   **FMUs are platform-specific.**
   An FMU contains compiled native binaries (e.g. ``binaries/linux64/MyModel.so`` on Linux,
   ``binaries/win64/MyModel.dll`` on Windows).
   An FMU compiled on one operating system will **not** run on another.

   Always compile the FMU on — or for — the platform where parameter estimation will be executed.
   When running on a compute cluster or inside a container, compile the FMU in the same environment.

Using the FMU in estim8
-----------------------

Pass the path to the compiled FMU to :class:`~estim8.models.FmuModel`:

.. code-block:: python

    from estim8 import FmuModel

    model = FmuModel(
        path="MyModel.fmu",
        fmi_type="ModelExchange",   # or "CoSimulation"
        default_parameters={"k": 1.0},
    )

Both the ``ModelExchange`` and ``CoSimulation`` FMI interfaces are supported.
``ModelExchange`` integrates the model equations directly using a Python-side ODE solver,
while ``CoSimulation`` delegates time integration to the solver embedded in the FMU.
