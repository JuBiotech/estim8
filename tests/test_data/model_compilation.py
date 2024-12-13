import argparse


def compileModel(modelname):
    modelpath = f"{os.getcwd()}/tests/test_data/{modelname}.mo"
    MO_model = ModelicaSystem(modelpath, modelname)
    # compile
    fmu = MO_model.convertMo2Fmu()
    shutil.copy2(fmu, f"{os.getcwd()}/tests/test_data/{modelname}.fmu")
    print(f"copied fmu to {os.getcwd()}/tests/test_data/{modelname}.fmu")
    return fmu


if __name__ == "__main__":
    import os
    import shutil

    from OMPython import ModelicaSystem, OMCSessionZMQ

    parser = argparse.ArgumentParser()
    parser.add_argument("--modelname", required=True)
    args, _ = parser.parse_known_args()
