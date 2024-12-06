import argparse


def compileModel(modelname):
    modelpath = f"{os.getcwd()}/tests/test_data/{modelname}.mo"
    MO_model = ModelicaSystem(modelpath, modelname)
    # compile
    fmu = MO_model.convertMo2Fmu()

    return fmu


if __name__ == "__main__":
    import os
    import shutil

    from OMPython import ModelicaSystem, OMCSessionZMQ

    parser = argparse.ArgumentParser()
    parser.add_argument("--modelname", required=True)
    args, _ = parser.parse_known_args()

    print(compileModel(args.modelname))
