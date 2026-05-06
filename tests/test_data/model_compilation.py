import argparse


def compileModel(modelname: str, output_name: str | None = None):
    modelpath = f"{os.getcwd()}/tests/test_data/{modelname}.mo"
    MO_model = ModelicaSystem(modelpath, modelname)
    # compile
    fmu = MO_model.convertMo2Fmu()
    if output_name is None:
        output_name = modelname
    shutil.copy2(fmu, f"{os.getcwd()}/tests/test_data/{output_name}.fmu")
    return fmu


if __name__ == "__main__":
    import os
    import shutil

    from OMPython import ModelicaSystem, OMCSessionZMQ

    parser = argparse.ArgumentParser()
    parser.add_argument("--modelname", required=True)
    parser.add_argument("--output_name", default=None, required=False)
    args, _ = parser.parse_known_args()
    compileModel(args.modelname, args.output_name)
