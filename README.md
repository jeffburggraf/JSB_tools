**Installation**

OpenMC and ROOT packages may cause problems on Windows but should work fine on a Mac.

Install conda. Install mamba with `conda install mamba`. Create a new conda environment

` mamba env create --name env_name --file=path/to/JSB_tools/enviroment.yml`

Or, update a current conda/mamba environment by (from within the environment)

`mamba env update file=path/to/JSB_tools/enviroment.yml`

If there are errors, it's probably because of ROOT. Uncomment the `  - root` line in environment.yml (first line after beginning of dependencies). If the environment succeeds, then try installing ROOT from source and specify the desired python interpreter (good luck). 