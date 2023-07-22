{ pkgs ? import <nixpkgs> { } }:

pkgs.mkShell rec {
  buildInputs = with pkgs; [
    stdenv.cc.cc.lib
    gnumake
    gnused
    python311
    cudaPackages.cudatoolkit
    cudaPackages.cudnn
    cudaPackages.tensorrt
    pipenv
    zlib
    util-linux # lscpu
  ];

  shellHook = with pkgs; ''
    export LD_LIBRARY_PATH="${lib.makeLibraryPath buildInputs}:$LD_LIBRARY_PATH"
    export LD_LIBRARY_PATH="${cudaPackages.cudatoolkit}/lib:$LD_LIBRARY_PATH"
    export CUDA_DIR="${cudaPackages.cudatoolkit}"
    export XLA_FLAGS="--xla_gpu_cuda_data_dir=${cudaPackages.cudatoolkit}"
    export PIPENV_VENV_IN_PROJECT=1
    export PIPENV_VERBOSITY=-1
    [ -v DOCKER ] && [ ! -f ".venv/bin/activate" ] && pipenv sync
    [ ! -v DOCKER ] && [ ! -f ".venv/bin/activate" ] && pipenv sync --dev
    [ ! -v DOCKER ] && exec pipenv shell --fancy
  '';
}
