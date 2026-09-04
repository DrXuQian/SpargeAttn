"""Build the actlize-backed PPU SpargeAttention extension.

This is deliberately separate from setup.py: NVIDIA's SM80/89/90 source graph
contains PTX contracts that are not valid PPU input.  A PPU build must opt in to
the PPU source graph rather than compile everything and hope unused PTX drops.
"""

import os
import subprocess
import sysconfig
from pathlib import Path

import torch
from setuptools import find_packages, setup
import torch.utils.cpp_extension as torch_cpp_extension


ROOT = Path(__file__).resolve().parent
PPU_SDK = os.environ.get("PPU_SDK") or os.environ.get("PPU_HOME")
if not PPU_SDK:
    raise RuntimeError("Set PPU_SDK (or PPU_HOME) to the PPU SDK root")
PPU_SDK = str(Path(PPU_SDK).resolve())
HGCC = Path(PPU_SDK) / "bin" / "hgcc"
if not HGCC.is_file():
    raise RuntimeError(f"hgcc not found at {HGCC}")
os.environ["PATH"] = f"{HGCC.parent}:{os.environ.get('PATH', '')}"

ACTLIZE = ROOT / "third_party" / "actlize"
if not (ACTLIZE / "include" / "cute" / "tensor.hpp").is_file():
    raise RuntimeError(
        "actlize submodule is missing; run: "
        "git submodule update --init third_party/actlize"
    )
SDK_INCLUDE = Path(PPU_SDK) / "include"
TARGET_INCLUDES = sorted((Path(PPU_SDK) / "targets").glob("*/include"))
if not SDK_INCLUDE.is_dir() or not TARGET_INCLUDES:
    raise RuntimeError(
        "PPU SDK must provide include/ and targets/<triple>/include; "
        f"got SDK_INCLUDE={SDK_INCLUDE} targets={TARGET_INCLUDES}"
    )

ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
CXX_FLAGS = ["-O3", "-std=c++17", f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]
HGCC_FLAGS = [
    "--forward-unknown-to-host-compiler",
    "--forward-unknown-to-host-linker",
    "-arch=ppu_10",
    "-x", "hg",
    "-DSWITCH_TO_HGGCRT",
    "-Xcompiler", "-ftemplate-depth=8192",
    "-Xllvm", "-wno-loop-miss-transform",
    "-Xllvm", "-ppu-simt-branch=false",
    "-Xllvm", "-ppu-patch-fence-ppu=false",
    "-Xllvm", "-ppu-cg-to-kp1=true",
    "-Xllvm", "-ppu-fix-uninit=true",
    "-Xllvm", "-ppu-max-vreg-count=256",
    "-Xllvm", "-ppu-sink-matrix-addr=true",
    "-Xllvm", "-ppu-sink-async-addr=true",
    "-Xllvm", "-ppu-sink-load-addr=true",
    "-Xllvm", "-ppu-sink-store-addr=true",
    "--expt-relaxed-constexpr",
    "-DUSE_CLANG",
    "-DCUTLASS_VERSIONS_GENERATED",
    "-DCUTLASS_USE_PACKED_TUPLE=1",
    "-DCUTE_USE_PACKED_TUPLE=1",
    "-DUSE_PPU=1",
    "-DUSE_AIU=1",
    f"-D_GLIBCXX_USE_CXX11_ABI={ABI}",
    "-O3",
    "-std=c++17",
    "--use_fast_math",
    "-fPIC",
]

DEVICE_SOURCES = [
    ROOT / "csrc/qattn/ppu/block_sparse_ppu.cu",
    ROOT / "csrc/qattn/ppu/radial_sparse_ppu.cu",
    ROOT / "csrc/qattn/ppu/quant_ppu.cu",
]
DEVICE_INCLUDES = [
    ROOT / "csrc" / "qattn" / "ppu",
    ACTLIZE / "include",
    SDK_INCLUDE,
    *TARGET_INCLUDES,
    Path(sysconfig.get_paths()["include"]),
    *[Path(path) for path in torch_cpp_extension.include_paths()],
]
PPU_LIB = Path(PPU_SDK) / "lib"
PPU_LIBRARIES = ["hggc_wrapper", "hggcrt1", "hggc", "hg_wrapper"]
for library in PPU_LIBRARIES:
    if not (PPU_LIB / f"lib{library}.so").is_file():
        raise RuntimeError(f"PPU runtime library is missing: {PPU_LIB}/lib{library}.so")


class PpuBuildExtension(torch_cpp_extension.BuildExtension):
    """Compile PPU device TUs with hgcc, then use the host linker for pybind."""

    def build_extensions(self):
        object_dir = Path(self.build_temp) / "ppu_obj"
        object_dir.mkdir(parents=True, exist_ok=True)
        device_objects = []
        include_flags = [f"-I{path}" for path in DEVICE_INCLUDES]
        for source in DEVICE_SOURCES:
            output = object_dir / f"{source.stem}.o"
            command = [
                str(HGCC), *HGCC_FLAGS, *include_flags,
                "-c", str(source), "-o", str(output),
            ]
            print(f"[hgcc] {source.relative_to(ROOT)}", flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
            device_objects.append(str(output))

        for extension in self.extensions:
            extension.extra_objects = list(extension.extra_objects or []) + device_objects
        super().build_extensions()


extension = torch_cpp_extension.CppExtension(
    name="spas_sage_attn._qattn_ppu",
    sources=["csrc/qattn/ppu/pybind_ppu.cpp"],
    include_dirs=[
        str(ROOT / "csrc" / "qattn" / "ppu"),
        str(ACTLIZE / "include"),
        str(SDK_INCLUDE),
        *[str(path) for path in TARGET_INCLUDES],
    ],
    extra_compile_args=CXX_FLAGS,
    library_dirs=[str(PPU_LIB)],
    libraries=PPU_LIBRARIES,
    runtime_library_dirs=[str(PPU_LIB)],
    extra_link_args=[
        "-Wl,--disable-new-dtags",
        f"-Wl,-rpath,{PPU_LIB}",
        "-Wl,-rpath,$ORIGIN",
        "-ldl",
    ],
)

setup(
    name="spas_sage_attn",
    version="0.1.0+ppu",
    packages=find_packages(),
    ext_modules=[extension],
    cmdclass={
        "build_ext": PpuBuildExtension.with_options(use_ninja=False)
    },
    python_requires=">=3.9",
)
