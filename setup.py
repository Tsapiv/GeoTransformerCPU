from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension


setup(
    name='geotransformer',
    version='1.0.0',
    packages=find_packages(include=['geotransformer', 'geotransformer.*']),
    package_data={'': ['.gitignore', '**/*.pth.tar']},
    ext_modules=[
        CppExtension(
            name='geotransformer.ext',
            sources=[
                'geotransformer/extensions/extra/cloud/cloud.cpp',
                'geotransformer/extensions/cpu/grid_subsampling/grid_subsampling.cpp',
                'geotransformer/extensions/cpu/grid_subsampling/grid_subsampling_cpu.cpp',
                'geotransformer/extensions/cpu/radius_neighbors/radius_neighbors.cpp',
                'geotransformer/extensions/cpu/radius_neighbors/radius_neighbors_cpu.cpp',
                'geotransformer/extensions/pybind.cpp',
            ],
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
