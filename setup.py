import setuptools

with open("src/langchain_decorators/__init__.py","rt") as f:
        for line in f.readlines():
            if line.startswith("__version__"):
                __version__ = line.split("=")[1].strip(" \n\"")

setuptools.setup(name='langchain-decorators',
                version=__version__,
                description='syntactic sugar for langchain',
                long_description=open('README.md').read(),
                long_description_content_type='text/markdown',
                author='Juraj Bezdek',
                author_email='juraj.bezdek@blip.solutions',
                url='https://github.com/ju-bezdek/langchain-decorators',
                package_dir={"": "src"},
                packages=setuptools.find_packages(where="src"),
                license='MIT License',
                zip_safe=False,
                keywords='langchain',

                classifiers=[
                ],
                python_requires='>=3.9',
                install_requires=[
                    "langchain",
                    "promptwatch"
                ]
                )
