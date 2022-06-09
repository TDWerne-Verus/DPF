from setuptools import setup, find_packages

VERSION = '1.02'
DESCRIPTION = 'DPF Data Loader'
LONG_DESCRIPTION = 'Parse DPF data directly from the V-Drive'

# Setting up
setup(
        name="dpf_data_loader", 
        version=VERSION,
        author="Daniel Bowen",
        author_email="daniel.bowen@verusresearch.net",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        test_suite="nose.collector",
        tests_require=["nose"],
        install_requires=["pandas"],
        
        keywords=['python', 'dpf'],
        classifiers= [
            "Programming Language :: Python :: 3",
            "Operating System :: Microsoft :: Windows",
        ]
)