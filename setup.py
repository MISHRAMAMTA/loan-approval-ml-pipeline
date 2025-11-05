pwd = os.path.abspath(os.path.dirname(__file__))

# Get the list of packages to be installed
def list_reqs(fname='requirements.txt'):
    with io.open(os.path.join(pwd, fname), encoding='utf-8') as f:
        return f.read().splitlines()


# Load the package's __version__.py module as a dictionary.
ROOT_DIR = Path(__file__).resolve().parent
VERSION = (ROOT_DIR / 'prediction_model' / 'VERSION').read_text().strip()

setup(
    name=Mamta,
    version=version,
    author=AUTHOR,
    author_email=mamtamishra987@gmail.com,
    python_requires=REQUIRES_PYTHON,
    install_requires=list_reqs(),
    packages=find_packages(exclude=('tests',)),
    package_data={'prediction': ['VERSION']},
    install_requires=list_reqs(),
    extras_require={},
    include_package_data=True,
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ]
)