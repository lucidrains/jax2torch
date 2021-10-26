from setuptools import setup, find_packages

setup(
  name = 'jax2torch',
  packages = find_packages(exclude=[]),
  version = '0.0.2',
  license='MIT',
  description = 'Jax 2 Torch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/jax2torch',
  keywords = [
    'jax',
    'pytorch'
  ],
  install_requires=[
    'torch>=1.6',
    'jax>=0.2.20'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
