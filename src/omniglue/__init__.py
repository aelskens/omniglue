from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("omniglue")
except PackageNotFoundError:
    __version__ = "dev"