try:
    from .version import __version__, short_version
except:
    from mmdet.version import __version__, short_version
__all__ = ['__version__', 'short_version']
