"""
This module defines functions that can be used together with linkcode Sphinx
extension to link source code hosted on GitHub from the API documentation page.
"""
import inspect
import os
import subprocess
import sys

from functools import partial
from operator import attrgetter

REVISION_CMD = 'git rev-parse --short HEAD'


def _get_git_revision():
    """
    Gets a short hash of the current commit.
    """
    try:
        revision = subprocess.check_output(REVISION_CMD.split()).strip()
    except (subprocess.CalledProcessError, OSError):
        print('Failed to execute git to get revision')
        return None
    return revision.decode('utf-8')


def _linkcode_resolve(domain, info, package, url_fmt, revision):
    """
    Determines a link to the online source for a class/method/function/etc.

    This is called by the ``sphinx.ext.linkcode`` Sphinx extension.

    An example with a long-untouched module that everyone has:

    >>> _linkcode_resolve('py', {'module': 'tty',
    ...                          'fullname': 'setraw'},
    ...                   package='tty',
    ...                   url_fmt='https://hg.python.org/cpython/file/'
    ...                           '{revision}/Lib/{package}/{path}#L{lineno}',
    ...                   revision='xxxx')
    'https://hg.python.org/cpython/file/xxxx/Lib/tty/tty.py#L18'
    """
    # pylint: disable=broad-except
    if revision is None:
        return None
    if domain != 'py':
        return None
    if not info.get('module') or not info.get('fullname'):
        return None

    class_name = info['fullname'].split('.')[0]
    module = __import__(info['module'], fromlist=[class_name])
    obj = attrgetter(info['fullname'])(module)

    try:
        function = inspect.getsourcefile(obj)
    except Exception:
        function = None
    if not function:
        try:
            function = inspect.getsourcefile(sys.modules[obj.__module__])
        except Exception:
            function = None
    if not function:
        return None

    function = os.path.relpath(
        function, start=os.path.dirname(__import__(package).__file__))
    try:
        lineno = inspect.getsourcelines(obj)[1]
    except Exception:
        lineno = ''
    return url_fmt.format(
        revision=revision, package=package, path=function, lineno=lineno)


def make_linkcode_resolve(package, url_fmt):
    """
    Returns a ``linkcode_resolve`` function for the given URL format.

    ``revision`` is a git commit reference (hash or name);
    ``package`` is the name of the root module of the package; and
    ``url_fmt`` is a GitHub link format string, e.g.,
    'https://github.com/ORG/REPO/blob/{revision}/{package}/{path}#L{lineno}'
    """
    revision = _get_git_revision()
    return partial(
        _linkcode_resolve, revision=revision, package=package, url_fmt=url_fmt)
