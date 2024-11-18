import tomllib
from pprint import pformat
from docutils import nodes
from importlib import import_module
from pprint import pformat
from docutils.parsers.rst import Directive
from sphinx import addnodes
import subprocess
import os
import sys
import ai3

cur_directory = os.path.dirname(__file__)
parent_directory = os.path.abspath(
    os.path.join(cur_directory, '..'))
sys.path.append(parent_directory)


class PrettyPrintIterable(Directive):
    required_arguments = 1

    def run(self):
        module_path, member_name = self.arguments[0].rsplit('.', 1)
        module = import_module(module_path)
        member = getattr(module, member_name)

        code = pformat(
            member,
            indent=2,
            width=80,
            depth=3,
            compact=False,
            sort_dicts=False,
        )

        literal = nodes.literal_block(code, code)
        literal['language'] = 'python'

        return [addnodes.desc_content('', literal)]


with open(os.path.join(parent_directory, 'pyproject.toml'), 'rb') as f:
    pyproject_data = tomllib.load(f)

pkg_name = pyproject_data.get('project', {}).get('name', '')
repo = pyproject_data.get('project', {}).get('homepage', '')
docs = pyproject_data.get('project', {}).get('documentation', '')
repo_main = repo + '/tree/main'
repo_src = repo_main + '/src/ai3'
repo_csrc = repo_src + '/csrc'

rst_prolog = f'''
.. _repo: {repo}
.. |repo| replace:: **Source Code**
.. _script: {repo_main + '/bench/gather.py'}
.. |script| replace:: *script*
.. _custom: {repo_src + '/custom'}
.. |custom| replace:: *custom*
.. _custom_cmake: {repo_main + '/cmake/custom.cmake'}
.. |custom_cmake| replace:: *custom.cmake*
.. _model_zoo: {repo_main + '/model_zoo/models.py'}
.. |model_zoo| replace:: *model_zoo*
.. _doc: {docs}
.. |doc| replace:: **Documentation**
.. |name| replace:: *{ai3.__name__}*
.. |pkg_name| replace:: *{pkg_name}*

.. _cuDNN: https://developer.nvidia.com/cudnn
.. |cuDNN| replace:: *cuDNN*
.. _SYCL: https://www.khronos.org/sycl
.. |SYCL| replace:: *SYCL*
'''

project = ai3.__name__
copyright = '2024, Timothy Cronin'
author = 'Timothy Cronin'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.doctest',
    'breathe',
]

master_file = 'index'

exclude_patterns = ['_build', 'Thumbs.db',
                    '.DS_Store', 'intro.rst', 'performance.rst']
html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_css_files = [
    'custom.css',
]

html_show_sourcelink = False
html_sidebars = {
    '**': []
}

version = ai3.__version__
version_parts = version.split('.')
dev = len(version_parts) != 3 or not all(part.isdigit()
                                         for part in version_parts)

if dev:
    version_match = 'latest'
else:
    version_match = version

html_theme_options = {
    'navbar_align': 'left',
    'navbar_center': ['version-switcher', 'navbar-nav'],
    'switcher': {
        'json_url': f'{docs}/latest/_static/switcher.json',
        'version_match': version_match,
    },
    'check_switcher': True,
    'icon_links': [
        {
            'name': 'GitHub',
            'url': repo,
            'icon': 'fa-brands fa-github',
            'type': 'fontawesome',
        },
        {
            'name': 'PyPI',
            'url': 'https://pypi.org/project/aithree/',
            'icon': 'fa-brands fa-python',
            'type': 'fontawesome',
        }
    ]
}

breathe_projects = {
    pkg_name: os.path.join(os.getcwd(), 'doxygen', 'xml')
}
breathe_default_project = pkg_name

doctest_global_setup = '''
import ai3
import torch
from torch import nn
import torchvision
from example.manual_conv2d import ConvNet
'''


def setup(app):
    app.add_directive('pprint', PrettyPrintIterable)
    subprocess.call('make clean', shell=True, cwd=cur_directory)
    subprocess.call(
        f'PROJECT_NAME=*{pkg_name}* REPO_MAIN={repo_main} REPO_SRC={repo_src} REPO_CSRC={repo_csrc} doxygen',
        shell=True, cwd=cur_directory)
