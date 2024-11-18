import re
import os
import ai3
import textwrap
import inspect

GITHUB_RAW = 'https://raw.githubusercontent.com/KLab-AI3/ai3/main'


def prune_rst_links_and_remove_args(obj) -> str:
    docstring = inspect.getdoc(obj)
    assert (docstring)

    docstring = re.sub(r':func:`([^`]+)`', r'*\1*', docstring)
    docstring = re.sub(r':class:`([^`]+)`', r'*\1*', docstring)
    docstring = re.sub(r':type:`([^`]+)`', r'*\1*', docstring)
    docstring = textwrap.dedent(docstring).strip()

    paragraphs = docstring.split('\n\n')
    paragraphs = [p for p in paragraphs if not p.startswith('Args:')]

    docstring = '\n\n'.join(paragraphs)

    return docstring


def clean_rst_prolog():
    from docs.conf import rst_prolog
    if rst_prolog.startswith('\n'):
        rst_prolog = rst_prolog.lstrip('\n')

    if not rst_prolog.endswith('\n'):
        rst_prolog += '\n'

    return rst_prolog


def fix_paths(contents):
    return re.sub(
        r'_static/(\S+)',
        lambda match: f"{GITHUB_RAW}/docs/_static/{match.group(1)}",
        contents
    )


if __name__ == '__main__':
    with open(os.path.join('docs', 'intro.rst'), 'r') as file:
        intro_lines = file.readlines()
        filtered_lines = [
            line for line in intro_lines
            if not line.startswith('.. include:')]
        intro = fix_paths(''.join(filtered_lines))
    with open(os.path.join('docs', 'performance.rst'), 'r') as file:
        performance = file.read()
        performance = fix_paths(performance)
    with open(os.path.join('docs', 'home_footnotes'), 'r') as file:
        footnotes = file.read()
    with open(os.path.join('docs', 'algo_platform_tables.rst'), 'r') as file:
        algo_platform_tables = file.read()
    with open('README.rst', 'w') as readme_file:
        readme_file.write(clean_rst_prolog())
        readme_file.write('\n')
        intro = re.sub(r":ref:`([^<]+) <.*?>`", r"\1", intro).strip()
        readme_file.write(intro)

        doc = prune_rst_links_and_remove_args(ai3)
        readme_file.write(''.join(doc.splitlines(keepends=True)[1:]))
        readme_file.write('\n\n')

        sc_doc = prune_rst_links_and_remove_args(ai3.swap_operation)

        readme_file.writelines(['*swap_operation*\n',
                                '~~~~~~~~~~~~~\n',
                                sc_doc])
        readme_file.write('\n\n')

        sb_doc = prune_rst_links_and_remove_args(ai3.convert)
        readme_file.writelines(['*convert*\n',
                                '~~~~~~~~~~~~~~\n',
                                sb_doc])
        readme_file.write('\n\n')

        readme_file.write(performance)
        readme_file.write('\n\n')

        readme_file.write(algo_platform_tables)
        readme_file.write('\n')
        readme_file.write(footnotes)
