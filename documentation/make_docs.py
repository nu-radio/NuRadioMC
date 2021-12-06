import subprocess
import os
import sys
import argparse
import logging

logging.basicConfig()
logger = logging.getLogger(name="Make_docs")
logger.setLevel(logging.INFO)

def filter_errs(errs, filter_string, multi_line=[], exclude=None, include_line_numbers=False):
    """Selects only lines that contain filter_string
    
    Optionally can include multiple lines around each match,
    and exclude matches containing 'exclude'
    """
    line_numbers = []
    for filter_substr in filter_string.split('|'):
        line_numbers += [
            i for i,err in enumerate(errs)
            if filter_substr in err
        ]
    line_numbers = sorted(list(set(line_numbers))) # remove duplicates
    matches = [(j, errs[j]) for j in line_numbers]
    
    if exclude != None:
        matches = [k for k in matches if (not exclude in k[1])]

    if len(multi_line) == 2:
        matches = [
            (k[0], errs[k[0]-multi_line[0]:k[0]+multi_line[1]])
            for k in matches
        ]
    
    if not include_line_numbers:
        return [k[1] for k in matches]
    else:
        return matches

err_sections = [
    '[reference target not found]',
    '[title underline too short]',
    '[stub file not found]',
    '[unexpected section title]',
    '[numpydoc]',
    '[indentation]',
    '[toctree]',
    '[other]'
]

# we try to classify some of the errors to be helpful
# these are tuples (name, string_to_match)
error_classes = [
    ('reference target not found', 'reference target not found|undefined label'),
    ('title underline too short', 'Title underline too short'),
    ('stub file not found', 'stub file not found'),
    ('unexpected section title', 'Unexpected section title'),
    ('numpydoc', 'numpydoc'),
    ('indentation', 'expected indent|expected unindent'),
    ('toctree', 'toctree')
]

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--no-clean', default=False, const=True, action='store_const')
    argparser.add_argument('--debug', default=False, const=True, action='store_const')
    parsed_args = argparser.parse_args()
    if parsed_args.debug: # set up the logger to also write output to make_docs.log
        logfile = 'make_docs.log'
        with open(logfile, 'w') as file:
            pass
        file_logger = logging.FileHandler(logfile)
        file_logger.setLevel(logger.getEffectiveLevel())
        logger.addHandler(file_logger)

    doc_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(doc_path)

    # we exclude scripts, examples and tests from the code documentation,
    # as apidoc doesn't really handle those well
    exclude_modules = []
    exclude_modules.append('../**/test')
    exclude_modules.append('../**/tests')
    exclude_modules.append('../**/scripts')
    exclude_modules.append('../**/examples')
    exclude_modules.append('../**/eventbrowser') 
    exclude_modules.append('../**/setup.py')
    exclude_modules.append('../**/CPPAnalyticRayTracing') # C code also doesn't work right now

    # create the automatic code documentation with apidoc
    for module in ['NuRadioReco', 'NuRadioMC']:
        output_folder = 'source/{}/apidoc'.format(module)
        if os.path.exists(output_folder):
            if not parsed_args.no_clean: # remove old apidoc folder
                logger.info('Removing old apidoc folder: {}'.format(output_folder))
                subprocess.check_output(['rm', '-rf', output_folder])

        module_path = '../{}/'.format(module)

        logger.info("Creating automatic documentation files with apidoc:")
        logger.info("excluding modules: {}".format(exclude_modules))
        subprocess.call(
            [
                'sphinx-apidoc', '-efMT', '--ext-autodoc', '--ext-intersphinx',
                '--ext-coverage', '--ext-githubpages', '-o', output_folder,
                module_path, *exclude_modules
            ]
        )
        # We don't use the top level NuRadioReco.rst / NuRadioMC.rst toctrees,
        # so we remove them to eliminate a sphinx warning
        subprocess.check_output([
            'rm', os.path.join(output_folder, '{}.rst'.format(module))])

    if not parsed_args.no_clean:
        logger.info('Removing old \'build\' directory...')
        subprocess.check_output(['make', 'clean'])
    sphinx_log = subprocess.run(['make', 'html'], stderr=subprocess.PIPE) # stdout=subprocess.PIPE,

    errs = sphinx_log.stderr.decode().split('\n')
    # output = sphinx_log.stdout.decode().split('\n')

    # broken cross-references. We ignore warnings originating from docstrings
    match_str = 'reference target not found|undefined label'
    xref_errs = filter_errs(errs, match_str)
    xref_errs_in_docstrings = filter_errs(xref_errs, 'docstring')
    xref_tofix = filter_errs(errs, match_str, exclude='docstring')
    logger.info('{} broken xrefs, of which {} outside docstrings'.format(len(xref_errs), len(xref_tofix)))

    # broken sections
    match_str = 'Title underline too short'
    section_errs = filter_errs(errs, match_str, multi_line=(0,4))
    section_errs_tofix = filter_errs(errs, match_str, multi_line=(0,4), exclude='docstring')
    logger.info('{} bad section titles, of which {} outside docstrings'.format(
        len(section_errs), len(section_errs_tofix)))

    # bad docstrings
    match_str = 'Unexpected section title'
    unexpected_title_errs = filter_errs(errs, match_str, multi_line=(1,4))
    logger.info('{} bad docstring sections'.format(len(unexpected_title_errs)))

    match_str = 'numpydoc'
    numpydoc_errs = filter_errs(errs, match_str, multi_line=(0,3))
    logger.info('{} numpydoc errors'.format(len(numpydoc_errs)))

    # missing stubs in .rst files - if these happen, may have to rerun apidoc!
    match_str = 'stub file not found'
    stub_errs = filter_errs(errs, match_str)
    logger.info('{} stub errs'.format(len(stub_errs)))

    match_str = 'expected indent|expected unindent'
    indentation_errs = filter_errs(errs, match_str)
    indentation_errs_outside_docstrings = filter_errs(errs, match_str, exclude='docstring')
    logger.info(
        '{} indentation errors, of which {} outside docstrings'.format(
        len(indentation_errs), len(indentation_errs_outside_docstrings))
    )

    match_str = 'toctree'
    toctree_errs = filter_errs(errs, match_str)
    logger.info(
        '{} toctree errors (missing document or missing entry)'.format(len(toctree_errs))
    )
    
    all_errs = (
        ['[reference target not found]'] + xref_tofix # we exclude broken xrefs in docstrings
        + ['[title underline too short]'] + [k for j in section_errs for k in j]
        + ['[stub file not found]'] + stub_errs 
        + ['[unexpected section title]'] + [k for j in unexpected_title_errs for k in j]
        + ['[numpydoc]'] + [k for j in numpydoc_errs for k in j]
        + ['[indentation]'] + indentation_errs
        + [err_sections[6]] + toctree_errs
    )

    other_errs = [
        err for err in errs 
        if (err not in all_errs) & (err not in xref_errs)
    ]

    fixable_errors = (len(all_errs) > len(err_sections))

    if fixable_errors:
        logger.warning("The documentation was not built without errors. Please fix the following errors!")
        print('\n'.join(all_errs))
    elif len(other_errs):
        logger.warning((
            "make_docs found some errors but doesn't know what to do with them.\n"
            "The documentation will not be rejected, but consider fixing the following anyway:"
            ))
        print('\n'.join(other_errs))

    if parsed_args.debug:
        logger.info('Logging output under {}'.format(logfile))
        with open(logfile, 'w') as file:
            # file.write('[stdout]\n')
            # file.write('\n'.join(output))
            # file.write('\n')
            file.write('[broken references in docstrings] # these can be ignored\n')
            file.write('\n'.join(xref_errs_in_docstrings))
            file.write('\n')
            file.write('\n'.join(all_errs))
            file.write('\n')
            file.write('[other]\n')
            file.write('\n'.join(other_errs))
    
    if fixable_errors:
        sys.exit(1)