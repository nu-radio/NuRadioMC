import subprocess
import os
import sys
import argparse
import logging
import re

logging.basicConfig()
logger = logging.getLogger(name="Make_docs")
logger.setLevel(logging.INFO)

# we do some error classification, because we don't want to fail on all errors
# This is done with simple string matching using re.search
# The classification is exclusive, i.e. once a match has been found the error 
# won't be included in a category lower down this list 
error_dict = {
        'reference target not found': dict(
            pattern='reference target not found|undefined label|unknown document', matches=[]
        ),
        'title underline too short': dict(
            pattern='Title underline too short', matches=[]
        ),
        'Unexpected section title': dict(
            pattern='Unexpected section title', matches=[]
        ),
        'numpydoc': dict(
            pattern='numpydoc', matches=[]
        ),
        'stub file not found': dict(
            pattern='stub file not found', matches=[]
        ),
        'indentation': dict(
            pattern='expected indent|expected unindent', matches=[]
        ),
        'toctree': dict(
            pattern='toctree', matches=[]
        ),
        'other': dict(
            pattern='', matches = []
        )
    }

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        prog="NuRadioMC.documentation.make_docs",
        description=(
            "A script to automatically generate the API documentation for NuRadioMC/NuRadioReco,"
            " and build the html documentation for online publishing."
        )
    )
    argparser.add_argument(
        '--no-clean', default=False, const=True, action='store_const',
        help=(
            "Do not delete existing html files and build only pages which have changed."
            " Useful if you are only modifying or adding (not moving/removing) pages.")
    )
    argparser.add_argument(
        '--debug', default=False, const=True, action='store_const',
        help="Store full debugging output in make_docs.log."
        )
    parsed_args = argparser.parse_args()
    if parsed_args.debug: # set up the logger to also write output to make_docs.log
        logfile = 'make_docs.log'
        with open(logfile, 'w') as file:
            pass
        file_logger = logging.FileHandler(logfile)
        file_logger.setLevel(logger.getEffectiveLevel())
        logger.addHandler(file_logger)
        pipe_stdout = None
    else:
        pipe_stdout = subprocess.PIPE # hide the stdout

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
        subprocess.run(
            [
                'sphinx-apidoc', '-efMT', '--ext-autodoc', '--ext-intersphinx',
                '--ext-coverage', '--ext-githubpages', '-o', output_folder,
                module_path, *exclude_modules
            ], stdout=pipe_stdout
        )
        # We don't use the top level NuRadioReco.rst / NuRadioMC.rst toctrees,
        # so we remove them to eliminate a sphinx warning
        subprocess.check_output([
            'rm', os.path.join(output_folder, '{}.rst'.format(module))])

    if not parsed_args.no_clean:
        logger.info('Removing old \'build\' directory...')
        subprocess.check_output(['make', 'clean'])
    sphinx_log = subprocess.run(['make', 'html'], stderr=subprocess.PIPE, stdout=pipe_stdout)

    # errs = sphinx_log.stderr.decode().split('\n')
    errs = re.split('\\x1b\[[0-9;]+m', sphinx_log.stderr.decode()) # split the errors
    # output = sphinx_log.stdout.decode().split('\n')

    for err in errs:
        if not err.split(): # whitespace only
            continue
        for key in error_dict.keys():
            if re.search(error_dict[key]['pattern'], err):
                error_dict[key]['matches'].append(err)
                break # we don't match errors to multiple categories

    fixable_errors = 0
    for key in error_dict.keys():
        if key == 'other': # we don't fail on these errors
            continue
        fixable_errors += len(error_dict[key]['matches'])
    
    print(2*'\n'+78*'-')
    if fixable_errors:
        logger.warning("The documentation was not built without errors. Please fix the following errors!")
        for key, item in error_dict.items():
            if len(item['matches']):
                print(f'[{key}]')
                print('\n'.join(item['matches']))
    elif len(error_dict['other']['matches']):
        logger.warning((
            "make_docs found some errors but doesn't know what to do with them.\n"
            "The documentation may not be rejected, but consider fixing the following anyway:"
            ))
        print('\n'.join(error_dict['other']['matches']))
    print(78*'-'+2*'\n')

    if sphinx_log.returncode:
        logger.error("The documentation failed to build, make_docs will raise an error.")
    # print(error_dict)
    if parsed_args.debug:
        logger.info('Logging output under {}'.format(logfile))
        with open(logfile, 'w') as file:
            # file.write('[stdout]\n')
            # file.write('\n'.join(output))
            # file.write('\n')
            for key, item in error_dict.items():
                if len(item['matches']):
                    file.write(f'\n[{key}]\n')
                    file.write('\n'.join(item['matches']))

    if fixable_errors or sphinx_log.returncode:
        sys.exit(1)