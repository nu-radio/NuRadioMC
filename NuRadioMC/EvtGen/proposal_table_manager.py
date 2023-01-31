#!/usr/bin/env python
from NuRadioMC.EvtGen.NuRadioProposal import ProposalFunctions
import proposal
import os
import shutil
import argparse
import logging
import sys

logger = logging.getLogger('ProposalTablesManager')

def produce_proposal_tables(config_file, tables_path=None):
    """
    produce proposal tables for all relevant flavors

    Parameters
    ----------
    config_file: string
        one of the default PROPOSAL configurations ['InfIce','SouthPole', 'MooresBay', 'Greenland']
    tables_path: string
        output path for the generated tables
    """
    proposal_func = ProposalFunctions(config_file=config_file, tables_path=tables_path, create_new=True)

    for particle_code in [-15, -13, 13, 15]:
        logger.warning(f"producing tables for {config_file}, particle {particle_code}")
        proposal_func._ProposalFunctions__get_propagator(particle_code=particle_code)


def get_compiler():
    # proposal (as of V 7.5.0) generates tables with different hashes for clang and gcc
    # infer which of the precalculation is relevant here
    compiler = "gcc"
    if "clang" in sys.version.lower():
        compiler = "clang"
    return compiler

def produce_proposal_tables_tarball(config_file, tables_path=None):
    """
    produce proposal tables tarball for all relevant flavors
    Note: the produced tarballs need to be placed in the
    NuRadioMC/proposal_tables/{version}/{compiler} directory on the desy cluster.

    Parameters
    ----------
    config_file: string
        one of the default PROPOSAL configurations ['InfIce','SouthPole', 'MooresBay', 'Greenland']
    tables_path: string
        output path for the generated tables
    """    

    if tables_path is None:
        proposal_func = ProposalFunctions(config_file=config_file)
        tables_path = proposal_func._ProposalFunctions__tables_path

    outdir = f"./{tables_path}/v{proposal.__version__}/{get_compiler()}"
    if not os.path.exists(outdir):
        os.makedirs(outdir)


    tarfile = config_file+".tar.gz"
    if os.path.isfile(os.path.join(outdir, tarfile)):
        logger.error(f"Output tarball {os.path.join(outdir, tarfile)} already exists.")
        raise IOError

    tables_path = os.path.join(outdir, config_file)
    produce_proposal_tables(config_file, tables_path)
        
    logger.warning("Producing gzipped tarball")

    shutil.make_archive(tables_path,
                'gztar', tables_path)

def download_proposal_tables(config_file, tables_path=None):
    """
    download precalculated proposal tables for all relevant flavors
    from the NuRadioMC data storage

    Parameters
    ----------
    config_file: string
        one of the default PROPOSAL configurations ['InfIce','SouthPole', 'MooresBay', 'Greenland']
    tables_path: string
        output path for the generated tables
    """

    if tables_path is None:
        proposal_func = ProposalFunctions(config_file=config_file, create_new=True)
        tables_path = proposal_func._ProposalFunctions__tables_path

    # does not exist yet -> download file
    import requests
    proposal_version = proposal.__version__
    URL = f'https://rnog-data.zeuthen.desy.de/proposal_tables/v{proposal_version}/{get_compiler()}/{config_file}.tar.gz'

    folder = tables_path #os.path.dirname(tables_path)
    if not os.path.exists(folder):
        os.makedirs(folder)
    logger.warning(
        "downloading pre-calculated proposal tables for {} from {}. This can take a while...".format(config_file, URL))
    r = requests.get(URL)
    if r.status_code != requests.codes.ok:
        logger.error("error in download of proposal tables")
        raise IOError

    with open(f"{tables_path}/{config_file}.tar.gz", "wb") as code:
        code.write(r.content)
    logger.warning("...download finished.")
    logger.warning(f"...unpacking archive to {tables_path}")
    shutil.unpack_archive(f"{tables_path}/{config_file}.tar.gz", tables_path)
    os.remove(f"{tables_path}/{config_file}.tar.gz")



if __name__ == "__main__":
    parser = argparse.ArgumentParser("NuRadioProposal tables I/O")
    parser.add_argument("option", choices=["create", "download"])
    parser.add_argument('config_file', help="one of the default configurations ['InfIce','SouthPole', 'MooresBay', 'Greenland'] or 'all'")
    parser.add_argument('-t', '--tables_path', default=None, help="target path for table creation/download")

    args = parser.parse_args()

    logger.warning(f"Your compiler type is {get_compiler()}")

    if args.option == "create":
        logger.warning(f"Creating proposal tables for {args.option}")
        if args.config_file == "all":
            cfgs = ['InfIce','SouthPole', 'MooresBay', 'Greenland']
            for cfg in cfgs:
                produce_proposal_tables_tarball(cfg, args.tables_path)
        else:
            produce_proposal_tables_tarball(args.config_file, args.tables_path)

    elif args.option == "download":
        logger.warning(f"Downloading proposal tables for {args.option}")
        download_proposal_tables(args.config_file, args.tables_path)
