#!/usr/bin/env python
from NuRadioMC.EvtGen.NuRadioProposal import ProposalFunctions
import proposal
import os
import shutil
import argparse
import logging

logger = logging.getLogger('ProposalTablesManager')

def produce_proposal_tables(config_file, tables_path=None):

    proposal_func = ProposalFunctions(config_file=config_file, tables_path=tables_path)

    for particle_code in [-15, -13, 13, 15]:
        logger.warning(f"producing tables for {config_file}, particle {particle_code}")
        proposal_func._ProposalFunctions__get_propagator(particle_code=particle_code)


def produce_proposal_tables_tarball(config_file, tables_path=None):
    if tables_path is None:
        proposal_func = ProposalFunctions(config_file=config_file)
        tables_path = proposal_func._ProposalFunctions__tables_path

    outdir = f"./{tables_path}/v{proposal.__version__}"
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
    if tables_path is None:
        proposal_func = ProposalFunctions(config_file=config_file)
        tables_path = proposal_func._ProposalFunctions__tables_path

    download_file = True
    if download_file:
        # does not exist yet -> download file
        import requests
        proposal_version = proposal.__version__
        URL = f'https://rnog-data.zeuthen.desy.de/proposal_tables/v{proposal_version}/{config_file}.tar.gz'

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
    parser.add_argument('config_file')
    parser.add_argument('-t', '--tables_path', default=None)

    args = parser.parse_args()

    if args.option == "create":
        if args.config_file == "all":
            cfgs = ['InfIce','SouthPole', 'MooresBay', 'Greenland']
            for cfg in cfgs:
                produce_proposal_tables_tarball(cfg, args.tables_path)
        else:
            produce_proposal_tables_tarball(args.config_file, args.tables_path)

    elif args.option == "download":
        download_proposal_tables(args.config_file, args.tables_path)
