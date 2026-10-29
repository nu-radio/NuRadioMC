#!/usr/bin/env python3
import argparse
from NuRadioMC.utilities._test_helpers import validate_nur_files, PARAMETER_RTOL

parser = argparse.ArgumentParser(description="Compare traces and channel/station parameters of two .nur files")
parser.add_argument('file1')
parser.add_argument('file2', help="reference file")
parser.add_argument('precision', type=int, nargs='?', default=7, help="decimals to which the traces have to agree")
parser.add_argument('--parameter-rtol', type=float, default=PARAMETER_RTOL,
                    help="relative tolerance for parameters which depend only on the signal")
parser.add_argument('--noise-parameter-rtol', type=float, default=None,
                    help="relative tolerance for NOISE_SENSITIVE_PARAMETERS. If not given (default, "
                    "e.g. for noiseless simulations), only their presence is checked")
args = parser.parse_args()

validate_nur_files(args.file1, args.file2, args.precision, args.parameter_rtol, args.noise_parameter_rtol)
