#! /usr/bin/env python

import string
from subprocess import Popen, PIPE
import os
from Utils import now, append_time, force_write
from LocalDefinitions import MACHINE_NAME, DICE_ROOT

import sys

def package_dice(logfile, build_type):

    dice_build_dir = DICE_ROOT + build_type
    message = ""
    status = "Passed"

    logfile.write(append_time("Package DICe with build type " + build_type + " ")+"\n") ; force_write(logfile)

    os.chdir(dice_build_dir)
    if os.name=='nt':
        command = ["do-pkg.bat"]
    else:
        command = ["./do-pkg"]
    p = Popen(command, stdout=logfile, stderr=logfile, shell=True, executable='/bin/bash')
    return_code = p.wait()
    force_write(logfile)
    msg = "DICe " + build_type + " Package:  Passed\n"
    if return_code != 0:
        msg = "DICe " + build_type + " Package:  FAILED\n"
        status = "FAILED"
    logfile.write(msg) ; force_write(logfile)
    message += msg
    logfile.write(append_time("\nPackage for DICe " + build_type + " complete ")+"\n") ; force_write(logfile)

    return (status, message)
