#! /usr/bin/env python

import string
from subprocess import Popen, PIPE
import os
from Utils import now, append_time, force_write
from LocalDefinitions import MACHINE_NAME, DICE_ROOT

import sys

def update_and_test_dice(logfile, build_type):

    dice_build_dir = DICE_ROOT + build_type
    message = ""
    status = "Passed"

    logfile.write(append_time("Updating and Testing DICe with build type " + build_type + " ")+"\n") ; force_write(logfile)

    # ---- clean DICe ----

    logfile.write(append_time("\nCleaning DICe " + build_type + " ")+"\n") ; force_write(logfile)
    os.chdir(dice_build_dir)
    command = ["./do-clean"]
    p = Popen(command, stdout=logfile, stderr=logfile)
    return_code = p.wait()
    force_write(logfile)
    msg = "DICe " + build_type + " clean:  Passed\n"
    if return_code != 0:
        msg = "DICe " + build_type + " clean:  FAILED\n"
        status = "FAILED"
    logfile.write(msg) ; force_write(logfile)
    message += msg
    logfile.write(append_time("\nDICe " + build_type + " clean complete ")+"\n") ; force_write(logfile)

    # ---- Update DICe ----

    logfile.write(append_time("\nUpdating DICe " + build_type + " ")+"\n") ; force_write(logfile)
    os.chdir(DICE_ROOT)
    command = ["git", "pull"]
    p = Popen(command, stdout=logfile, stderr=logfile)
    return_code = p.wait()
    force_write(logfile)
    msg = "DICe " + build_type + " git pull:  Passed\n"
    if return_code != 0:
        msg = "DICe " + build_type + " git pull:  FAILED\n"
        status = "FAILED"
    logfile.write(msg) ; force_write(logfile)
    message += msg
    logfile.write(append_time("\nDICe " + build_type + " update complete ")+"\n") ; force_write(logfile)

    # ---- Run CMake for DICe ----

    logfile.write(append_time("\nRunning CMake for DICe " + build_type + " ")+"\n") ; force_write(logfile)
    os.chdir(dice_build_dir)
    command = ["./do-cmake"]
    p = Popen(command, stdout=logfile, stderr=logfile, shell=True, executable='/bin/bash')
    return_code = p.wait()
    force_write(logfile)
    msg = "DICe " + build_type + " CMake:  Passed\n"
    if return_code != 0:
        msg = "DICe " + build_type + " CMake:  FAILED\n"
        status = "FAILED"
    logfile.write(msg) ; force_write(logfile)
    message += msg
    logfile.write(append_time("\nCMake for DICe " + build_type + " complete ")+"\n") ; force_write(logfile)

    # ---- build DICe ----

    logfile.write(append_time("\nBuilding DICe " + build_type + " ")+"\n") ; force_write(logfile)
    os.chdir(dice_build_dir)
    command = ["./do-make"]
    p = Popen(command, stdout=logfile, stderr=logfile)
    return_code = p.wait()
    force_write(logfile)
    msg = "DICe " + build_type + " build:  Passed\n"
    if return_code != 0:
        msg = "DICe " + build_type + " build:  FAILED\n"
        status = "FAILED"
    logfile.write(msg) ; force_write(logfile)
    message += msg
    logfile.write(append_time("\nDICe " + build_type + " build complete ")+"\n") ; force_write(logfile)

    # ---- run DICe tests ----

    logfile.write(append_time("Running DICe " + build_type + " tests ")+"\n") ; force_write(logfile)
    message += "\nDICe " + build_type + " CTest results:\n\n"
    os.chdir(dice_build_dir)
    command = ["./do-test"]
    p = Popen(command, stdout=PIPE, stderr=PIPE)
    return_code = p.wait()
    results = p.communicate()[0]
    message += results
    logfile.write(results) ; force_write(logfile)
    if return_code != 0:
        status = "FAILED"
    logfile.write(append_time("\nDICe " + build_type + " tests complete ")+"\n") ; force_write(logfile)

    return (status, message)

if __name__ == "__main__":

    print "\n--UpdateAndTestDICe.py--\n"
    log_file = open("UpdateAndTestDICe.log", 'w')
    log_file.write("DICe update and test " + now() + "\n\n") ; force_write(log_file)
    status, message = update_and_test_dice(log_file, "Release")
    print message
    print "Log file written to UpdateAndTestDICe.log\n"
