#! /usr/bin/env python

import string
from subprocess import Popen, PIPE, STDOUT
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
    print "    cleaning..."
    logfile.write(append_time("\nCleaning DICe " + build_type + " ")+"\n") ; force_write(logfile)
    os.chdir(dice_build_dir)
    if os.name=='nt':
        command = ["do-clean.bat"]
    else:
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
    print "    git pull..."
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
    print "    configuring..."
    logfile.write(append_time("\nRunning CMake for DICe " + build_type + " ")+"\n") ; force_write(logfile)
    
    os.chdir(dice_build_dir)
    if os.name=='nt':
        command = ["do-cmake.bat"]
    else:
        command = ["./do-cmake"]
    p = Popen(command, stdout=logfile, stderr=logfile)
    return_code = p.wait()
    force_write(logfile)
    # try to build 5 times if this is windows
    # this is to address the lock on manifest files
    # for exectuables
    if os.name=='nt':
        for i in range(0,5):
            print "      windows build attempt " + str(i)
            p = Popen(command, stdout=logfile, stderr=logfile)
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
    print "    compiling..."
    logfile.write(append_time("\nBuilding DICe " + build_type + " ")+"\n") ; force_write(logfile)
    os.chdir(dice_build_dir)
    if os.name=='nt':
        command = ["do-make.bat"]
    else:
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
    print "    testing..."
    logfile.write(append_time("Running DICe " + build_type + " tests ")+"\n") ; force_write(logfile)
    message += "\nDICe " + build_type + " CTest results:\n\n"
    os.chdir(dice_build_dir)
    if os.name=='nt':
        command = ["do-test.bat"]
    else:
        command = ["./do-test"]

    lines = []
    p = Popen(command, bufsize=1, stdin=open(os.devnull), stdout=PIPE, stderr=STDOUT)
    for line in iter(p.stdout.readline, ''):
        #print line,          # print to stdout immediately
        lines.append(line)   # capture for later
    p.stdout.close()
    return_code = p.wait()
    for line in lines:
        message += line #results
        logfile.write(line)
    force_write(logfile)
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
