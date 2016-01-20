#! /usr/bin/env python

import string
from subprocess import Popen
import os
import shutil
from Utils import now, append_time, force_write
from LocalDefinitions import TRILINOS_ROOT

def update_trilinos(logfile, build_type):


    trilinos_build_dir = TRILINOS_ROOT + build_type
    message = ""
    status = "Passed"
    
    logfile.write(append_time("Updating Trilinos with build type "+build_type+" ")+"\n") ; force_write(logfile)

    # ---- Clean Trilinos ----
    print "    cleaning..."
    logfile.write(append_time("\nCleaning Trilinos")+"\n") ; force_write(logfile)
    if os.path.exists(trilinos_build_dir+'/include'):
        shutil.rmtree(trilinos_build_dir+'/include')
    if os.path.exists(trilinos_build_dir+'/lib'):
        shutil.rmtree(trilinos_build_dir+'/lib')
    os.chdir(trilinos_build_dir)
    if os.name=='nt':
        command = ["do-clean.bat"]
    else:
        command = ["./do-clean"]
    p = Popen(command, stdout=logfile, stderr=logfile)
    return_code = p.wait()
    force_write(logfile)
    msg = "Trilinos " + build_type + " clean:  Passed\n"
    if return_code != 0:
        msg = "Trilinos " + build_type + " clean:  FAILED\n"
        status = "FAILED"
    logfile.write(msg) ; force_write(logfile)
    message += msg
    logfile.write(append_time("\nTrilinos clean complete")+"\n") ; force_write(logfile)
    
    # ---- update Trilinos ----
    print "    git pull..."
    logfile.write(append_time("\nUpdating Trilinos")+"\n") ; force_write(logfile)
    os.chdir(TRILINOS_ROOT)
    command = ["git", "pull"]
    p = Popen(command, stdout=logfile, stderr=logfile)
    return_code = p.wait()
    force_write(logfile)
    msg = "Trilinos " + build_type + " eg pull:  Passed\n"
    if return_code != 0:
       msg = "Trilinos " + build_type + " eg pull:  FAILED\n"
       status = "FAILED"
    logfile.write(msg) ; force_write(logfile)
    message += msg
    logfile.write(append_time("\nTrilinos update complete")+"\n") ; force_write(logfile)

    # ---- Run CMake for Trilinos ----
    print "    configuring..."
    logfile.write(append_time("\nRunning CMake for Trilinos")+"\n") ; force_write(logfile)
    os.chdir(trilinos_build_dir)
    if os.name=='nt':
        command = ["do-cmake.bat"]
    else:
        command = ["./do-cmake"]
    p = Popen(command, stdout=logfile, stderr=logfile)
    return_code = p.wait()
    force_write(logfile)
    msg = "Trilinos " + build_type + " CMake:  Passed\n"
    if return_code != 0:
        msg = "Trilinos " + build_type + " CMake:  FAILED\n"
        status = "FAILED"
    logfile.write(msg) ; force_write(logfile)
    message += msg
    logfile.write(append_time("\nCMake for Trilinos complete")+"\n") ; force_write(logfile)

    # ---- Build and install Trilinos ----
    print "    compiling..."
    logfile.write(append_time("\nBuilding Trilinos")+"\n") ; force_write(logfile)
    os.chdir(trilinos_build_dir)
    if os.name=='nt':
        command = ["do-make.bat"]
    else:
        command = ["./do-make"]
    p = Popen(command, stdout=logfile, stderr=logfile)
    return_code = p.wait()
    force_write(logfile)
    msg = "Trilinos " + build_type + " build:  Passed\n"
    if return_code != 0:
        msg = "Trilinos " + build_type + " build:  FAILED\n"
        status = "FAILED"
    logfile.write(msg) ; force_write(logfile)
    message += msg
    logfile.write(append_time("\nTrilinos build complete")+"\n") ; force_write(logfile)

    return (status, message)

if __name__ == "__main__":

    print "\n--UpdateTrilinos.py--\n"
    log_file = open("UpdateTrilinos.log", 'w')
    log_file.write("Trilinos update " + now() + "\n\n") ; force_write(log_file)
    status, message = update_trilinos(log_file, "Debug")
    print message
    status, message = update_trilinos(log_file, "Release")
    print message
    print "Log file written to UpdateTrilinos.log\n"
