#! /usr/bin/env python

import string
from datetime import date, datetime
from subprocess import Popen
import os
import stat
import glob
import smtplib
from email.mime.text import MIMEText
from Utils import now, append_time, force_write
from UpdateTrilinos import update_trilinos
from UpdateAndTestDICe import update_and_test_dice
from LocalDefinitions import MACHINE_NAME, OPERATING_SYSTEM, LOG_FILE_DIRECTORY, TRILINOS_ACTIVE_CONFIGS, DICE_ACTIVE_CONFIGS

if __name__ == "__main__":

    distribution_list = ['dzturne@sandia.gov']
    status = "Passed"
    message = "DICe nightly testing on " + MACHINE_NAME + " " + OPERATING_SYSTEM + " " + now() + "\n\n"
    logfile = open(LOG_FILE_DIRECTORY+'/NightlyTesting-'+str(date.today())+'.log', 'w')

    NUM_CONFIGS = len(TRILINOS_ACTIVE_CONFIGS) + len(DICE_ACTIVE_CONFIGS)
    print "Total configurations: " + str(NUM_CONFIGS)
    CURRENT_INDEX = 1
   
    for test_name in TRILINOS_ACTIVE_CONFIGS:
        STAT = float(CURRENT_INDEX)/float(NUM_CONFIGS) * 100.0
        print "[" + str(int(STAT)) +"%] Building Trilinos configuration: " + test_name
        result, msg = update_trilinos(logfile, test_name)
        if result != "Passed":
            status = result
        message += msg + "\n"
        CURRENT_INDEX = CURRENT_INDEX + 1

    for test_name in DICE_ACTIVE_CONFIGS:
        STAT = float(CURRENT_INDEX)/float(NUM_CONFIGS) * 100.0
        print "[" + str(int(STAT)) +"%] Building DICe configuration: " + test_name
        result, msg = update_and_test_dice(logfile, test_name)
        if result != "Passed":
            status = result
        message += msg + "\n"
        CURRENT_INDEX = CURRENT_INDEX + 1

    # ---- email the test results ----

    msg = MIMEText(message)
    msg['Subject'] = status + " DICe Nightly Tests"
    msg['From'] = MACHINE_NAME + ' Test Script <dzturne@sandia.gov>'
    msg['To'] = ""
    for address in distribution_list:
        msg['To'] += address + ", "

    s = smtplib.SMTP()
    s.connect()
    s.sendmail('dzturne@sandia.gov', distribution_list, msg.as_string())
    s.quit()
    
    logfile.close()
