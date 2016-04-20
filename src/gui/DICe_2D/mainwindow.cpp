// @HEADER
// ************************************************************************
//
//               Digital Image Correlation Engine (DICe)
//                 Copyright 2015 Sandia Corporation.
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact: Dan Turner (dzturne@sandia.gov)
//
// ************************************************************************
// @HEADER

#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>
#include <QFileInfo>
#include <QDirIterator>
#include <QDesktopServices>
#include <QTextStream>
#include <iostream>

#include <DICe_InputVars.h>
#include <simpleqtvtk.h>
#include <Teuchos_XMLParameterListHelpers.hpp>

#ifndef GITSHA1
  #define GITSHA1 "Version not available"
#endif

DICe::gui::Input_Vars * DICe::gui::Input_Vars::input_vars_ptr_ = NULL;

void stringToUpper(std::string & string){
    std::transform(string.begin(), string.end(),string.begin(), ::toupper);
}

// free function to tokenize a file line
std::vector<std::string> tokenize_line(std::fstream &dataFile,
                                       const std::string & delim,
                                       const bool capitalize){
    static int MAX_CHARS_PER_LINE = 512;
    static int MAX_TOKENS_PER_LINE = 20;
    static std::string DELIMITER = delim;
    std::vector<std::string> tokens(MAX_TOKENS_PER_LINE);

    // read an entire line into memory
    char * buf = new char[MAX_CHARS_PER_LINE];
    dataFile.getline(buf, MAX_CHARS_PER_LINE);

    // parse the line into blank-delimited tokens
    int n = 0; // a for-loop index

    // parse the line
    char * token;
    token = strtok(buf, DELIMITER.c_str());
    tokens[0] = token ? token : ""; // first token
    if(capitalize)
        stringToUpper(tokens[0]);
    bool first_char_is_pound = tokens[0].find("#") == 0;
    if (tokens[0] != "" && tokens[0]!="#" && !first_char_is_pound){ // zero if line is blank or starts with a comment char
        for (n = 1; n < MAX_TOKENS_PER_LINE; n++){
            token = strtok(0, DELIMITER.c_str());
            tokens[n] = token ? token : ""; // subsequent tokens
            if (tokens[n]=="") break; // no more tokens
            if(capitalize)
                stringToUpper(tokens[n]);
        }
    }
    tokens.resize(n);
    delete [] buf;
    return tokens;
}

QList<QList<QPoint> > readShapes(std::fstream &dataFile){
    QList<QList<QPoint> > shapes;
    while(!dataFile.eof()){
      std::vector<std::string> shape_tokens = tokenize_line(dataFile);
      if(shape_tokens.size()==0)continue;
      if(shape_tokens[0]=="END") break;
      else if(shape_tokens[1]=="POLYGON"){
          QList<QPoint> pts;
          while(!dataFile.eof()){
            std::vector<std::string> tokens = tokenize_line(dataFile);
            if(tokens.size()==0) continue; // comment or blank line
            if(tokens[0]=="END") break;
            // read the vertices
            while(!dataFile.eof()){
              std::vector<std::string> vertex_tokens = tokenize_line(dataFile);
              if(vertex_tokens.size()==0)continue;
              if(vertex_tokens[0]=="END") break;
              QPoint pt(atoi(vertex_tokens[0].c_str()),atoi(vertex_tokens[1].c_str()));
              pts.append(pt);
            }
          }
          shapes.append(pts);
      }
    }
    return shapes;
}

MainWindow::MainWindow(QWidget *parent) :
QMainWindow(parent),
ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    this->resetDefaults();

    ui->simpleQtVTKWidget->disableResultsTab();

    // reset the default working directory
    ui->workingDirLabel->setText(".");
    DICe::gui::Input_Vars::instance()->set_working_dir(QString("."));

    // set up the console output
    qout = new QDebugStream(std::cout, ui->consoleEdit);

    // set up the process that will run DICe
    diceProcess = new QProcess(this);
    checkValidDiceProcess = new QProcess(this);
    diceProcess->setProcessChannelMode(QProcess::MergedChannels);
    connect(diceProcess, SIGNAL(readyReadStandardOutput()), this, SLOT(readOutput()));
    connect(diceProcess, SIGNAL(finished(int)), this, SLOT(execWrapUp(int)));
    connect(checkValidDiceProcess, SIGNAL(readyReadStandardOutput()), this, SLOT(readValidationOutput()));

    // set up the export files action
    connect(ui->actionExport_input_files, SIGNAL(triggered()), this, SLOT(exportInputFiles()));

    // set up the web page launcher
    connect(ui->actionAbout_DICe, SIGNAL(triggered()), this, SLOT(launchDICePage()));

    // set up the load working dir action
    connect(ui->actionLoad_working_dir, SIGNAL(triggered()), this, SLOT(userLoadWorkingDir()));

    // set the the change exec location action
    connect(ui->actionSet_backend_exec, SIGNAL(triggered()), this, SLOT(setExec()));

    // flag when errors occur with checkValidDiceProcess or diceProcess
    connect(checkValidDiceProcess, SIGNAL(error(QProcess::ProcessError)),this,SLOT(setError()));
    connect(checkValidDiceProcess, SIGNAL(started()),this,SLOT(resetError()));
    connect(diceProcess, SIGNAL(error(QProcess::ProcessError)),this,SLOT(setError()));
    connect(diceProcess, SIGNAL(started()),this,SLOT(resetError()));

    QString configPath = QDir::homePath();
#ifdef WIN32
    configPath +=  "\\.dice";
#else
    configPath +=  "/.dice";
#endif
    QFileInfo configFile(configPath);
    // check if file exists and if yes: Is it really a file and not a directory?
    // if the file does not exist, create it with the default locations
    if (!configFile.exists() || !configFile.isFile()) {
        QFile saveConfig(configPath);
#ifdef WIN32
        execPath =  "C:\\Program Files (x86)\\Digital Image Correlation Engine\\dice.exe";
#else
        execPath =  "/Applications/DICe.app/Contents/MacOS/dice";
#endif
        if (saveConfig.open(QIODevice::ReadWrite)) {
            QTextStream stream(&saveConfig);
            stream << execPath << endl;
            saveConfig.close();
        }
        else{
            QMessageBox msgBox;
            msgBox.setText("Unable to save DICe configuration file to the home directory.");
            msgBox.exec();
            exit(1);
        }
    }
    else{ // read the file and get the executable location
        QFile savedConfig(configPath);
        if (savedConfig.open(QIODevice::ReadOnly)) {
            execPath = savedConfig.readLine();
            execPath = execPath.simplified(); // remove white space
            savedConfig.close();
        }
        else{
            QMessageBox msgBox;
            msgBox.setText("Unable to read DICe configuration file (/home/.dice)");
            msgBox.exec();
            exit(1);
        }
    }
    // test the executable
    if(testForValidExec())
        exit(1);

    std::cout << "Using DICe from " << execPath.toStdString() << std::endl;
    ui->progressBar->setValue(0);

    // if the current working directory has results/input files, load them
    if(loadExistingFilesCheck(DICe::gui::Input_Vars::instance()->get_working_dir()))
        this->loadWorkingDir(DICe::gui::Input_Vars::instance()->get_working_dir(),false);

}

void MainWindow::resetDefaults(){
    // set up the analysis defaults
    ui->translationShapeCheck->setChecked(true);

    // add the initialization methods
    ui->initMethodCombo->addItem("USE_FIELD_VALUES");
    ui->initMethodCombo->addItem("USE_NEIGHBOR_VALUES");
    ui->initMethodCombo->addItem("USE_NEIGHBOR_VALUES_FIRST_STEP");

    // add the optimization methods
    ui->optMethodCombo->addItem("GRADIENT_BASED");
    ui->optMethodCombo->addItem("SIMPLEX");

    // add the interpolation methods
    ui->interpMethodCombo->addItem("KEYS_FOURTH");
    ui->interpMethodCombo->addItem("BICUBIC");
    ui->interpMethodCombo->addItem("BILINEAR");

    // add the strain methods
    ui->strainCombo->addItem("VIRTUAL_STRAIN_GAUGE");
    ui->strainCheck->setChecked(true);

    // set up the step size and subset size
    ui->subsetSize->setMinimum(10);
    ui->subsetSize->setMaximum(50);
    ui->subsetSize->setValue(25);
    ui->stepSize->setMinimum(1);
    ui->stepSize->setMaximum(100);
    ui->stepSize->setValue(15);
    ui->strainWindowSpin->setMinimum(ui->stepSize->value()*2);
    ui->strainWindowSpin->setMaximum(ui->stepSize->value()*6);
    ui->strainWindowSpin->setValue(ui->stepSize->value()*2);
    ui->strainWindowSpin->setSingleStep(ui->stepSize->value());

    // reset the progress bar
    ui->progressBar->setValue(0);

    // set the default output fields
    ui->xCheck->setChecked(true);
    ui->yCheck->setChecked(true);
    ui->dispXCheck->setChecked(true);
    ui->dispYCheck->setChecked(true);
    ui->sigmaCheck->setChecked(true);
    ui->gammaCheck->setChecked(true);
    ui->betaCheck->setChecked(true);
    ui->statusCheck->setChecked(true);
}


MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_refFileButton_clicked()
{
    // open file dialog box to select reference file
    QFileInfo refFileInfo = QFileDialog::getOpenFileName(this,
      tr("Select reference file"), ".",
      tr("Tagged Image File Format (*.tiff *.tif);;Portable Network Graphics (*.png);;Joint Photographic Experts Group (*.jpg *.jpeg)"));
    
    if(refFileInfo.fileName()=="") return;
    
    // display the name of the file in the reference file box
    if(ui->simpleQtVTKWidget->readImageFile(refFileInfo.filePath().toStdString(),true)) return;

    // set the reference image in the Input_Vars singleton
    DICe::gui::Input_Vars::instance()->set_ref_file_info(refFileInfo);

    ui->simpleQtVTKWidget->changeInteractionMode(0);
    ui->simpleQtVTKWidget->disableResultsTab();

    // if the deformed images are populated then activate the write and run buttons
    if(DICe::gui::Input_Vars::instance()->has_def_files()){
        ui->actionExport_input_files->setEnabled(true);
        ui->runButton->setEnabled(true);
    }
}

void MainWindow::on_defFileButton_clicked()
{
    // get a pointer to the file name list in the Input_Vars singleton
    QStringList * list = DICe::gui::Input_Vars::instance()->get_def_file_list();
    list->clear();
    
    // open a file dialog box to select the deformed images
    QFileDialog defDialog(this);
    defDialog.setFileMode(QFileDialog::ExistingFiles);
    QStringList defFileNames = defDialog.getOpenFileNames(this,
      tr("Select reference file"), ".",
      tr("Tagged Image File Format (*.tiff *.tif);;Portable Network Graphics (*.png);;Joint Photographic Experts Group (*.jpg *.jpeg)"));
    
    if(defFileNames.size()==0) return;

    // clear the widget list
    ui->defListWidget->clear();
    // add the names of the files to a list widget
    for(QStringList::iterator it=defFileNames.begin(); it!=defFileNames.end();++it){
        list->append(*it);
        QFileInfo currentFile = *it;
        QString shortDefFileName = currentFile.fileName();
        ui->defListWidget->addItem(shortDefFileName);
    }

    // update the picture in the def preview
    ui->defListWidget->setCurrentRow(0);
    on_defListWidget_itemClicked(ui->defListWidget->currentItem());

    // if the reference image is populated then activate the write and run buttons
    if(DICe::gui::Input_Vars::instance()->has_ref_file()){
        ui->actionExport_input_files->setEnabled(true);
        ui->runButton->setEnabled(true);
    }
}

void MainWindow::on_defListWidget_itemClicked(QListWidgetItem *item)
{
    // get the index of the item that was clicked
    int row = item->listWidget()->row(item);
    // determine the file name for the clicked item
    QStringList * list = DICe::gui::Input_Vars::instance()->get_def_file_list();
    QFileInfo defFile = list->at(row);
    
    // display the file in the small view below the list
    ui->defFileLabel->setText(defFile.filePath());
    QPixmap pix(defFile.filePath());
    int w = ui->defImageShow->width();
    int h = ui->defImageShow->height();
    ui->defImageShow->setPixmap(pix.scaled(w,h,Qt::KeepAspectRatio));
}

bool MainWindow::loadExistingFilesCheck(const QString & dir){
    // if there are existing files in the dir ask if they should be over-written:
    QString inputFile = dir;
#ifdef WIN32
    inputFile +=  "\\input.xml";
#else
    inputFile +=  "/input.xml";
#endif
    QFileInfo checkFile(inputFile);
    // check if file exists and if yes: Is it really a file and not a directory?
    if (checkFile.exists() && checkFile.isFile()) {
        QMessageBox msgBox;
        msgBox.setText("Existing input/results files found in the working directory\n"
                       "Load these existing files? (otherwise they will be overwritten)");
        msgBox.setStandardButtons(QMessageBox::Yes);
        msgBox.addButton(QMessageBox::No);
        msgBox.setDefaultButton(QMessageBox::Yes);
        if(msgBox.exec() == QMessageBox::No){
            return false;
        }
        else
            return true;
    }
    return false;
}

void MainWindow::on_workingDirButton_clicked()
{
    // open a file dialog and get the directory
    QString dir = QFileDialog::getExistingDirectory(this,tr("Open Directory"),".",
                      QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);

    if(dir=="") return;    

    // set the reference image in the Input_Vars singleton
    DICe::gui::Input_Vars::instance()->set_working_dir(dir);

    // display the name of the file in the reference file box
    ui->workingDirLabel->setText(dir);

    if(loadExistingFilesCheck(dir))
        this->loadWorkingDir(dir,false);
}

void MainWindow::launchDICePage(){
    QMessageBox msgBox;
    QString version = QString::fromStdString(GITSHA1);
    msgBox.setText("Digital Image Correlation Engine (DICe)\n\nVersion: " + version
                   +"\n\nCopyright 2015 Sandia Corporation");
    QAbstractButton *myNoButton = msgBox.addButton(trUtf8("Close"), QMessageBox::NoRole);
    QAbstractButton *myYesButton = msgBox.addButton(trUtf8("Github"), QMessageBox::YesRole);
    msgBox.exec();
    if(msgBox.clickedButton() == myYesButton){
        QDesktopServices::openUrl(QUrl("https://github.com/dicengine/dice", QUrl::TolerantMode));
        return;
    }
    else if(msgBox.clickedButton() == myNoButton){
        return;
    }
}

bool MainWindow::testForValidExec(){
    QStringList args;
    args << "--version";
    checkValidDiceProcess->start(execPath,args);
    checkValidDiceProcess->waitForFinished();
    checkValidDiceProcess->close();
    if(checkValidDiceProcess->exitStatus()||checkValidDiceProcess->exitCode()||execError){
        QMessageBox msgBox;
        msgBox.setText("Invalid DICe backend executable:\n" + execPath
                       + "\n\nThe executable path in home/.dice may be invalid,"
                         "\nor an invalid executable was selected by the user.");
        msgBox.exec();
        return true;
    }
    return false;
}

int MainWindow::setExec(){
    QString configPath = QDir::homePath();
#ifdef WIN32
    configPath +=  "\\.dice";
#else
    configPath +=  "/.dice";
#endif
    QFileInfo execFileInfo = QFileDialog::getOpenFileName(this,
      tr("Select the DICe backend executable to use"), ".",
      tr("Executable (dice.exe dice)"));
    if(execFileInfo.fileName()==""){
        return 1;
    }
    execPath = execFileInfo.filePath();
    // try the non-standard location:
    if(testForValidExec())
        return 1;

    // save the selection to the config file:
    QFile saveConfig(configPath);
    if (saveConfig.open(QIODevice::ReadWrite)) {
        QTextStream stream(&saveConfig);
        stream << execPath << endl;
        saveConfig.close();
    }
    else{
        QMessageBox msgBox;
        msgBox.setText("Error saving DICe configuration file.");
        msgBox.exec();
        return 1;
    }
    return 0;
}

void MainWindow::userLoadWorkingDir(){
    QMessageBox msgBox;
    msgBox.setText("Input and results files will be imported\n"
                   "from the selected directory\n"
                   "to the current working directory.\n\n"
                   "Warning: Importing will clear existing data!\n\nContinue?");
    msgBox.setStandardButtons(QMessageBox::Yes);
    msgBox.addButton(QMessageBox::No);
    msgBox.setDefaultButton(QMessageBox::Yes);
    if(msgBox.exec() == QMessageBox::No){
        return;
    }

    // open a file dialog to select the directory:
    QString dir = QFileDialog::getExistingDirectory(this,tr("Open Directory"),".",
                      QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);

    if(dir=="") return;
    this->loadWorkingDir(dir,true);
}

void MainWindow::loadWorkingDir(const QString & dir, const bool overwrite){

    // if there are existing files in the dir ask if they should be over-written:
    QString inputFile = dir;
    QString paramsFile = dir;
    QString subsetFile = dir;
    QString resultsDir = dir;
    QString currentWorkingDir = DICe::gui::Input_Vars::instance()->get_working_dir();
    QString currentResultsDir = currentWorkingDir;
    bool foundInput = false;
    bool foundParams = false;
    bool foundSubset = false;
    bool foundResults = false;
#ifdef WIN32
    inputFile +=  "\\input.xml";
    paramsFile +=  "\\params.xml";
    subsetFile +=  "\\subset_defs.txt";
    resultsDir += "\\results";
    currentWorkingDir += "\\";
    currentResultsDir += "\\results\\";
#else
    inputFile +=  "/input.xml";
    paramsFile +=  "/params.xml";
    subsetFile +=  "/subset_defs.txt";
    resultsDir += "/results";
    currentWorkingDir += "/";
    currentResultsDir += "/results/";
#endif
    QFileInfo checkFile(inputFile);
    // check if file exists and if yes: Is it really a file and not a directory?
    if (checkFile.exists() && checkFile.isFile()) {
        foundInput = true;
    }
    checkFile = paramsFile;
    // check if file exists and if yes: Is it really a file and not a directory?
    if (checkFile.exists() && checkFile.isFile()) {
        foundParams = true;
    }
    checkFile = subsetFile;
    // check if file exists and if yes: Is it really a file and not a directory?
    if (checkFile.exists() && checkFile.isFile()) {
        foundSubset = true;
    }
    //std::cout << " found input: " << foundInput << " params " << foundParams << " subset " << foundSubset << std::endl;

    if(!foundInput){
        if(overwrite){
            QMessageBox msgBox;
            msgBox.setText("Unable to inport working directory because\nthe input.xml file is missing.");
            msgBox.exec();
        }
        return;
    }
    checkFile = resultsDir;
    // check if dir exists and if yes: Is it really a dir and not a file?
    if (checkFile.exists() && !checkFile.isFile()) {
        foundResults = true;
    }

    Teuchos::RCP<Teuchos::ParameterList> stringParams = Teuchos::rcp( new Teuchos::ParameterList() );
    Teuchos::Ptr<Teuchos::ParameterList> stringParamsPtr(stringParams.get());
    Teuchos::updateParametersFromXmlFile(inputFile.toStdString(), stringParamsPtr);
    bool inputValid = true;
    if(!stringParams->isParameter("step_size")){
        std::cout << "Error: input.xml missing step_size parameter" << std::endl;
        inputValid = false;
    }
    if(!stringParams->isParameter("subset_size")){
        std::cout << "Error: input.xml missing subset_size parameter" << std::endl;
        inputValid = false;
    }
    if(!stringParams->isParameter("reference_image")){
        std::cout << "Error: input.xml missing reference_image parameter" << std::endl;
        inputValid = false;
    }
    if(!stringParams->isParameter("deformed_images")){
        std::cout << "Error: input.xml missing deformed_images parameter" << std::endl;
        inputValid = false;
    }
    if(stringParamsPtr==Teuchos::null||!inputValid){
        QMessageBox msgBox;
        msgBox.setText("Error: reading input.xml failed!");
        msgBox.exec();
        return;
    }

    // ensure that if the input requests a subset file that it exists
    const bool hasSubsetFile = stringParams->isParameter("subset_file");
    if(hasSubsetFile&&(!foundSubset)){
        QMessageBox msgBox;
        msgBox.setText("Error: subset file required, but could not find subset_defs.txt file!");
        msgBox.exec();
        return;
    }

    // check the deformed images:
    Teuchos::ParameterList def_image_sublist = stringParams->sublist("deformed_images");
    if(def_image_sublist.numParams()<=0){
        QMessageBox msgBox;
        msgBox.setText("Error: list of deformed images is of size 0");
        msgBox.exec();
        return;
    }
    for(Teuchos::ParameterList::ConstIterator it=def_image_sublist.begin();it!=def_image_sublist.end();++it){
      const bool active = def_image_sublist.get<bool>(it->first);
      if(active){
          QFileInfo currentFile = QString::fromStdString(it->first);
          if(!(currentFile.exists() && currentFile.isFile())){
              QMessageBox msgBox;
              msgBox.setText("Invalid deformed image: " + currentFile.filePath());
              msgBox.exec();
              return;
          }
      }
    }
    // remove existing input file if it exists:
    QFileInfo inputInfo = inputFile;
    QFileInfo existingInput = currentWorkingDir+QString::fromStdString("input.xml");
    if(existingInput.exists()&&overwrite){
        std::cout << "Removing input.xml from current working directory" << std::endl;
        QFile::remove(existingInput.filePath());
    }
    bool inputCopySuccess = true;
    if(overwrite)
        inputCopySuccess = QFile::copy(inputInfo.filePath(),currentWorkingDir+inputInfo.fileName());
    if(!inputCopySuccess){
        QMessageBox msgBox;
        msgBox.setText("Error: could not copy input.xml to working dir.");
        msgBox.exec();
        return;
    }

    // read the subset file:
    QList<QList<QPoint> > boundaryShapes;
    QList<QList<QPoint> > excludedShapes;
    if(hasSubsetFile){
        QFileInfo subsetInfo = subsetFile;
        bool hasROIDef = false;
        bool subsetError = false;
        std::fstream dataFile(subsetFile.toStdString().c_str(), std::ios_base::in);
        while(!dataFile.eof()){
            std::vector<std::string> tokens = tokenize_line(dataFile);
            if(tokens.size()==0)continue;
            if(tokens.size()<2){
                subsetError=true;
                break;
            }
            if(tokens[0]=="BEGIN"){
                if(tokens[1]=="REGION_OF_INTEREST"){
                    hasROIDef = true;
                    // now read boundary or excluded regions
                    while(!dataFile.eof()){
                        std::vector<std::string> block_tokens = tokenize_line(dataFile);
                        if(block_tokens.size()==0)continue;
                        else if(block_tokens[0]=="END") break;
                        else if(block_tokens[0]=="BEGIN"){
                            if(block_tokens.size()<2){
                                std::cout << "invalid block token size " << block_tokens.size() << std::endl;
                                dataFile.close();
                                return;
                            }
                            if(block_tokens[1]=="BOUNDARY"){
                                boundaryShapes = readShapes(dataFile);
                                // TODO push the shapes back on the vtkwidget storage
                            }
                            else if(block_tokens[1]=="EXCLUDED"){
                                excludedShapes = readShapes(dataFile);
                            }
                            else{
                                std::cout << "invalid block token " << block_tokens[1] << std::endl;
                                dataFile.close();
                                return;
                            }
                        } // begin
                        else{
                            std::cout << "invalid block token " << block_tokens[0] << std::endl;
                            dataFile.close();
                            return;
                        }
                    }
                }
            }
        }
        dataFile.close();
        if(!hasROIDef||subsetError){
            QMessageBox msgBox;
            msgBox.setText("Invalid subset_defs.txt file: ");
            msgBox.exec();
            return;
        }
        QFileInfo existingSubsetInfo = currentWorkingDir+QString::fromStdString("subset_defs.txt");
        if(existingSubsetInfo.exists() && overwrite){
            std::cout << "Removing subset_defs.txt from current working directory" << std::endl;
            QFile::remove(existingSubsetInfo.filePath());
        }
        bool subsetCopySuccess = true;
        if(overwrite)
            subsetCopySuccess = QFile::copy(subsetInfo.filePath(),currentWorkingDir+subsetInfo.fileName());
        if(!subsetCopySuccess){
            QMessageBox msgBox;
            msgBox.setText("Error: could not copy subset_defs.txt to working dir.");
            msgBox.exec();
        }
    }

    // load the reference image:
    // display the name of the file in the reference file box
    QFileInfo refFileInfo = QString::fromStdString(stringParams->get<std::string>("reference_image",""));
    if(ui->simpleQtVTKWidget->readImageFile(refFileInfo.filePath().toStdString(),true)){
        QMessageBox msgBox;
        msgBox.setText("Error: could not read the reference image: " + refFileInfo.filePath());
        msgBox.exec();
        return;
    }

    // set the reference image in the Input_Vars singleton
    DICe::gui::Input_Vars::instance()->set_ref_file_info(refFileInfo);

    // Import the deformed images:
    QStringList * list = DICe::gui::Input_Vars::instance()->get_def_file_list();
    list->clear();
    ui->defListWidget->clear();
    for(Teuchos::ParameterList::ConstIterator it=def_image_sublist.begin();it!=def_image_sublist.end();++it){
      const bool active = def_image_sublist.get<bool>(it->first);
      if(active){
          list->append(QString::fromStdString(it->first));
          QFileInfo currentFile = QString::fromStdString(it->first);
          QString shortDefFileName = currentFile.fileName();
          ui->defListWidget->addItem(shortDefFileName);
      }
    }
    // update the picture in the def preview
    ui->defListWidget->setCurrentRow(0);
    on_defListWidget_itemClicked(ui->defListWidget->currentItem());

    // import the shapes
    if(hasSubsetFile){
        // import shapes (boundaryShapes,excludedShapes);
        ui->simpleQtVTKWidget->importVertices(boundaryShapes,excludedShapes);
    }

    // import information:
    const int subsetSize = stringParams->get<int>("subset_size");
    ui->subsetSize->setValue(subsetSize);
    const int stepSize = stringParams->get<int>("step_size");
    ui->stepSize->setValue(stepSize);

    // import the parameters if they exist:
    if(foundParams){
        QFileInfo paramsInfo = paramsFile;
        Teuchos::RCP<Teuchos::ParameterList> corrParams = Teuchos::rcp( new Teuchos::ParameterList() );
        Teuchos::Ptr<Teuchos::ParameterList> corrParamsPtr(corrParams.get());
        Teuchos::updateParametersFromXmlFile(paramsFile.toStdString(), corrParamsPtr);
        if(corrParams->isParameter("interpolation_method")){
            std::string value = corrParams->get<std::string>("interpolation_method");
            stringToUpper(value);
            int id = ui->interpMethodCombo->findText(QString::fromStdString(value));
            if(id!=-1)
                ui->interpMethodCombo->setCurrentIndex(id);
        }
        if(corrParams->isParameter("optimization_method")){
            std::string value = corrParams->get<std::string>("optimization_method");
            stringToUpper(value);
            int id = ui->optMethodCombo->findText(QString::fromStdString(value));
            if(id!=-1)
                ui->optMethodCombo->setCurrentIndex(id);
        }
        if(corrParams->isParameter("initialization_method")){
            std::string value = corrParams->get<std::string>("initialization_method");
            stringToUpper(value);
            int id = ui->initMethodCombo->findText(QString::fromStdString(value));
            if(id!=-1)
                ui->initMethodCombo->setCurrentIndex(id);
        }
        ui->translationShapeCheck->setChecked(corrParams->get<bool>("enable_translation",false));
        ui->normalShapeCheck->setChecked(corrParams->get<bool>("enable_normal_strain",false));
        ui->rotationShapeCheck->setChecked(corrParams->get<bool>("enable_rotation",false));
        ui->shearShapeCheck->setChecked(corrParams->get<bool>("enable_shear_strain",false));

        // read the output spec
        if(corrParams->isParameter("output_spec")){
            ui->rotationCheck->setChecked(false);
            ui->normalXCheck->setChecked(false);
            ui->normalYCheck->setChecked(false);
            ui->shearCheck->setChecked(false);
            ui->sigmaCheck->setChecked(false);
            ui->gammaCheck->setChecked(false);
            ui->betaCheck->setChecked(false);
            ui->statusCheck->setChecked(false);
            ui->numActiveCheck->setChecked(false);
            ui->contrastCheck->setChecked(false);
            ui->noiseCheck->setChecked(false);
            Teuchos::ParameterList output_field_sublist = corrParams->sublist("output_spec");
            for(Teuchos::ParameterList::ConstIterator it=output_field_sublist.begin();it!=output_field_sublist.end();++it){
                const bool active = output_field_sublist.get<bool>(it->first);
                if(active){
                    std::string field = it->first;
                    stringToUpper(field);

                    if(field=="ROTATION_Z")
                        ui->rotationCheck->setChecked(true);
                    else if(field=="NORMAL_STRAIN_X")
                        ui->normalXCheck->setChecked(true);
                    else if(field=="NORMAL_STRAIN_Y")
                        ui->normalYCheck->setChecked(true);
                    else if(field=="SHEAR_STRAIN_XY")
                        ui->shearCheck->setChecked(true);
                    else if(field=="SIGMA")
                        ui->sigmaCheck->setChecked(true);
                    else if(field=="GAMMA")
                        ui->gammaCheck->setChecked(true);
                    else if(field=="BETA")
                        ui->betaCheck->setChecked(true);
                    else if(field=="STATUS_FLAG")
                        ui->statusCheck->setChecked(true);
                    else if(field=="NUM_ACTIVE_PIXELS")
                        ui->numActiveCheck->setChecked(true);
                    else if(field=="CONTRAST_LEVEL")
                        ui->contrastCheck->setChecked(true);
                    else if(field=="NOISE_LEVEL")
                        ui->noiseCheck->setChecked(true);
                }
            }
        }
        QFileInfo existingParams = currentWorkingDir+QString::fromStdString("params.xml");
        if(existingParams.exists()&&overwrite){
            std::cout << "Removing params.xml from current working directory" << std::endl;
            QFile::remove(existingParams.filePath());
        }
        bool paramsCopySuccess = true;
        if(overwrite)
            paramsCopySuccess = QFile::copy(paramsInfo.filePath(),currentWorkingDir+paramsInfo.fileName());
        if(!paramsCopySuccess){
            QMessageBox msgBox;
            msgBox.setText("Error: could not copy params.xml to working dir.");
            msgBox.exec();
        }
    }

    // activate the run/write buttons
    ui->actionExport_input_files->setEnabled(true);
    ui->runButton->setEnabled(true);

    // copy results files if they exist:
    if(foundResults){
        if(overwrite){
            // copy results files if they exist:
            // create the working directory
            if(!QDir(currentResultsDir).exists()){
                QDir().mkdir(currentResultsDir);
            }

            std::cout << "Clearing existing results directory " << std::endl;
            QStringList currentResFiles;
            QDirIterator  currentIt(currentResultsDir, QStringList() << "*.txt",
                                    QDir::Files,QDirIterator::Subdirectories);
            while (currentIt.hasNext()){
                std::cout << "Removing file " << currentIt.next().toStdString() << std::endl;
                QDir cDir;
                cDir.remove(currentIt.filePath());
            }

            // copy files to this working directory
            QStringList resFiles;
            QDirIterator it(resultsDir, QStringList() << "*.txt",
                            QDir::Files,QDirIterator::Subdirectories);
            while (it.hasNext()){
                QString fromFile = it.next();
                QString toFile = currentResultsDir+it.fileName();
                std::cout << "Copying file: " << fromFile.toStdString() << std::endl;
                std::cout << "          to: " << toFile.toStdString() << std::endl;
                bool success = QFile::copy(it.filePath(),currentResultsDir+it.fileName());
                if(!success){
                    QMessageBox msgBox;
                    msgBox.setText("Error: could not copy results files to working dir.");
                    msgBox.exec();
                    return;
                }
            }
        } // end overwrite
        // load the results viewer
        prepResultsViewer();
    }
}

void MainWindow::exportInputFiles(){
    // save off the existing working directory
    QString existingDir = DICe::gui::Input_Vars::instance()->get_working_dir();

    // open a file dialog to select the directory:
    QString dir = QFileDialog::getExistingDirectory(this,tr("Open Directory"),".",
                      QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);

    if(dir=="") return;

    // if there are existing files in the dir ask if they should be over-written:
    QString inputFile = dir;
#ifdef WIN32
    inputFile +=  "\\input.xml";
#else
    inputFile += "/input.xml";
#endif
    QFileInfo checkFile(inputFile);
    // check if file exists and if yes: Is it really a file and no directory?
    if (checkFile.exists() && checkFile.isFile()) {
        QMessageBox msgBox;
        msgBox.setText("Found existing input files in this directory.\nOverwrite?");
        msgBox.setStandardButtons(QMessageBox::Yes);
        msgBox.addButton(QMessageBox::No);
        msgBox.setDefaultButton(QMessageBox::Yes);
        if(msgBox.exec() == QMessageBox::No){
            return;
        }
    }

    // set the reference image in the Input_Vars singleton
    DICe::gui::Input_Vars::instance()->set_working_dir(dir);

    // write the files
    writeInputFiles();

    // set the reference image in the Input_Vars singleton
    DICe::gui::Input_Vars::instance()->set_working_dir(existingDir);
}

void MainWindow::writeInputFiles(){
    std::cout << "Writing input files..." << std::endl << std::endl;
    // set the parameter values:
    DICe::gui::Input_Vars::instance()->set_subset_size(ui->subsetSize->value());
    DICe::gui::Input_Vars::instance()->set_step_size(ui->stepSize->value());
    DICe::gui::Input_Vars::instance()->set_initialization_method(ui->initMethodCombo->currentText().toStdString());
    DICe::gui::Input_Vars::instance()->set_optimization_method(ui->optMethodCombo->currentText().toStdString());
    DICe::gui::Input_Vars::instance()->set_interpolation_method(ui->interpMethodCombo->currentText().toStdString());
    DICe::gui::Input_Vars::instance()->set_enable_translation(ui->translationShapeCheck->isChecked());
    DICe::gui::Input_Vars::instance()->set_enable_rotation(ui->rotationShapeCheck->isChecked());
    DICe::gui::Input_Vars::instance()->set_enable_normal_strain(ui->normalShapeCheck->isChecked());
    DICe::gui::Input_Vars::instance()->set_enable_shear_strain(ui->shearShapeCheck->isChecked());
    std::cout << " strain check " << ui->strainCheck->isChecked() << std::endl;
    std::cout << " strain combo " << ui->strainCombo->currentText().toStdString() << std::endl;
    if(ui->strainCheck->isChecked()&&ui->strainCombo->currentText()=="VIRTUAL_STRAIN_GAUGE"){
        DICe::gui::Input_Vars::instance()->set_enable_vsg(true);
    }
    else
        DICe::gui::Input_Vars::instance()->set_enable_vsg(false);
    DICe::gui::Input_Vars::instance()->set_strain_window_size(ui->strainWindowSpin->value());
    // output fields
    std::vector<std::string> output_fields;
    if(ui->xCheck->isChecked()) output_fields.push_back("COORDINATE_X");
    if(ui->yCheck->isChecked()) output_fields.push_back("COORDINATE_Y");
    if(ui->dispXCheck->isChecked()) output_fields.push_back("DISPLACEMENT_X");
    if(ui->dispYCheck->isChecked()) output_fields.push_back("DISPLACEMENT_Y");
    if(ui->rotationCheck->isChecked()) output_fields.push_back("ROTATION_Z");
    if(ui->normalXCheck->isChecked()) output_fields.push_back("NORMAL_STRAIN_X");
    if(ui->normalYCheck->isChecked()) output_fields.push_back("NORMAL_STRAIN_Y");
    if(ui->shearCheck->isChecked()) output_fields.push_back("SHEAR_STRAIN_XY");
    if(ui->noiseCheck->isChecked()) output_fields.push_back("NOISE_LEVEL");
    if(ui->contrastCheck->isChecked()) output_fields.push_back("CONTRAST_LEVEL");
    if(ui->sigmaCheck->isChecked()) output_fields.push_back("SIGMA");
    if(ui->gammaCheck->isChecked()) output_fields.push_back("GAMMA");
    if(ui->betaCheck->isChecked()) output_fields.push_back("BETA");
    if(ui->numActiveCheck->isChecked()) output_fields.push_back("ACTIVE_PIXELS");
    if(ui->statusCheck->isChecked()) output_fields.push_back("STATUS_FLAG");
    DICe::gui::Input_Vars::instance()->set_output_fields(output_fields);

    // export the vertices from the region defs
    ui->simpleQtVTKWidget->exportVertices(DICe::gui::Input_Vars::instance()->boundaryShapes(),DICe::gui::Input_Vars::instance()->excludedShapes());
    if(DICe::gui::Input_Vars::instance()->boundaryShapes()->size()==0&&
            DICe::gui::Input_Vars::instance()->excludedShapes()->size()!=0){
        QMessageBox msgBox;
        msgBox.setText("Warning: there are excluded regions defined,\nbut no included regions. Excluded regions will be ignored.");
        msgBox.exec();
    }
    DICe::gui::Input_Vars::instance()->write_input_file();
    DICe::gui::Input_Vars::instance()->write_params_file();
    DICe::gui::Input_Vars::instance()->write_subset_file();
}

void MainWindow::prepResultsViewer()
{
    QString dir = DICe::gui::Input_Vars::instance()->get_working_dir();
    std::stringstream working_dir_ss;
#ifdef WIN32
    working_dir_ss << dir.toStdString() << "\\results\\";
#else
    working_dir_ss << dir.toStdString() << "/results/";
#endif
    QStringList resFiles;
    QDirIterator it(QString::fromStdString(working_dir_ss.str()), QStringList() << "DICe_solution_*.txt",
                   QDir::Files,QDirIterator::Subdirectories);
    while (it.hasNext()){
       resFiles.push_back(it.next());
    }
    //for(QStringList::iterator sit=resFiles.begin();sit!=resFiles.end();++sit)
    //    std::cout << " FILE " << sit->toStdString() << std::endl;

    // get the image files
    QStringList defImageFiles = *DICe::gui::Input_Vars::instance()->get_def_file_list();

    QStringList images;
    for(QStringList::iterator sit=defImageFiles.begin();sit!=defImageFiles.end();++sit)
        images.push_back(*sit);
    //for(QStringList::iterator sit=images.begin();sit!=images.end();++sit)
    //    std::cout << " Image FILE " << sit->toStdString() << std::endl;

    if(defImageFiles.size()>0){
        try{
            ui->simpleQtVTKWidget->setFileNames(resFiles,images);
        }catch(std::exception & e){
            std::cout << "Exception was thrown !!" << e.what() << std::endl;
        }
    }
    else{
        QMessageBox msgBox;
        msgBox.setText("ERROR: there are no deformed images");
        msgBox.exec();
        exit(EXIT_FAILURE);
    }

    // switch the active tab to results
    ui->simpleQtVTKWidget->changeInteractionMode(1);
}

void MainWindow::readValidationOutput(){
    ui->consoleEdit->append(checkValidDiceProcess->readAllStandardOutput());
}

void MainWindow::readOutput(){
    ui->consoleEdit->append(diceProcess->readAllStandardOutput());
    if(ui->progressBar->value()<ui->progressBar->maximum()-1)
        ui->progressBar->setValue(ui->progressBar->value()+1);
}

void MainWindow::on_runButton_clicked()
{    

    // check for existing input for results files
    // if there are existing files in the dir ask if they should be over-written:
    QString inputFile = DICe::gui::Input_Vars::instance()->get_working_dir();
#ifdef WIN32
    inputFile +=  "\\input.xml";
#else
    inputFile += "/input.xml";
#endif
    QFileInfo checkFile(inputFile);
    // check if file exists and if yes: Is it really a file and no directory?
    if (checkFile.exists() && checkFile.isFile()) {
        QMessageBox msgBox;
        msgBox.setText("Found existing input or results files\nin the working directory.\n\nOverwrite and continue?");
        msgBox.setStandardButtons(QMessageBox::Yes);
        msgBox.addButton(QMessageBox::No);
        msgBox.setDefaultButton(QMessageBox::Yes);
        if(msgBox.exec() == QMessageBox::No){
            return;
        }
    }

    // reset the progress bar
    ui->progressBar->setValue(1);
    ui->progressBar->setMaximum(ui->defListWidget->count()+2);

    // write intput files, abort if write fails
    writeInputFiles();

    QString dir = DICe::gui::Input_Vars::instance()->get_working_dir();
    std::stringstream working_dir_ss;
#ifdef WIN32
    working_dir_ss << dir.toStdString() << "\\results\\";
#else
    working_dir_ss << dir.toStdString() << "/results/";
#endif
    std::cout << "Clearing results directory " << std::endl;
    QStringList resFiles;
    QDirIterator it(QString::fromStdString(working_dir_ss.str()), QStringList() << "*.txt",
                   QDir::Files,QDirIterator::Subdirectories);
    while (it.hasNext()){
        std::cout << "Removing file " << it.next().toStdString() << std::endl;
        QDir cDir;
        cDir.remove(it.filePath());
    }
    std::cout << "Running correlation..." << std::endl << std::endl;

    ui->runButton->setEnabled(false);
    ui->refFileButton->setEnabled(false);
    ui->defFileButton->setEnabled(false);
    ui->workingDirButton->setEnabled(false);

    QStringList args;
    args << "-i" << DICe::gui::Input_Vars::instance()->input_file_name().c_str() << "-v" << "-t";

    diceProcess->start(execPath,args);
}

void MainWindow::execWrapUp(int code){
    ui->runButton->setEnabled(true);
    ui->refFileButton->setEnabled(true);
    ui->defFileButton->setEnabled(true);
    ui->workingDirButton->setEnabled(true);


    if(diceProcess->exitStatus()||diceProcess->exitCode()||execError){
        QMessageBox msgBox;
        msgBox.setText("ERROR: DICe execution failed");
        msgBox.exec();
        return;
    }
    else
        std::cout << "DICe execution SUCCESSFUL" << std::endl;
    ui->progressBar->setValue(ui->progressBar->maximum());

    diceProcess->close();

    // after the run is complete, prepare the results viewer:
    prepResultsViewer();
}

void MainWindow::on_diceButton_clicked()
{
    QDesktopServices::openUrl(QUrl("https://github.com/dicengine/dice", QUrl::TolerantMode));
}

void MainWindow::on_stepSize_editingFinished()
{
    ui->strainWindowSpin->setMinimum(ui->stepSize->value()*2);
    ui->strainWindowSpin->setMaximum(ui->stepSize->value()*6);
    ui->strainWindowSpin->setValue(ui->stepSize->value()*2);
    ui->strainWindowSpin->setSingleStep(ui->stepSize->value());
}
