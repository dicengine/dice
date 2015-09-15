if [ -f /etc/bashrc ]; then
        . /etc/bashrc
fi

echo "--- Module purge"
module purge
echo "--- Loading modules"
module load openmpi/1.8.7/host/gnu/4.7.2/cuda/7.5.7
module load cmake
module list
echo "--- Congfiguring DICe ---"
./do-cmake

if [ $? -eq 0 ] 
then
    echo "--- Configuration successful "
else
    echo " "
    echo "--- Error: Configuraion unsuccessful, aborting "
    echo " "
    exit 0
fi

echo " " 
echo "--- Configuring and building DICe_utils"
echo " " 

# add -j 16 option
make DICe_utils

if [ $? -eq 0 ] 
then
    echo " "
    echo "--- Building DICe_utils successful "
    echo " "
else
    echo " "
    echo "--- Error: Building DICe_utils unsuccessful, aborting "
    echo " "
    exit 0
fi

echo "--- Loading nvcc modules"
module load nvcc-wrapper/gnu
module list

echo " " 
echo "--- Building DICe"
echo " " 

# add -j 16 option or make clean
make

if [ $? -eq 0 ] 
then
    echo " "
    echo "--- Building DICe successful "
    echo " "
else
    echo " "
    echo "--- Error: Building DICe unsuccessful, aborting "
    echo " "
    exit 0
fi

echo " "
echo "--- BUILD COMPLETE"
echo " " 