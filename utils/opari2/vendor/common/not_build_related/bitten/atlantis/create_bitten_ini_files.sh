#!/bin/bash

PATHS="trunk"
COMPONENTS="utility otf2 opari2"
COMPILERS="gcc intel pgi studio"
KINDS_OF_LIB="static shared"
BITTEN_DIR="/home/roessel/bitten"

MACHINES="x86-32 x86-64"

mkdir -p ${BITTEN_DIR}/config

for path in $PATHS; do
    for component in $COMPONENTS; do
	for compiler in $COMPILERS; do
	    for kind_of_lib in $KINDS_OF_LIB; do
		for machine in $MACHINES; do
		    CONFIG_FILE="${BITTEN_DIR}/config/scorep_${component}_${path}_${compiler}_${kind_of_lib}_${machine}.ini"
		    echo "[os]" > $CONFIG_FILE
		    echo "name = Debian GNU/Linux" >> $CONFIG_FILE
		    echo "family = posix" >> $CONFIG_FILE
		    echo "version = squeeze" >> $CONFIG_FILE
		    echo >> $CONFIG_FILE
		    echo "[machine]" >> $CONFIG_FILE
		    echo "name = $machine" >> $CONFIG_FILE
		    echo "processor = Six-Core AMD Opteron(tm) Processor 2427" >> $CONFIG_FILE
		    echo >> $CONFIG_FILE
		    echo "[scorep]" >> $CONFIG_FILE
		    echo "compiler = $compiler" >> $CONFIG_FILE
		    echo "kind_of_lib = $kind_of_lib" >> $CONFIG_FILE
		    echo "path = $path" >> $CONFIG_FILE
		    echo >> $CONFIG_FILE
		    echo "[authentication]" >> $CONFIG_FILE
		    echo "username = croessel" >> $CONFIG_FILE
		    echo "password = ********" >> $CONFIG_FILE
		done
	    done
	done
    done
done


#########################################
# Component silc, i.e. with MPI
#########################################

MPIS="mpich2 openmpi impi"
component="silc"

for path in $PATHS; do
    for compiler in $COMPILERS; do
	for mpi in $MPIS; do
	    if [ "${mpi}" == "openmpi" ] && [ "${compiler}" == "studio" ]; then 
		continue 
	    fi
	    if [ "${mpi}" == "impi" ] && [ "${compiler}" == "studio" ]; then 
		continue 
	    fi
	    if [ "${mpi}" == "impi" ] && [ "${compiler}" == "pgi" ]; then 
		continue 
	    fi
            
            configure_args_1="--with-nocross-compiler-suite=${compiler} --with-mpi=${mpi}"
            if [ "${mpi}" == "openmpi" ] && [ "${compiler}" == "gcc" ]; then
                configure_args_1="${configure_args_1} --with-sionconfig=yes"
            fi

	    for kind_of_lib in $KINDS_OF_LIB; do
                if test "x${kind_of_lib}" = "xstatic"; then
                    configure_args_2="${configure_args_1} --enable-static --disable-shared --enable-silent-rules"
                else
                    configure_args_2="${configure_args_1} --enable-shared --disable-static"
                fi
		for machine in $MACHINES; do
		    CONFIG_FILE="${BITTEN_DIR}/config/scorep_${component}_${path}_${compiler}_${mpi}_${kind_of_lib}_${machine}.ini"
		    echo "[os]" > $CONFIG_FILE
		    echo "name = Debian GNU/Linux" >> $CONFIG_FILE
		    echo "family = posix" >> $CONFIG_FILE
		    echo "version = squeeze" >> $CONFIG_FILE
		    echo >> $CONFIG_FILE
		    echo "[machine]" >> $CONFIG_FILE
		    echo "name = $machine" >> $CONFIG_FILE
		    echo "processor = Six-Core AMD Opteron 2427" >> $CONFIG_FILE
		    echo >> $CONFIG_FILE
		    echo "[scorep]" >> $CONFIG_FILE
		    echo "compiler = $compiler" >> $CONFIG_FILE
		    echo "mpi = $mpi" >> $CONFIG_FILE
		    echo "kind_of_lib = $kind_of_lib" >> $CONFIG_FILE
		    echo "path = $path" >> $CONFIG_FILE
                    echo "prefix = /opt/packages/scorep-latest-${kind_of_lib}/${compiler}-${mpi}-${machine}" >> $CONFIG_FILE
                    configure_args_3="${configure_args_2} --with-pdt=/opt/packages/pdt/3.16/i386_linux/bin --with-papi-header=/opt/packages/papi/4.1.2.1/include --with-papi-lib=/opt/packages/papi/4.1.2.1/lib"
                    echo "configure_args = $configure_args_3"  >> $CONFIG_FILE
		    echo >> $CONFIG_FILE
		    echo "[authentication]" >> $CONFIG_FILE
		    echo "username = croessel" >> $CONFIG_FILE
		    echo "password = ********" >> $CONFIG_FILE
		done
	    done
	done
    done
done
