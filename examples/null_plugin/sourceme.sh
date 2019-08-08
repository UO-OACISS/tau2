TAUROOT=`tau-config | grep TAUROOT | sed 's/TAUROOT=//g'`
TAUARCH=`tau-config | grep TAUARCH | sed 's/TAUARCH=//g'`
TAU_BASEDIR=`tau-config | grep BASEDIR | sed 's/BASEDIR=//g'`

export TAU_PLUGINS=libTAU-null-plugin.so
export TAU_PLUGINS_PATH=${TAUROOT}/${TAUARCH}/lib/shared-mpi-pthread


