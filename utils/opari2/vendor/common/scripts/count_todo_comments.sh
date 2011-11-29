echo
echo "checking for TODO in Score-P"
echo "----------------------------"
echo


FILES="src/measurement/online_access/scorep_oa_sockets.c
src/measurement/online_access/scorep_oa_connection.h
src/measurement/online_access/scorep_oa_mri_control.h
src/measurement/online_access/SCOREP_OA_Init.h
src/measurement/online_access/lex.yy.c
src/measurement/scorep_definition_cube4.c
src/measurement/SCOREP_Definitions.c
test/jacobi
tools/oa_registry/regsrv_sockets.c
tools/mpi_wrapgen"

for i in $FILES; do
  echo "$i:"
  grep -R "TODO" $i | grep -v .svn | grep -v "backup" | grep -v "GENERATE_TODOLIST" | grep -v "build-config/ltmain.sh" | grep -v doxygen | grep -c TODO
  echo
done

echo
echo
echo "checking for TODO in OTF 2"
echo "----------------------------"
echo


FILES="otf2/src/OTF2_EvtReader.c
otf2/src/tools/otf2_compare/
otf2/src/tools/otf2_trace_gen/otf2_trace_gen.c
otf2/src/OTF2_File_Sion.c
otf2/src/OTF2_Buffer.c
otf2/src/OTF2_InternalArchive.c
otf2/src/OTF2_DefReader.c
otf2/test/OTF2_PrintLocal_test/OTF2_PrintLocal_test.c
otf2/include/otf2/OTF2_GlobEvtReader.h"

for i in $FILES; do
  echo "$i:"
  grep -R "TODO" vendor/$i | grep -v .svn | grep -v "backup" | grep -v "GENERATE_TODOLIST" | grep -v "build-config/ltmain.sh" | grep -v doxygen | grep -c TODO
  echo
done

echo
echo
echo "checking for TODO in Opari 2"
echo "----------------------------"
echo


FILES="vendor/opari2/test/data/jacobi/"

for i in $FILES; do
  echo "$i:" 
  grep -R "TODO" $i | grep -v .svn | grep -v "backup" | grep -v "GENERATE_TODOLIST" | grep -v "build-config/ltmain.sh" | grep -v doxygen | grep -c TODO
  echo
done

echo
echo
echo "checking for todo in Score-P"
echo "----------------------------"
echo


FILES="include/scorep/SCOREP_Tau.h
src/adapters/tau/SCOREP_Tau.c
src/adapters/pomp/SCOREP_Pomp_RegionInfo.h
src/services/scorep_timer_cycle_counter_tsc.c
src/services/include/SCOREP_Timing.h
src/measurement/scorep_environment.c
src/measurement/SCOREP_Events.c
src/measurement/scorep_thread.c
src/measurement/paradigm/
src/measurement/scorep_runtime_management.h
src/measurement/scorep_definition_management.c
src/measurement/SCOREP_Memory.c
src/measurement/tracing/scorep_tracing_definitions.c
src/measurement/tracing/SCOREP_Tracing.c
src/measurement/SCOREP_RuntimeManagement.c
src/measurement/online_access/scorep_oa_connection.c
src/measurement/SCOREP_Definitions.c
src/measurement/include/SCOREP_Events.h
src/measurement/include/SCOREP_Memory.h
src/measurement/include/SCOREP_RuntimeManagement.h
src/measurement/include/SCOREP_Definitions.h
src/measurement/include/SCOREP_Types.h
src/measurement/SCOREP_Config.c
src/measurement/scorep_definitions.h"


for i in $FILES; do
  echo "$i:"
  grep -R "todo" $i | grep -v .svn | grep -v "backup" | grep -v "GENERATE_TODOLIST" | grep -v "build-config/ltmain.sh" | grep -v doxygen | grep -c todo 
  echo
done

echo
echo
echo "checking for todo in Utility"
echo "----------------------------"
echo


FILES="src/id_map/SCOREP_IdMap.c
src/memory/SCOREP_Allocator.c"

for i in $FILES; do
  echo "$i:"
  grep -c "todo" vendor/otf2/vendor/utility/$i 
  echo
done

echo
echo
echo "checking for todo in Opari2"
echo "----------------------------"
echo


FILES="src/pomp-lib-dummy/pomp2_region_info.c
src/pomp-lib-dummy/pomp2_lib.c"

for i in $FILES; do
  echo "$i:"
  grep -c "todo" vendor/opari2/$i 
  echo
done






