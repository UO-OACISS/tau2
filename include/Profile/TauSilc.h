// #include <SILC_Utils.h>
// #include <SILC_Events.h>
// #include <SILC_Definitions.h>
// #include <SILC_RuntimeManagement.h>


#define UINT32_MAX  (0xffffffff)

extern "C" {

void SILC_InitMeasurement();
#define SILC_INVALID_SOURCE_FILE UINT32_MAX
#define SILC_INVALID_LINE_NO 0
typedef enum SILC_AdapterType
  {
    SILC_ADAPTER_USER,
    SILC_ADAPTER_COMPILER,
    SILC_ADAPTER_MPI,
    SILC_ADAPTER_POMP,
    SILC_ADAPTER_PTHREAD,

    SILC_INVALID_ADAPTER_TYPE /**< For internal use only. */
  } SILC_AdapterType;

typedef enum SILC_RegionType
  {
    SILC_REGION_UNKNOWN = 0,
    SILC_REGION_FUNCTION,
    SILC_REGION_LOOP,
    SILC_REGION_USER,
    SILC_REGION_PHASE,
    SILC_REGION_DYNAMIC,

    SILC_REGION_DYNAMIC_PHASE,
    SILC_REGION_DYNAMIC_LOOP,
    SILC_REGION_DYNAMIC_FUNCTION,
    SILC_REGION_DYNAMIC_LOOP_PHASE,

    SILC_REGION_MPI_COLL_BARRIER,
    SILC_REGION_MPI_COLL_ONE2ALL,
    SILC_REGION_MPI_COLL_ALL2ONE,
    SILC_REGION_MPI_COLL_ALL2ALL,
    SILC_REGION_MPI_COLL_OTHER,

    SILC_REGION_OMP_PARALLEL,
    SILC_REGION_OMP_LOOP,
    SILC_REGION_OMP_SECTIONS,
    SILC_REGION_OMP_SECTION,
    SILC_REGION_OMP_WORKSHARE,
    SILC_REGION_OMP_SINGLE,
    SILC_REGION_OMP_MASTER,
    SILC_REGION_OMP_CRITICAL,
    SILC_REGION_OMP_ATOMIC,
    SILC_REGION_OMP_BARRIER,
    SILC_REGION_OMP_IMPLICIT_BARRIER,
    SILC_REGION_OMP_FLUSH,
    SILC_REGION_OMP_CRITICAL_SBLOCK, /**< @todo what is SBLOCK? */
    SILC_REGION_OMP_SINGLE_SBLOCK,

    SILC_INVALID_REGION_TYPE /**< For internal use only. */
  } SILC_RegionType;


typedef uint32_t SILC_RegionHandle;
typedef uint32_t SILC_SourceFileHandle;
typedef uint32_t SILC_LineNo;

SILC_RegionHandle SILC_DefineRegion(const char*           regionName,
				    SILC_SourceFileHandle fileHandle,
				    SILC_LineNo           beginLine,
				    SILC_LineNo           endLine,
				    SILC_AdapterType      adapter,
				    SILC_RegionType       regionType);



void SILC_EnterRegion(SILC_RegionHandle regionHandle);
void SILC_ExitRegion(SILC_RegionHandle regionHandle);

}
