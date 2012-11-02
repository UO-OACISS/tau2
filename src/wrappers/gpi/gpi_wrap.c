#include <GPI.h>
#include <Profile/Profiler.h>
#include <stdio.h>

int TAUDECL tau_totalnodes(int set_or_get, int value);
static int tau_gpi_tagid=0 ;
#define TAU_GPI_TAGID tau_gpi_tagid=tau_gpi_tagid%250
#define TAU_GPI_TAGID_NEXT (++tau_gpi_tagid) % 250 


/**********************************************************
   getNumberOfQueuesGPI
 **********************************************************/

int  __real_getNumberOfQueuesGPI() ;
int  __wrap_getNumberOfQueuesGPI()  {

  int retval = 0;
  TAU_PROFILE_TIMER(t,"int getNumberOfQueuesGPI() C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_getNumberOfQueuesGPI();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   getQueueDepthGPI
 **********************************************************/

int  __real_getQueueDepthGPI() ;
int  __wrap_getQueueDepthGPI()  {

  int retval = 0;
  TAU_PROFILE_TIMER(t,"int getQueueDepthGPI() C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_getQueueDepthGPI();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   getNumberOfCountersGPI
 **********************************************************/

int  __real_getNumberOfCountersGPI() ;
int  __wrap_getNumberOfCountersGPI()  {

  int retval = 0;
  TAU_PROFILE_TIMER(t,"int getNumberOfCountersGPI() C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_getNumberOfCountersGPI();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   getVersionGPI
 **********************************************************/

float  __real_getVersionGPI() ;
float  __wrap_getVersionGPI()  {

  float retval = 0;
  TAU_PROFILE_TIMER(t,"float getVersionGPI() C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_getVersionGPI();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   getPortGPI
 **********************************************************/

unsigned short  __real_getPortGPI() ;
unsigned short  __wrap_getPortGPI()  {

  unsigned short retval = 0;
  TAU_PROFILE_TIMER(t,"unsigned short getPortGPI() C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_getPortGPI();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   setNetworkGPI
 **********************************************************/

int  __real_setNetworkGPI(GPI_NETWORK_TYPE a1) ;
int  __wrap_setNetworkGPI(GPI_NETWORK_TYPE a1)  {

  int retval = 0;
  TAU_PROFILE_TIMER(t,"int setNetworkGPI(GPI_NETWORK_TYPE) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_setNetworkGPI(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   setPortGPI
 **********************************************************/

int  __real_setPortGPI(unsigned short a1) ;
int  __wrap_setPortGPI(unsigned short a1)  {

  int retval = 0;
  TAU_PROFILE_TIMER(t,"int setPortGPI(unsigned short) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_setPortGPI(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   setMtuSizeGPI
 **********************************************************/

int  __real_setMtuSizeGPI(unsigned int a1) ;
int  __wrap_setMtuSizeGPI(unsigned int a1)  {

  int retval = 0;
  TAU_PROFILE_TIMER(t,"int setMtuSizeGPI(unsigned int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_setMtuSizeGPI(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   setNpGPI
 **********************************************************/

void  __real_setNpGPI(unsigned int a1) ;
void  __wrap_setNpGPI(unsigned int a1)  {

  TAU_PROFILE_TIMER(t,"void setNpGPI(unsigned int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_setNpGPI(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   generateHostlistGPI
 **********************************************************/

int  __real_generateHostlistGPI() ;
int  __wrap_generateHostlistGPI()  {

  int retval = 0;
  TAU_PROFILE_TIMER(t,"int generateHostlistGPI() C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_generateHostlistGPI();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   getHostnameGPI
 **********************************************************/

const char *  __real_getHostnameGPI(unsigned int a1) ;
const char *  __wrap_getHostnameGPI(unsigned int a1)  {

  const char * retval = 0;
  TAU_PROFILE_TIMER(t,"const char *getHostnameGPI(unsigned int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_getHostnameGPI(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   getNumaSocketGPI
 **********************************************************/

int  __real_getNumaSocketGPI(int a1, char ** a2) ;
int  __wrap_getNumaSocketGPI(int a1, char ** a2)  {

  int retval = 0;
  TAU_PROFILE_TIMER(t,"int getNumaSocketGPI(int, char **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_getNumaSocketGPI(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   enableMultiSocketGPI
 **********************************************************/

int  __real_enableMultiSocketGPI() ;
int  __wrap_enableMultiSocketGPI()  {

  int retval = 0;
  TAU_PROFILE_TIMER(t,"int enableMultiSocketGPI() C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_enableMultiSocketGPI();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   enableMultiSocketLoggingGPI
 **********************************************************/

void  __real_enableMultiSocketLoggingGPI() ;
void  __wrap_enableMultiSocketLoggingGPI()  {

  TAU_PROFILE_TIMER(t,"void enableMultiSocketLoggingGPI() C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_enableMultiSocketLoggingGPI();
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   disablePortCheckGPI
 **********************************************************/

int  __real_disablePortCheckGPI() ;
int  __wrap_disablePortCheckGPI()  {

  int retval = 0;
  TAU_PROFILE_TIMER(t,"int disablePortCheckGPI() C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_disablePortCheckGPI();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   pingDaemonGPI
 **********************************************************/

int  __real_pingDaemonGPI(const char * a1) ;
int  __wrap_pingDaemonGPI(const char * a1)  {

  int retval = 0;
  TAU_PROFILE_TIMER(t,"int pingDaemonGPI(const char *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_pingDaemonGPI(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   isMasterProcGPI
 **********************************************************/

int  __real_isMasterProcGPI(int a1, char ** a2) ;
int  __wrap_isMasterProcGPI(int a1, char ** a2)  {

  int retval = 0;
  TAU_PROFILE_TIMER(t,"int isMasterProcGPI(int, char **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_isMasterProcGPI(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   checkSharedLibsGPI
 **********************************************************/

int  __real_checkSharedLibsGPI(const char * a1) ;
int  __wrap_checkSharedLibsGPI(const char * a1)  {

  int retval = 0;
  TAU_PROFILE_TIMER(t,"int checkSharedLibsGPI(const char *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_checkSharedLibsGPI(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   checkPortGPI
 **********************************************************/

int  __real_checkPortGPI(const char * a1, unsigned short a2) ;
int  __wrap_checkPortGPI(const char * a1, unsigned short a2)  {

  int retval = 0;
  TAU_PROFILE_TIMER(t,"int checkPortGPI(const char *, unsigned short) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_checkPortGPI(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   findProcGPI
 **********************************************************/

int  __real_findProcGPI(const char * a1) ;
int  __wrap_findProcGPI(const char * a1)  {

  int retval = 0;
  TAU_PROFILE_TIMER(t,"int findProcGPI(const char *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_findProcGPI(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   clearFileCacheGPI
 **********************************************************/

int  __real_clearFileCacheGPI(const char * a1) ;
int  __wrap_clearFileCacheGPI(const char * a1)  {

  int retval = 0;
  TAU_PROFILE_TIMER(t,"int clearFileCacheGPI(const char *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_clearFileCacheGPI(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   runIBTestGPI
 **********************************************************/

int  __real_runIBTestGPI(const char * a1) ;
int  __wrap_runIBTestGPI(const char * a1)  {

  int retval = 0;
  TAU_PROFILE_TIMER(t,"int runIBTestGPI(const char *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_runIBTestGPI(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   startGPI
 **********************************************************/

int  __wrap_getRankGPI();

int  __real_startGPI(int a1, char ** a2, const char * a3, unsigned long a4) ;
int  __wrap_startGPI(int a1, char ** a2, const char * a3, unsigned long a4)  {

  int retval = 0;
  Tau_create_top_level_timer_if_necessary();
  TAU_PROFILE_TIMER(t,"int startGPI(int, char **, const char *, unsigned long) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_startGPI(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);
  tau_totalnodes(1,getNodeCountGPI());
  TAU_PROFILE_SET_NODE(getRankGPI()); 

  return retval;

}


/**********************************************************
   shutdownGPI
 **********************************************************/

void  __real_shutdownGPI() ;
void  __wrap_shutdownGPI()  {

  TAU_PROFILE_TIMER(t,"void shutdownGPI() C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shutdownGPI();
  TAU_PROFILE_STOP(t);
  Tau_stop_top_level_timer_if_necessary(); 

}


/**********************************************************
   killProcsGPI
 **********************************************************/

int  __real_killProcsGPI() ;
int  __wrap_killProcsGPI()  {

  int retval = 0;
  TAU_PROFILE_TIMER(t,"int killProcsGPI() C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_killProcsGPI();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   getRankGPI
 **********************************************************/

int  __real_getRankGPI() ;
int  __wrap_getRankGPI()  {

  int retval = 0;
  TAU_PROFILE_TIMER(t,"int getRankGPI() C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_getRankGPI();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   getNodeCountGPI
 **********************************************************/

int  __real_getNodeCountGPI() ;
int  __wrap_getNodeCountGPI()  {

  int retval = 0;
  TAU_PROFILE_TIMER(t,"int getNodeCountGPI() C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_getNodeCountGPI();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   getMemWorkerGPI
 **********************************************************/

unsigned long  __real_getMemWorkerGPI() ;
unsigned long  __wrap_getMemWorkerGPI()  {

  unsigned long retval = 0;
  TAU_PROFILE_TIMER(t,"unsigned long getMemWorkerGPI() C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_getMemWorkerGPI();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   getDmaMemPtrGPI
 **********************************************************/

void *  __real_getDmaMemPtrGPI() ;
void *  __wrap_getDmaMemPtrGPI()  {

  void * retval = 0;
  TAU_PROFILE_TIMER(t,"void *getDmaMemPtrGPI() C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_getDmaMemPtrGPI();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   barrierGPI
 **********************************************************/

void  __real_barrierGPI() ;
void  __wrap_barrierGPI()  {

  TAU_PROFILE_TIMER(t,"void barrierGPI() C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_barrierGPI();
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   allReduceGPI
 **********************************************************/

int  __real_allReduceGPI(void * a1, void * a2, unsigned char a3, GPI_OP a4, GPI_TYPE a5) ;
int  __wrap_allReduceGPI(void * a1, void * a2, unsigned char a3, GPI_OP a4, GPI_TYPE a5)  {

  int retval = 0;
  TAU_PROFILE_TIMER(t,"int allReduceGPI(void *, void *, unsigned char, GPI_OP, GPI_TYPE) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_allReduceGPI(a1, a2, a3, a4, a5);
  /* TAU_ALLREDUCE_DATA(a3*sizeof(a5));
   * We do not know a good way to get size of a5. The type is 
   * typedef enum { GPI_INT=0, GPI_UINT=1, GPI_FLOAT=2, GPI_DOUBLE=3, GPI_LONG=4, GPI_ULONG=5 } GPI_TYPE;
   *
   */
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   readDmaGPI
 **********************************************************/

int  __real_readDmaGPI(unsigned long a1, unsigned long a2, int a3, unsigned int a4, unsigned int a5) ;
int  __wrap_readDmaGPI(unsigned long a1, unsigned long a2, int a3, unsigned int a4, unsigned int a5)  {

  int retval = 0;
  TAU_PROFILE_TIMER(t,"int readDmaGPI(unsigned long, unsigned long, int, unsigned int, unsigned int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  /* a3 is size, a4 is remote rank */
  TAU_TRACE_SENDMSG_REMOTE(TAU_GPI_TAGID_NEXT, Tau_get_node(), a3, a4);

  retval  =  __real_readDmaGPI(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG(TAU_GPI_TAGID, a4, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   writeDmaGPI
 **********************************************************/

int  __real_writeDmaGPI(unsigned long a1, unsigned long a2, int a3, unsigned int a4, unsigned int a5) ;
int  __wrap_writeDmaGPI(unsigned long a1, unsigned long a2, int a3, unsigned int a4, unsigned int a5)  {

  int retval = 0;
  TAU_PROFILE_TIMER(t,"int writeDmaGPI(unsigned long, unsigned long, int, unsigned int, unsigned int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  /* a3 is size, a4 is remote rank */
  /* TAU_VERBOSE("Writing to remote node: writeDmaGPI: a4 rank = %d a3 size = %d\n", a4, a3); */
  TAU_TRACE_SENDMSG(TAU_GPI_TAGID_NEXT, a4, a3);
  retval  =  __real_writeDmaGPI(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG_REMOTE(TAU_GPI_TAGID, Tau_get_node(), a3,  a4);

  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   sendDmaGPI
 **********************************************************/

int  __real_sendDmaGPI(unsigned long a1, int a2, unsigned int a3, unsigned int a4) ;
int  __wrap_sendDmaGPI(unsigned long a1, int a2, unsigned int a3, unsigned int a4)  {

  int retval = 0;
  TAU_PROFILE_TIMER(t,"int sendDmaGPI(unsigned long, int, unsigned int, unsigned int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  /* a2 is size, a3 is remote rank */
  TAU_TRACE_SENDMSG(TAU_GPI_TAGID_NEXT, a3, a2);

  retval  =  __real_sendDmaGPI(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   recvDmaGPI
 **********************************************************/

int  __real_recvDmaGPI(unsigned long a1, int a2, unsigned int a3, unsigned int a4) ;
int  __wrap_recvDmaGPI(unsigned long a1, int a2, unsigned int a3, unsigned int a4)  {

  int retval = 0;
  TAU_PROFILE_TIMER(t,"int recvDmaGPI(unsigned long, int, unsigned int, unsigned int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_recvDmaGPI(a1, a2, a3, a4);
  /* a2 is size, a3 is remote rank */
  TAU_TRACE_RECVMSG(TAU_GPI_TAGID_NEXT, a3, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   waitDmaGPI
 **********************************************************/

int  __real_waitDmaGPI(unsigned int a1) ;
int  __wrap_waitDmaGPI(unsigned int a1)  {

  int retval = 0;
  TAU_PROFILE_TIMER(t,"int waitDmaGPI(unsigned int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_waitDmaGPI(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   waitDma2GPI
 **********************************************************/

int  __real_waitDma2GPI(unsigned int a1) ;
int  __wrap_waitDma2GPI(unsigned int a1)  {

  int retval = 0;
  TAU_PROFILE_TIMER(t,"int waitDma2GPI(unsigned int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_waitDma2GPI(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   sendDmaPassiveGPI
 **********************************************************/

int  __real_sendDmaPassiveGPI(unsigned long a1, int a2, unsigned int a3) ;
int  __wrap_sendDmaPassiveGPI(unsigned long a1, int a2, unsigned int a3)  {

  int retval = 0;
  TAU_PROFILE_TIMER(t,"int sendDmaPassiveGPI(unsigned long, int, unsigned int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  /* a2 is size, a3 is remote rank */
  TAU_TRACE_SENDMSG(TAU_GPI_TAGID_NEXT, a3, a2);

  retval  =  __real_sendDmaPassiveGPI(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   waitDmaPassiveGPI
 **********************************************************/

int  __real_waitDmaPassiveGPI() ;
int  __wrap_waitDmaPassiveGPI()  {

  int retval = 0;
  TAU_PROFILE_TIMER(t,"int waitDmaPassiveGPI() C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_waitDmaPassiveGPI();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   recvDmaPassiveGPI
 **********************************************************/

int  __real_recvDmaPassiveGPI(unsigned long a1, int a2, int * a3) ;
int  __wrap_recvDmaPassiveGPI(unsigned long a1, int a2, int * a3)  {

  int retval = 0;
  TAU_PROFILE_TIMER(t,"int recvDmaPassiveGPI(unsigned long, int, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_recvDmaPassiveGPI(a1, a2, a3);
  /* a2 is size, a3 is remote rank */
  TAU_TRACE_RECVMSG(TAU_GPI_TAGID_NEXT, *a3, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   getCommandGPI
 **********************************************************/

int  __real_getCommandGPI() ;
int  __wrap_getCommandGPI()  {

  int retval = 0;
  TAU_PROFILE_TIMER(t,"int getCommandGPI() C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_getCommandGPI();
  /* command is an int in size, 0 is remote rank (master) */
  TAU_TRACE_RECVMSG(TAU_GPI_TAGID_NEXT, sizeof(int), 0); 
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   setCommandGPI
 **********************************************************/

int  __real_setCommandGPI(int a1) ;
int  __wrap_setCommandGPI(int a1)  {

  int retval = 0; 
  int i; 
  int N; 
  TAU_PROFILE_TIMER(t,"int setCommandGPI(int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  N=__wrap_getNodeCountGPI();
  /* command is an int in size, i is remote rank (1 to N-1) */
  for(i=1; i < N; i++) {
    TAU_TRACE_SENDMSG(TAU_GPI_TAGID_NEXT, sizeof(int), i); 
  }
  retval  =  __real_setCommandGPI(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   getCommandFromNodeIdGPI
 **********************************************************/

long  __real_getCommandFromNodeIdGPI(unsigned int a1) ;
long  __wrap_getCommandFromNodeIdGPI(unsigned int a1)  {

  long retval = 0;
  TAU_PROFILE_TIMER(t,"long getCommandFromNodeIdGPI(unsigned int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_getCommandFromNodeIdGPI(a1);
  /* command is an int in size, a1 is remote rank */
  TAU_TRACE_RECVMSG(TAU_GPI_TAGID_NEXT, sizeof(int), a1); 
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   setCommandToNodeIdGPI
 **********************************************************/

long  __real_setCommandToNodeIdGPI(unsigned int a1, long a2) ;
long  __wrap_setCommandToNodeIdGPI(unsigned int a1, long a2)  {

  long retval = 0;
  TAU_PROFILE_TIMER(t,"long setCommandToNodeIdGPI(unsigned int, long) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  /* command is an int in size (a2 is cmd), a1 is remote rank (1 to N-1) */
  TAU_TRACE_SENDMSG(TAU_GPI_TAGID_NEXT, sizeof(int), a1); 
  retval  =  __real_setCommandToNodeIdGPI(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   atomicFetchAddTileCntGPI
 **********************************************************/

unsigned long  __real_atomicFetchAddTileCntGPI(unsigned long a1) ;
unsigned long  __wrap_atomicFetchAddTileCntGPI(unsigned long a1)  {

  unsigned long retval = 0;
  TAU_PROFILE_TIMER(t,"unsigned long atomicFetchAddTileCntGPI(unsigned long) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_atomicFetchAddTileCntGPI(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   atomicCmpSwapTileCntGPI
 **********************************************************/

unsigned long  __real_atomicCmpSwapTileCntGPI(unsigned long a1, unsigned long a2) ;
unsigned long  __wrap_atomicCmpSwapTileCntGPI(unsigned long a1, unsigned long a2)  {

  unsigned long retval = 0;
  TAU_PROFILE_TIMER(t,"unsigned long atomicCmpSwapTileCntGPI(unsigned long, unsigned long) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_atomicCmpSwapTileCntGPI(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   atomicResetTileCntGPI
 **********************************************************/

int  __real_atomicResetTileCntGPI() ;
int  __wrap_atomicResetTileCntGPI()  {

  int retval = 0;
  TAU_PROFILE_TIMER(t,"int atomicResetTileCntGPI() C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_atomicResetTileCntGPI();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   atomicFetchAddCntGPI
 **********************************************************/

unsigned long  __real_atomicFetchAddCntGPI(unsigned long a1, unsigned int a2) ;
unsigned long  __wrap_atomicFetchAddCntGPI(unsigned long a1, unsigned int a2)  {

  unsigned long retval = 0;
  TAU_PROFILE_TIMER(t,"unsigned long atomicFetchAddCntGPI(unsigned long, unsigned int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_atomicFetchAddCntGPI(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   atomicCmpSwapCntGPI
 **********************************************************/

unsigned long  __real_atomicCmpSwapCntGPI(unsigned long a1, unsigned long a2, unsigned int a3) ;
unsigned long  __wrap_atomicCmpSwapCntGPI(unsigned long a1, unsigned long a2, unsigned int a3)  {

  unsigned long retval = 0;
  TAU_PROFILE_TIMER(t,"unsigned long atomicCmpSwapCntGPI(unsigned long, unsigned long, unsigned int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_atomicCmpSwapCntGPI(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   atomicResetCntGPI
 **********************************************************/

int  __real_atomicResetCntGPI(unsigned int a1) ;
int  __wrap_atomicResetCntGPI(unsigned int a1)  {

  int retval = 0;
  TAU_PROFILE_TIMER(t,"int atomicResetCntGPI(unsigned int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_atomicResetCntGPI(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   globalResourceLockGPI
 **********************************************************/

int  __real_globalResourceLockGPI() ;
int  __wrap_globalResourceLockGPI()  {

  int retval = 0;
  TAU_PROFILE_TIMER(t,"int globalResourceLockGPI() C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_globalResourceLockGPI();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   globalResourceUnlockGPI
 **********************************************************/

int  __real_globalResourceUnlockGPI() ;
int  __wrap_globalResourceUnlockGPI()  {

  int retval = 0;
  TAU_PROFILE_TIMER(t,"int globalResourceUnlockGPI() C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_globalResourceUnlockGPI();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   getErrorVectorGPI
 **********************************************************/

unsigned char *  __real_getErrorVectorGPI(unsigned int a1) ;
unsigned char *  __wrap_getErrorVectorGPI(unsigned int a1)  {

  unsigned char * retval = 0;
  TAU_PROFILE_TIMER(t,"unsigned char *getErrorVectorGPI(unsigned int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_getErrorVectorGPI(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   openDmaRequestsGPI
 **********************************************************/

int  __real_openDmaRequestsGPI(unsigned int a1) ;
int  __wrap_openDmaRequestsGPI(unsigned int a1)  {

  int retval = 0;
  TAU_PROFILE_TIMER(t,"int openDmaRequestsGPI(unsigned int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_openDmaRequestsGPI(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   openDmaPassiveRequestsGPI
 **********************************************************/

int  __real_openDmaPassiveRequestsGPI() ;
int  __wrap_openDmaPassiveRequestsGPI()  {

  int retval = 0;
  TAU_PROFILE_TIMER(t,"int openDmaPassiveRequestsGPI() C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_openDmaPassiveRequestsGPI();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   getChannelFdGPI
 **********************************************************/

int  __real_getChannelFdGPI() ;
int  __wrap_getChannelFdGPI()  {

  int retval = 0;
  TAU_PROFILE_TIMER(t,"int getChannelFdGPI() C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_getChannelFdGPI();
  TAU_PROFILE_STOP(t);
  return retval;

}

