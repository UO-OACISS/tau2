/*
  DESCRIPTION:

    The example code demonstrates gather/scatter operations with the GPI.
    In this example the master node scatters data packets divided into smaller
    blocks to all worker nodes and afterwards gathers the data packets back
    into their orginal positions. At the end the result is checked.
*/


#include <GPI.h>
#include <GpiLogger.h>
#include <MCTP1.h>

#include <signal.h>
#include <assert.h>
#include <cstring>
#include <stdio.h>
#include <fcntl.h>


#define GB 1073741824

#define PACKETSIZE  (1<<24)
#define BLOCKSIZE (4096)
#define MTUSIZE (2048)


void signalHandlerMaster(int sig){

  //do master node signal handling...
  //kill the gpi processes on all worker nodes, only callable from master
  killProcsGPI();
  //shutdown nicely
  shutdownGPI();

  exit(-1);
}


void signalHandlerWorker(int sig){

  //do worker node signal handling...
  //shutdown nicely
  shutdownGPI();

  exit(-1);
}


int checkEnv(int argc, char* argv[]){

  int errors = 0;

  if(isMasterProcGPI(argc, argv) == 1){

    const int nodes = generateHostlistGPI();
    const unsigned short port = getPortGPI();

    //check setup of all nodes
    for(int rank=0; rank<nodes; rank++){

      int retval;

      //translate rank to hostname
      const char *host = getHostnameGPI(rank);

      //check daemon on host
      if(pingDaemonGPI(host) != 0){

        gpi_printf("Daemon ping failed on host %s with rank %d\n", host, rank);
        errors++;
        continue;
      }

      //check port on host
      if((retval=checkPortGPI(host, port)) != 0){

        gpi_printf("Port check failed (return value %d) on host %s with rank %d\n", retval, host, rank);
        errors++;

        //check for running binaries
        if(findProcGPI(host) == 0){

          gpi_printf("Another GPI binary is running and blocking the port\n");
          if(killProcsGPI() == 0){

            gpi_printf("Successfully killed old GPI binary\n");
            errors--;
          }
        }
      }

      //check shared lib setup on host
      if((retval=checkSharedLibsGPI(host)) != 0){

        gpi_printf("Shared libs check failed (return value %d) on host %s with rank %d\n", retval, host, rank);
        errors++;
      }

      //final test
      if((retval=runIBTestGPI(host)) != 0){

        gpi_printf("IB test failed (return value %d) on host %s with rank %d\n", retval, host, rank);
        errors++;
      }
    }
  }

  return errors;
}


int scatter(void *memptr, const unsigned long scattersize, const unsigned long blocksize){

  //divide data into blocks
  const unsigned long iterations = scattersize/blocksize;
  const unsigned long remainder = scattersize%blocksize;

  const int nodecount = getNodeCountGPI();
  const int queuedepth = getQueueDepthGPI();

  unsigned long total = 0;
  int requests = 0;
  int error = 0;

  //for each node...
  for(int j=1; j<nodecount; j++){

    //for each packet...
    for(unsigned long i=0; i<iterations; i++){

      //...scatter packet
      error += writeDmaGPI(total, 0, blocksize, j, GPIQueue0);
      total += blocksize;
      requests++;

      //check for queue saturation
      if(requests == queuedepth){

        //and wait for all previous cmds to finish if necessary
        if(waitDmaGPI(GPIQueue0) == -1)
          error += -1;

        requests = 0;
      }
    }

    //take care of the last block that might not be full size
    if(remainder != 0){

      error += writeDmaGPI(total, 0, remainder, j, GPIQueue0);
      total += remainder;
    }
  }

  if(waitDmaGPI(GPIQueue0) == -1)
    error += -1;

  return error;
}


int gather(void *memptr, const unsigned long gathersize, const unsigned long blocksize){

  //divide data into blocks
  const unsigned long iterations = gathersize/blocksize;
  const unsigned long remainder = gathersize%blocksize;

  const int nodecount = getNodeCountGPI();
  const int queuedepth = getQueueDepthGPI();

  unsigned long total = 0;
  int requests = 0;
  int error = 0;

  //for each node...
  for(int j=1; j<nodecount; j++){

    //for each packet...
    for(unsigned long i=0; i<iterations; i++){

      //...gather packet
      error += readDmaGPI(total, 0, blocksize, j, GPIQueue0);
      total += blocksize;
      requests++;

      //check for queue saturation
      if(requests == queuedepth){

        //and wait for all previous cmds to finish if necessary
        if(waitDmaGPI(GPIQueue0) == -1)
          error += -1;

        requests = 0;
      }
    }

    //take care of the last block that might not be full size
    if(remainder != 0){

      error += readDmaGPI(total, 0, remainder, j, GPIQueue0);
      total += remainder;
    }
  }

  if(waitDmaGPI(GPIQueue0) == -1)
    error += -1;

  return error;
}


int check(void *memptr, const unsigned long size){

  const char *ptr = static_cast<const char*>(memptr);

  for(unsigned long i=0; i<size; i++)
    if(ptr[i] != 0)
      return -1;

  return 0;
}


int main(int argc, char *argv[]){

  //check the runtime eviroment
  if(checkEnv(argc, argv) != 0)
    return -1;

  //everything good to go, start the GPI
  if(startGPI(argc, argv, "", GB) != 0){

    gpi_printf("GPI start-up failed\n");
    killProcsGPI();
    shutdownGPI();
    return -1;
  }

  const int rank = getRankGPI();

  gpi_printf("getRankGPI returns %d\n", rank);
  if(rank == 0){

    //setup signal handling
    signal(SIGINT, signalHandlerWorker);

    //mctp timer for high resolution timing
    mctpInitTimer();

    //init memory
    const int workercount = getNodeCountGPI()-1;
    void *memptr = getDmaMemPtrGPI();
    memset(memptr, 0, workercount*PACKETSIZE);

    barrierGPI();

    mctpStartTimer();

    //scatter
    if(scatter(memptr, PACKETSIZE, BLOCKSIZE) != 0)
      gpi_printf("Errors during scatter\n");

    mctpStopTimer();

    const unsigned long tsize = static_cast<unsigned long>(workercount)*PACKETSIZE;
    gpi_printf("Scattered %i bytes to %i nodes in %f msecs (%f GB/s)\n", tsize, workercount, mctpGetTimerMSecs(), static_cast<double>(tsize)/1073741824.0/mctpGetTimerSecs());

    //reinit memory to detect errors
    memset(memptr, 255, workercount*PACKETSIZE);

    mctpStartTimer();

    //gather
    if(gather(memptr, PACKETSIZE, BLOCKSIZE) != 0)
      gpi_printf("Errors during gather\n");

    mctpStopTimer();

    gpi_printf("Gathered %i bytes from %i nodes in %f msecs (%f GB/s)\n", tsize, workercount, mctpGetTimerMSecs(), static_cast<double>(tsize)/1073741824.0/mctpGetTimerSecs());

    //check
    if(check(memptr, workercount*PACKETSIZE) != 0)
      gpi_printf("Check not successful!\n");
  }
  else{
    signal(SIGINT, signalHandlerMaster);

    //init memory
    void *memptr = getDmaMemPtrGPI();
    memset(memptr, 255, PACKETSIZE);

    barrierGPI();
  }

  //everything finished, syncronize
  barrierGPI();

  //shutdown
  shutdownGPI();

  return 0;
}

