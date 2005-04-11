/////////////////////////////////////////
// File: pgm.C
// This file adapted from charm/pgms/converse/context-switch
/////////////////////////////////////////

#include <unistd.h>
#include <iostream.h>
#include <TAU.h>


#define NUM_THREADS 3

CpvDeclare(int, exitHandlerIdx);
static char endmsg[CmiMsgHeaderSizeBytes];

int numThreadsDone = 0;

void ExitHandler(void *msg) {
  numThreadsDone++;
  if (numThreadsDone >= NUM_THREADS)
    CsdExitScheduler();
}


int bar(int i) {
  TAU_PROFILE("bar", "", TAU_DEFAULT);
  double sum;

  if (i % 2) {
    CthYield();
  }

  for (int j=0; j< 13453450; j++) {

    sum = sum * 5.6 * i * j;
    double bob = 5.6 * 3.0f;
  }

}



void Yielder(void *arg) {
  TAU_REGISTER_THREAD();
  TAU_PROFILE("Yielder", "", TAU_DEFAULT);

  register int i;

  int id = (int)arg;

  for(i=0;i<3;i++) {
    bar(i);
    CmiPrintf("%d: yielding %d\n", id, i);
    CthYield();
  }

  TAU_PROFILE_EXIT();
  CmiSetHandler(endmsg, CpvAccess(exitHandlerIdx));
  CmiSyncSend(CmiMyPe(), CmiMsgHeaderSizeBytes, endmsg);
}

void test_init(int argc, char **argv) {
  TAU_PROFILE_SET_NODE(CmiMyPe());
  TAU_PROFILE("test_init", "", TAU_DEFAULT);


  CpvInitialize(int, exitHandlerIdx);
  CpvAccess(exitHandlerIdx) = CmiRegisterHandler((CmiHandler)ExitHandler);

  for (int i=0; i < NUM_THREADS; i++) {
    CthThread yielder = CthCreate((CthVoidFn)Yielder,  (void*)1, 64000);
    CthAwaken(yielder);
  }
}

int main(int argc, char **argv) {
  ConverseInit(argc, argv, test_init, 0, 0);
}

