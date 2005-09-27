#include <TAU_tf.h>

#include <stdio.h>

#define MAIN 1
#define FOO 2
#define BAR 3

#define USER_EVENT_1 4

int main(int argc, char** argv) {

  printf ("Begin!\n");

  Ttf_FileHandleT file;

  file = Ttf_OpenFileForOutput("tau.trc","tau.edf");

  if (file == NULL) {
    fprintf (stderr, "Error openging trace for output\n");
    return -1;
  }

  Ttf_DefThread(file, 0, 0, "node 0");
  Ttf_DefThread(file, 1, 0, "node 1");
  Ttf_DefStateGroup(file, "TAU_DEFAULT", 1);

  Ttf_DefState(file, MAIN, "main", 1);
  Ttf_DefState(file, FOO, "foo", 1);
  Ttf_DefState(file, BAR, "bar", 1);

  Ttf_DefUserEvent(file, USER_EVENT_1, "User Event 1", 1);

  double s = 1e6;

  Ttf_EnterState(file, 1*s, 0, 0, MAIN);
  Ttf_EnterState(file, 2*s, 0, 0, FOO);
  Ttf_EnterState(file, 3*s, 0, 0, BAR);
  Ttf_EventTrigger(file, 3.1*s, 0, 0, USER_EVENT_1, 500);
  Ttf_EventTrigger(file, 3.9*s, 0, 0, USER_EVENT_1, 1000);

  Ttf_EnterState(file, 4*s, 1, 0, MAIN);
  Ttf_SendMessage(file, 4.5*s, 
		  1, 0, // from 1,0
		  0, 0, // to 0,0
		  500,  // length
		  42,   // tag
		  0);   // communicator
  Ttf_EnterState(file, 5*s, 1, 0, FOO);
  Ttf_EnterState(file, 6*s, 1, 0, BAR);
  Ttf_LeaveState(file, 7*s, 1, 0, BAR);
  Ttf_LeaveState(file, 8*s, 1, 0, FOO);
  Ttf_LeaveState(file, 9*s, 1, 0, MAIN);


  Ttf_LeaveState(file, 10*s, 0, 0, BAR);
  Ttf_RecvMessage(file, 10.5*s, 
		  1, 0, // from 1,0
		  0, 0, // to 0,0
		  500,  // length
		  42,   // tag
		  0);   // communicator
  Ttf_LeaveState(file, 11*s, 0, 0, FOO);
  Ttf_LeaveState(file, 12*s, 0, 0, MAIN);

  Ttf_CloseOutputFile(file);
  return 0;
}
