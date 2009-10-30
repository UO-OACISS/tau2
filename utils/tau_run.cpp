/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 1999  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
*****************************************************************************
**    Description: tau_run is a utility program that spawns a user 	   **
**		   application and instruments it using DynInst package    **
**		   (from U. Maryland). 					   **
**		   Profile/trace data is generated as the program executes **
****************************************************************************/


#include "BPatch.h"
#include "BPatch_Vector.h"
#include "BPatch_function.h"
#include "BPatch_thread.h"
#include "BPatch_snippet.h" 

//#include <iostream.h>
//#include <stdio.h>
//#include <string.h>
#include <unistd.h>

#ifdef i386_unknown_nt4_0
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#endif

#include "tau_platforms.h"

//#include <string>
//using namespace std;

#define MUTNAMELEN 64
#define FUNCNAMELEN 32*1024
#define NO_ERROR -1

int expectError = NO_ERROR;
int debugPrint = 0;
int binaryRewrite = 0; /* by default, it is turned off */

template class BPatch_Vector<BPatch_variableExpr*>;
void checkCost(BPatch_snippet snippet);

BPatch *bpatch;

// control debug printf statements
#define dprintf if (debugPrint) printf

/* For selective instrumentation */
extern int processInstrumentationRequests(char *fname);
extern bool areFileIncludeExcludeListsEmpty(void);
extern bool processFileForInstrumentation(const string& file_name); 
extern void printExcludeList();
extern bool instrumentEntity(const string& function_name);

/* re-writer */
BPatch *bpatch_global = NULL;
BPatch_function *name_reg;
BPatch_Vector<BPatch_snippet *> funcNames;
int addName(char *name)
{
   static int funcID = 0;
   BPatch_constExpr *name_param = new BPatch_constExpr(name);
   BPatch_constExpr *id_param = new BPatch_constExpr(funcID);
   BPatch_Vector<BPatch_snippet *> params;
   params.push_back(name_param);
   params.push_back(id_param);

   BPatch_funcCallExpr *call = new BPatch_funcCallExpr(*name_reg, params);
   funcNames.push_back(call);
   return funcID++;
}

void insertTrace(BPatch_function* functionToInstrument,
                 BPatch_addressSpace* mutatee,
                 BPatch_function* traceEntryFunc,
                 BPatch_function* traceExitFunc)
{
  char name[1024];
  char modname[1024];

  functionToInstrument->getModuleName(modname, 1024);
  if (strstr(modname, "libdyninstAPI_RT"))
     return;

  functionToInstrument->getName(name, 1024);
  int id = addName(name);
  BPatch_Vector<BPatch_snippet *> traceArgs;
  traceArgs.push_back(new BPatch_constExpr(id));

  BPatch_Vector<BPatch_point*>* funcEntry = functionToInstrument->findPoint(BPatch_entry);
  BPatch_Vector<BPatch_point*>* funcExit = functionToInstrument->findPoint(BPatch_exit);

  BPatch_funcCallExpr entryTrace(*traceEntryFunc, traceArgs);
  BPatch_funcCallExpr exitTrace(*traceExitFunc, traceArgs);

  mutatee->insertSnippet(entryTrace, *funcEntry, BPatch_callBefore, BPatch_lastSnippet);
  mutatee->insertSnippet(exitTrace, *funcExit, BPatch_callAfter, BPatch_lastSnippet);
}

struct boundFindFunc
{
  boundFindFunc(BPatch_image* img, BPatch_Vector<BPatch_function*>& funcPtrs) : m_image(img), m_outputStorage(funcPtrs)
  {
  }
  void operator()(const char* name)
  {
    m_image->findFunction(name, m_outputStorage);
  }

  BPatch_image* m_image;
  BPatch_Vector<BPatch_function*>& m_outputStorage;

};




//
// Error callback routine. 
//
void errorFunc(BPatchErrorLevel level, int num, const char **params){
    char line[256];

    const char *msg = bpatch->getEnglishErrorString(num);
    bpatch->formatErrorString(line, sizeof(line), msg, params);

    if (num != expectError) {
        printf("Error #%d (level %d): %s\n", num, level, line);
        // We consider some errors fatal.
        if (num == 101)
	  exit(-1);
    }//if
}//errorFunc()

//
// For compatibility purposes
//

BPatch_function * tauFindFunction (BPatch_image *appImage, char * functionName)
{
   // Extract the vector of functions 
   BPatch_Vector<BPatch_function *> found_funcs;
   if ((NULL == appImage->findFunction(functionName, found_funcs)) || !found_funcs.size()) {
     dprintf("tau_run: Unable to find function %s\n", functionName); 
     return NULL;
   }
   return found_funcs[0]; // return the first function found 
   // FOR DYNINST 3.0 and previous versions:
   // return appImage->findFunction(functionName);
}

//
// invokeRoutineInFunction calls routine "callee" with no arguments when 
// Function "function" is invoked at the point given by location
//
BPatchSnippetHandle *invokeRoutineInFunction(BPatch_thread *appThread,
	BPatch_image *appImage, BPatch_function *function, 
	BPatch_procedureLocation loc, BPatch_function *callee, 
	BPatch_Vector<BPatch_snippet *> *callee_args){
  
    // First create the snippet using the callee and the args 
    const BPatch_snippet *snippet = new BPatch_funcCallExpr(*callee, *callee_args);

    if (snippet == NULL) {
        fprintf(stderr, "Unable to create snippet to call callee\n");
        exit(1);
    }//if

    // Then find the points using loc (entry/exit) for the given function
    const BPatch_Vector<BPatch_point *> *points = function->findPoint(loc); 

    if(points!=NULL){
      // Insert the given snippet at the given point 
      if (loc == BPatch_entry){
	appThread->insertSnippet(*snippet, *points, BPatch_callBefore, BPatch_lastSnippet);
	//      appThread->insertSnippet(*snippet, *points);
      }//if
      else{
	appThread->insertSnippet(*snippet, *points);
      }//else
    }//if
    delete snippet; // free up space
}//invokeRoutineInFunction()

// 
// Initialize calls TauInitCode, the initialization routine in the user
// application. It is executed exactly once, before any other routine.
//
void Initialize(BPatch_thread *appThread, BPatch_image *appImage, 
	BPatch_Vector<BPatch_snippet *>& initArgs){
  // Find the initialization function and call it
  BPatch_function *call_func = tauFindFunction(appImage,"TauInitCode");
  if (call_func == NULL) {
    fprintf(stderr, "Unable to find function TauInitCode\n");
    exit(1);
  }//if

  BPatch_funcCallExpr call_Expr(*call_func, initArgs);
  if (binaryRewrite)
  {
    // checkCost(call_Expr);
    // locate the entry point for main 
    BPatch_function *main_entry = tauFindFunction(appImage, "main");
    if (main_entry == NULL) {
      fprintf(stderr, "tau_run: Unable to find function main\n");
      exit(1);
    }
    const BPatch_Vector<BPatch_point *> *points = main_entry->findPoint(BPatch_entry);
    const BPatch_snippet *snippet = new BPatch_funcCallExpr(*call_func, initArgs);
    // We invoke the Init snippet before any other call in main! 
    if((points!=NULL) && (snippet != NULL)){
      // Insert the given snippet at the given point
      appThread->insertSnippet(*snippet, *points, BPatch_callBefore, BPatch_firstSnippet);
    }
    else 
    {
      fprintf(stderr, "tau_run: entry points for main or snippet for TauInit are null\n");
      exit(1);
    }
  }
  else
  {
    appThread->oneTimeCode(call_Expr);
  }
  /* Originall, we used:
    appThread->oneTimeCode(call_Expr);
    But this does not work for binary rewriting, so we just call the routine 
    explicitly before main. Works for F90 as well. */
}//Initialize()

// FROM TEST3
//   check that the cost of a snippet is sane.  Due to differences between
//   platforms, it is impossible to check this exactly in a machine independent
//   manner.
void checkCost(BPatch_snippet snippet){
    float cost;
    BPatch_snippet copy;

    // test copy constructor too.
    copy = snippet;
    cost = snippet.getCost();
    if (cost < 0.0)
      printf("*Error*: negative snippet cost\n");
    else if (cost == 0.0) 
      printf("*Warning*: zero snippet cost\n");
    else if (cost > 0.01) 
      printf("*Error*: snippet cost of %f, exceeds max expected of 0.1",cost);
}//checkCost()

int errorPrint = 0; // external "dyninst" tracing
/* OLD void errorFunc1(BPatchErrorLevel level, int num, const char **params) */
void errorFunc1(BPatchErrorLevel level, int num,  const char* const* params)
{
    if (num == 0) {
      // conditional reporting of warnings and informational messages
      if (errorPrint) {
	if (level == BPatchInfo){ 
	  if (errorPrint > 1) 
	    printf("%s\n", params[0]); 
	}//if
	else
	  printf("%s", params[0]);
      }//if
    }//if 
    else {
      // reporting of actual errors
      char line[256];
      const char *msg = bpatch->getEnglishErrorString(num);
      bpatch->formatErrorString(line, sizeof(line), msg, params);      
      if (num != expectError) {
	printf("Error #%d (level %d): %s\n", num, level, line);        
	// We consider some errors fatal.
	if (num == 101)
	  exit(-1);
      }//if
    }//else
}//errorFunc1()

// We've a null error function when we don't want to display an error
/* OLD void errorFuncNull(BPatchErrorLevel level, int num, const char **params) */
void errorFuncNull(BPatchErrorLevel level, int num,  const char* const* params)
{
  // It does nothing.
}//errorFuncNull()
 
// END OF TEST3 code
// Constraints for instrumentation. Returns true for those modules that 
// shouldn't be instrumented. 
int moduleConstraint(char *fname){ // fname is the name of module/file 
  int len = strlen(fname);

  if (areFileIncludeExcludeListsEmpty()) 
  { // there are no user sepecified constraints on modules. Use our default 
    // constraints 
    if ((strcmp(fname, "DEFAULT_MODULE") == 0) ||
       ((fname[len-2] == '.') && (fname[len-1] == 'c')) || 
       ((fname[len-2] == '.') && (fname[len-1] == 'C')) || 
       ((fname[len-3] == '.') && (fname[len-2] == 'c') && (fname[len-1] == 'c')) || 
       ((fname[len-4] == '.') && (fname[len-3] == 'c') && (fname[len-2] == 'p') && (fname[len-1] == 'p')) || 
       ((fname[len-4] == '.') && (fname[len-3] == 'f') && (fname[len-2] == '9') && (fname[len-1] == '0')) || 
       ((fname[len-4] == '.') && (fname[len-3] == 'F') && (fname[len-2] == '9') && (fname[len-1] == '0')) || 
       ((fname[len-2] == '.') && (fname[len-1] == 'F')) || 
       ((fname[len-2] == '.') && (fname[len-1] == 'f')) || 
       (strcmp(fname, "LIBRARY_MODULE") == 0)){
      /* It is ok to instrument this module. Constraint doesn't exist. */
      return false;
    }//if
    else
      return true;
  } // the selective instrumentation file lists are not empty! 
  else
  { 
    // See if the file should be instrumented 
    if (processFileForInstrumentation(string(fname)))
    { // Yes, it should be instrumented. moduleconstraint should return false! 
      return false; 
    }
    else
    { // No, the file should not be instrumented. Constraint exists return true
      return true; 
    }

  }
}//moduleConstraint()

// Constriant for routines. The constraint returns true for those routines that 
// should not be instrumented.
int routineConstraint(char *fname){ // fname is the function name
  if ((strncmp(fname, "Tau", 3) == 0) ||
            (strncmp(fname, "Profiler", 8) == 0) ||
            (strstr(fname, "FunctionInfo") != 0) ||
            (strncmp(fname, "RtsLayer", 8) == 0) ||
            (strncmp(fname, "DYNINST", 7) == 0) ||
            (strncmp(fname, "PthreadLayer", 12) == 0) ||
	    (strncmp(fname, "threaded_func", 13) == 0) ||
            (strncmp(fname, "targ8", 5) == 0) ||
            (strncmp(fname, "The", 3) == 0)){
    return true; // Don't instrument 
  }//if
  else{ // Should the routine fname be instrumented?
    if (instrumentEntity(string(fname))){ // Yes it should be instrumented. Return false
      return false; 
    }//if
    else{ // No. The selective instrumentation file says: don't instrument it
      return true;
    }//else
  }//else
}//routineConstraint()

// 
// check if the application has an MPI library routine MPI_Comm_rank
// 
int checkIfMPI(BPatch_image * appImage, BPatch_function * & mpiinit,
		BPatch_function * & mpiinitstub, bool binaryRewrite){

  mpiinit 	= tauFindFunction(appImage, "MPI_Comm_rank");
  mpiinitstub 	= tauFindFunction(appImage, "TauMPIInitStub");

  if (mpiinitstub == (BPatch_function *) NULL)
    printf("*** TauMPIInitStub not found! \n");
  
  if (mpiinit == (BPatch_function *) NULL) {
    dprintf("*** MPI_Comm_rank not found looking for PMPI_Comm_rank...\n");
    if (!binaryRewrite) {
      //mpiinit = tauFindFunction(appImage, "PMPI_Comm_rank");
      mpiinit = tauFindFunction(appImage, "mpi_comm_rank_");
    }
  }//if
  
  if (mpiinit == (BPatch_function *) NULL) { 
    dprintf("*** This is not an MPI Application! \n");
    return 0;  // It is not an MPI application
  }//if
  else
    return 1;   // Yes, it is an MPI application.
}//checkIfMPI()


int tauRewriteBinary(BPatch *bpatch_global, const char *mutateeName, char *outfile, char* libname)
{
  using namespace std;
  BPatch_function *mpiinit;
  BPatch_function *mpiinitstub;

  dprintf("Inside tauRewriteBinary, name=%s, out=%s\n", mutateeName, outfile);
  BPatch_binaryEdit* mutateeAddressSpace = bpatch_global->openBinary(mutateeName, false);

  bpatch_global->setLivenessAnalysis(false);
  mutateeAddressSpace->allowTraps(false);
  BPatch_image* mutateeImage = mutateeAddressSpace->getImage();
  BPatch_Vector<BPatch_function*>* allFuncs = mutateeImage->getProcedures();

  bool result = mutateeAddressSpace->loadLibrary(libname);
  assert(result);
  BPatch_function* entryTrace = tauFindFunction(mutateeImage, "traceEntry");
  BPatch_function* exitTrace = tauFindFunction(mutateeImage, "traceExit");
  BPatch_function* setupFunc = tauFindFunction(mutateeImage, "my_otf_init");
  BPatch_function* cleanupFunc = tauFindFunction(mutateeImage, "my_otf_cleanup");
  BPatch_function* mainFunc = tauFindFunction(mutateeImage, "main");
  name_reg = tauFindFunction(mutateeImage, "trace_register_func");

  if(!mainFunc)
  {
    fprintf(stderr, "Couldn't find main(), aborting\n");
    return -1;
  }
  if(!entryTrace || !exitTrace || !setupFunc || !cleanupFunc)
  {
    fprintf(stderr, "Couldn't find OTF functions, aborting\n");
    return -1;
  }

  BPatch_Vector<BPatch_point*>* mainEntry = mainFunc->findPoint(BPatch_entry);
  assert(mainEntry);
  assert(mainEntry->size());
  assert((*mainEntry)[0]);

  mutateeAddressSpace->beginInsertionSet();

  BPatch_constExpr isMPI(checkIfMPI(mutateeImage, mpiinit, mpiinitstub, true));
  BPatch_Vector<BPatch_snippet*> init_params;
  init_params.push_back(&isMPI);
  BPatch_funcCallExpr setup_call(*setupFunc, init_params);
  funcNames.push_back(&setup_call);

  for (BPatch_Vector<BPatch_function*>::iterator it=allFuncs->begin();
  it != allFuncs->end(); it++)
  {
    char fname[FUNCNAMELEN];
    (*it)->getName(fname, FUNCNAMELEN);
    dprintf("Processing %s...\n", fname); 
    if (!routineConstraint(fname)) { // ok to instrument
      insertTrace(*it, mutateeAddressSpace, entryTrace, exitTrace);
    }
    else {
     dprintf("Not instrumenting %s\n", fname);
    }
  }


  BPatch_sequence sequence(funcNames);
  mutateeAddressSpace->insertSnippet(sequence,
                                     *mainEntry,
                                     BPatch_callBefore,
                                     BPatch_firstSnippet);

  mutateeAddressSpace->finalizeInsertionSet(false, NULL);

  std::string modifiedFileName(outfile);
  chdir("result");
  mutateeAddressSpace->writeFile(modifiedFileName.c_str());
  return 0;
}


//
// entry point 
//
int main(int argc, char **argv){
  int i,j;                                       //for loop variables
  int instrumented=0;                            //count of instrumented functions
  int errflag=0;                                 //determine if error has occured.  default 0
  bool loadlib=false;                            //do we have a library loaded? default false
  char mutname[MUTNAMELEN];                      //variable to hold mutator name (ie tau_run)
  char outfile[MUTNAMELEN];                      // variable to hold output file
  char fname[FUNCNAMELEN], libname[FUNCNAMELEN]; //function name and library name variables
  BPatch_thread *appThread;                      //application thread
  BPatch_function *mpiinit;                      
  BPatch_function *mpiinitstub;
  bpatch = new BPatch;                           //create a new version. 
  string functions;                              //string variable to hold function names 
  // commandline option processing args
  int vflag = 0;
  char *xvalue = NULL;
  char *fvalue = NULL;
  char *ovalue = NULL;
  int index;
  int c;


  bpatch->setTrampRecursive(true); /* enable C++ support */
  // parse the command line arguments--first, there need to be atleast two arguments,
  // the program name (tau_run) and the application it is running.  If there are not
  // at least two arguments, set the error flag.
  if ( argc < 2 )
    errflag=1;
  //now can loop through the options.  If the first character is '-', then we know we have 
  //an option.  Check to see if it is one of our options and process it.  If it is unrecognized,
  //then set the errflag to report an error.  When we come to a non '-' charcter, then we must
  //be at the application name.
  else{
    opterr = 0; 
     
    while ((c = getopt (argc, argv, "vX:o:f:")) != -1)
      switch (c)
      {
        case 'v':
          vflag = 1;
          debugPrint = 1; /* Verbose option set */
          break;
        case 'X':
          xvalue = optarg;
	  loadlib = true; /* load an external measurement library */
	  sprintf(libname,"lib%s.so", &xvalue[3]);
          break;
        case 'f':
          fvalue = optarg; /* choose a selective instrumentation file */
	  processInstrumentationRequests(fvalue);
	  dprintf("Loading instrumentation requests file %s\n", fvalue);
          break;
        case 'o':
          ovalue = optarg;
          binaryRewrite = 1; /* binary rewrite is true */
          strcpy(outfile, ovalue);
          break;
        case '?':
          if (optopt == 'X' || optopt == 'f' || optopt == 'o' )
            fprintf (stderr, "Option -%c requires an argument.\n", optopt);
          else if (isprint (optopt))
            fprintf (stderr, "Unknown option `-%c'.\n", optopt);
          else
            fprintf (stderr,
                        "Unknown option character `\\x%x'.\n",
                        optopt);
	    errflag=1;
          default:
	    errflag=1;
        }
     
      dprintf ("vflag = %d, xvalue = %s, ovalue = %s, fvalue = %s\n",
               vflag, xvalue, ovalue, fvalue);
     
      strncpy(mutname, argv[optind],strlen(argv[optind])+1);
      for (index = optind; index < argc; index++)
        dprintf ("Non-option argument %s\n", argv[index]);
    }

  dprintf("mutatee name = %s\n", mutname);
  
  //did we load a library?  if not, load the default
  if(!loadlib){
    sprintf(libname,"libTAU.so");
    loadlib=true;
  }//if
  else {
    fprintf(stderr, "%s> Loading %s ...\n", mutname, libname);
  }

  //has an error occured in the command line arguments?
  if(errflag){
    fprintf (stderr, "usage: %s [-Xrun<Taulibrary> ] [-v] [-o outfile] [-f <inst_req> ] <application> [args]\n", argv[0]);
    fprintf (stderr, "%s instruments and executes <application> to generate performance data\n", argv[0]);
    fprintf (stderr, "-v is an optional verbose option\n"); 
    fprintf (stderr, "-o <outfile> is for binary rewriting\n");
    fprintf (stderr, "e.g., \n");
    fprintf (stderr, "%%%s -XrunTAU -f sel.dat a.out 100 \n", argv[0]);
    fprintf (stderr, "Loads libTAU.so from $LD_LIBRARY_PATH, loads selective instrumentation requests from file sel.dat and executes a.out \n"); 
    exit (1);
  }//if(errflag)

  // Register a callback function that prints any error messages
  bpatch->registerErrorCallback(errorFunc1);

  dprintf("Before createProcess\n");
  // Specially added to disable Dyninst 2.0 feature of debug parsing. We were
  // getting an assertion failure under Linux otherwise 
  // bpatch->setDebugParsing(false); 
  // removed for DyninstAPI 4.0

  if (binaryRewrite) {
    tauRewriteBinary(bpatch, mutname, outfile, (char *)libname);
    return 0; // exit from the application 
  }
#ifdef TAU_DYNINST41PLUS
  appThread = bpatch->createProcess(argv[optind], (const char **)&argv[optind] , NULL);
#else
  appThread = bpatch->createProcess(argv[optind], &argv[optind] , NULL);
#endif /* TAU_DYNINST41PLUS */
  dprintf("After createProcess\n");

  if (!appThread){ 
    printf("tau_run> createProcess failed\n");
    exit(1);
  }//if

  if (binaryRewrite)
  { // enable dumping 
    appThread->enableDumpPatchedImage();
  }

  //get image
  BPatch_image *appImage = appThread->getImage();
  BPatch_Vector<BPatch_module *> *m = appImage->getModules();

  // Load the TAU library that has entry and exit routines.
  // Do not load the TAU library if we're rewriting the binary. Use LD_PRELOAD
  // instead. The library may be loaded at a different location. 
#ifdef __SP1__
  if (loadlib==true) {
#else /* SP1 */
  if (loadlib == true) {
#endif /* SP1 */
    //try and load the library
    if (appThread->loadLibrary(libname, true) == true){  
      //now, check to see if the library is listed as a module in the
      //application image
      char name[FUNCNAMELEN];
      bool found = false;
      for (i = 0; i < m->size(); i++) {
        (*m)[i]->getName(name, sizeof(name));
        if (strcmp(name, libname) == 0) {
	  found = true;
          break;
        }//if 
      }//for
      if (found) {
  	dprintf("%s loaded properly\n", libname);
      }//if
      else {
	printf("Error in loading library %s\n", libname);
 	exit(1);
      }//else
    }//if
    else{
      printf("ERROR:%s not loaded properly. \n", libname);
      printf("Please make sure that its path is in your LD_LIBRARY_PATH environment variable.\n");
      exit(1);
    }//else
  }//loadlib == true
 
  BPatch_function *inFunc;
  BPatch_function *enterstub = tauFindFunction(appImage, "TauRoutineEntry");
  BPatch_function *exitstub = tauFindFunction(appImage, "TauRoutineExit");
  BPatch_function *terminationstub = tauFindFunction(appImage, "TauProgramTermination");
  BPatch_Vector<BPatch_snippet *> initArgs;
  
  char modulename[256];
  for (j=0; j < m->size(); j++) {
    sprintf(modulename, "Module %s\n", (*m)[j]->getName(fname, FUNCNAMELEN));
    BPatch_Vector<BPatch_function *> *p = (*m)[j]->getProcedures();
    dprintf("%s", modulename);

    if (!moduleConstraint(fname)) { // constraint 
      for (i=0; i < p->size(); i++) {
        // For all procedures within the module, iterate  
        (*p)[i]->getName(fname, FUNCNAMELEN);
        dprintf("Name %s\n", fname);
        if (routineConstraint(fname)){ // The above procedures shouldn't be instrumented
           dprintf("don't instrument %s\n", fname);
        }//if
        else{ // routines that are ok to instrument
	  functions.append("|");
	  functions.append(fname);
	}//else
      }//for
    }//if(!moduleConstraint)
  }//for - Module 

  // form the args to InitCode
  BPatch_constExpr funcName(functions.c_str());

  // We need to find out if the application is an MPI app. If it is, we 
  // should send it 1 for isMPI so TAU_PROFILE_SET_NODE needn't be executed.
  // If, however, it is a sequential program then it should be sent 0 for isMPI.
 
  // When we look for MPI calls, we shouldn't display an error message for
  // not find MPI_Comm_rank in the case of a sequential app. So, we turn the
  // Error callback to be Null and turn back the error settings later. This
  // way, it works for both MPI and sequential tasks. 
  
  bpatch->registerErrorCallback(errorFuncNull); // turn off error reporting
  BPatch_constExpr isMPI(checkIfMPI(appImage, mpiinit, mpiinitstub,binaryRewrite));
  bpatch->registerErrorCallback(errorFunc1); // turn it back on

  initArgs.push_back(&funcName);
  initArgs.push_back(&isMPI);

  Initialize(appThread, appImage, initArgs);
  dprintf("Did Initialize\n");

  // In our tests, the routines started execution concurrently with the 
  // one time code. To avoid this, we first start the one time code and then
  // iterate through the list of routines to select for instrumentation and
  // instrument these. So, we need to iterate twice. 

  for (j=0; j < m->size(); j++) {
    sprintf(modulename, "Module %s\n", (*m)[j]->getName(fname, FUNCNAMELEN));
    BPatch_Vector<BPatch_function *> *p = (*m)[j]->getProcedures();
    dprintf("%s", modulename);

    if (!moduleConstraint(fname)) { // constraint
      for (i=0; i < p->size(); i++){
        // For all procedures within the module, iterate
        (*p)[i]->getName(fname, FUNCNAMELEN);
        dprintf("Name %s\n", fname);
  	if (routineConstraint(fname)){ // The above procedures shouldn't be instrumented
           dprintf("don't instrument %s\n", fname);
        } // Put the constraints above 
        else{ // routines that are ok to instrument
 	  dprintf("Assigning id %d to %s\n", instrumented, fname);
          instrumented ++;
          BPatch_Vector<BPatch_snippet *> *callee_args = new BPatch_Vector<BPatch_snippet *>();
          BPatch_constExpr *constExpr = new BPatch_constExpr(instrumented);
	  callee_args->push_back(constExpr);
    
          inFunc = (*p)[i];
          dprintf("Instrumenting-> %s Entry\n", fname);	  
          invokeRoutineInFunction(appThread, appImage, inFunc, BPatch_entry, enterstub, callee_args);
          dprintf("Instrumenting-> %s Exit...", fname);	  
          invokeRoutineInFunction(appThread, appImage, inFunc, BPatch_exit, exitstub, callee_args);
	  dprintf("Done\n");
          delete callee_args;
          delete constExpr;
        }//else -- routines that are ok to instrument
      }//for -- procedures 
    }//if -- module constraint
  }//for -- modules


#ifdef __SP1__
  BPatch_function *exitpoint = tauFindFunction(appImage, "exit");
#else /* SP1 */
  BPatch_function *exitpoint = tauFindFunction(appImage, "_exit");
#endif /* SP1 */

  if (exitpoint == NULL) {
    fprintf(stderr, "UNABLE TO FIND exit() \n");
    // exit(1);
  }
  else {
    // When _exit is invoked, call TauProgramTermination routine 
    BPatch_Vector<BPatch_snippet *> *exitargs = new BPatch_Vector<BPatch_snippet *>();
    BPatch_constExpr *Name = new BPatch_constExpr("_exit");
    exitargs->push_back(Name);
    invokeRoutineInFunction(appThread, appImage, exitpoint, BPatch_entry, terminationstub , exitargs);
    delete exitargs;
    delete Name;
  }//else

  if (mpiinit == NULL) { 
    dprintf("*** MPI_Comm_rank not found. This is not an MPI Application! \n");
  }
  else { // we've found either MPI_Comm_rank or PMPI_Comm_rank! 
   dprintf("FOUND MPI_Comm_rank or PMPI_Comm_rank! \n");
   BPatch_Vector<BPatch_snippet *> *mpistubargs = new BPatch_Vector<BPatch_snippet *>();
   BPatch_paramExpr paramRank(1);
   
   mpistubargs->push_back(&paramRank);
   invokeRoutineInFunction(appThread, appImage, mpiinit, BPatch_exit, mpiinitstub, mpistubargs);
   delete mpistubargs;
  }

  /* check to see if we have to rewrite the binary image */ 
  if (binaryRewrite)
  {
    char * directory = appThread->dumpPatchedImage(outfile);
    /* see if it was rewritten properly */ 
    if (directory) 
    {
      printf("The instrumented executable image is stored in %s directory\n",
        directory);
    }
    else
    {
      fprintf(stderr, "Error: Binary rewriting did not work: No directory name \
returned\n\nIf you are using Dyninst 5.2 this feature is no longer \
supported and \
tau_run will run the application using dynamic instrumentation....\n");
    }
    delete bpatch;
    return 0;
  }
   

  dprintf("Executing...\n");
  appThread->continueExecution();
  
  while (!appThread->isTerminated()){        
    bpatch->waitForStatusChange();
    sleep(1);
  }//while

  if (appThread->isTerminated()){
    dprintf ("End of application\n");
  }//if

  //cleanup
  delete bpatch;
  return 0;
}// EOF tau_run.cpp
