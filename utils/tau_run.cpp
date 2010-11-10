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
#include "BPatch_process.h"
#include "BPatch_snippet.h" 
#include "BPatch_statement.h" 

//#include <iostream.h>
//#include <stdio.h>
#include <string.h>
#include <unistd.h>

#ifdef i386_unknown_nt4_0
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#endif


#include "tau_platforms.h"
#include "tau_instrument.dyn.h"

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
extern int processInstrumentationRequests(char *fname, vector<tauInstrument *>& instrumentList);
extern bool areFileIncludeExcludeListsEmpty(void);
extern bool processFileForInstrumentation(const string& file_name); 
extern void printExcludeList();
extern bool instrumentEntity(const string& function_name);
extern bool matchName(const string& str1, const string& str2);

/* prototypes for routines below */
int getFunctionFileLineInfo(BPatch_image* mutateeAddressSpace, 
			    BPatch_function *f, char *newname);


/* re-writer */
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

// gets information (line number, filename, and column number) about
// the instrumented loop and formats it properly.
void getLoopFileLineInfo(BPatch_image* mutateeImage,
			 BPatch_flowGraph* cfGraph,
			 BPatch_basicBlockLoop* loopToInstrument,
			 BPatch_function *f,
			 char *newname)
{

  const char *filename;
  char fname[1024];
  const char *typeName;
  bool info1, info2;
  int row1, col1, row2, col2;
  char varTypes[256];
  char *tempString;
  BPatch_type *returnType;

  BPatch_Vector<BPatch_point*>* loopStartInst = cfGraph->findLoopInstPoints(BPatch_locLoopStartIter, loopToInstrument);
  BPatch_Vector<BPatch_point*>* loopExitInst = cfGraph->findLoopInstPoints(BPatch_locLoopEndIter, loopToInstrument);
  //BPatch_Vector<BPatch_point*>* loopExitInst = cfGraph->findLoopInstPoints(BPatch_locLoopExit, loopToInstrument);
  

  unsigned long baseAddr = (unsigned long)(*loopStartInst)[0]->getAddress();
  unsigned long lastAddr = (unsigned long)(*loopExitInst)[loopExitInst->size()-1]->getAddress();
  printf("size of lastAddr = %d: baseAddr = %uld, lastAddr = %uld\n", loopExitInst->size(), baseAddr, lastAddr);



  f->getName(fname, 1024);

  returnType = f->getReturnType();
 
  if(returnType)
    {
      typeName = returnType->getName();
    }
      else
	typeName = "void";
 
  BPatch_Vector< BPatch_statement > lines;
  BPatch_Vector< BPatch_statement > linesEnd;

  info1 = mutateeImage->getSourceLines(baseAddr, lines);
  
  if(info1)
    {
      filename = lines[0].fileName();
      row1 = lines[0].lineNumber();
      col1 = lines[0].lineOffset();
      if (col1 < 0) col1 = 0;
      
      //      info2 = mutateeImage->getSourceLines((unsigned long) (lastAddr -1), lines);



      // This following section is attempting to remedy the limitations of getSourceLines
      // for loops. As the program goes through the loop, the resulting lines go from the 
      // loop head, through the instructions present in the loop, to the last instruction
      // in the loop, back to the loop head, then to the next instruction outside of the
      // loop. What this section does is starts at the last instruction in the loop, then
      // goes through the addresses until it reaches the next instruction outside of the 
      // loop. We then bump back a line. This is not a perfect solution, but we will work
      // with the Dyninst team to find something better.
      info2 = mutateeImage->getSourceLines((unsigned long) lastAddr, linesEnd );
      printf("size of linesEnd = %d\n", linesEnd.size());
      int i;
      for(i=0; i < linesEnd.size(); i++) { 
        printf("row=%d, col=%d\n", linesEnd[i].lineNumber(), linesEnd[i].lineOffset());
      }

#ifdef OLD
      unsigned long q = lastAddr + 1;

      // Goes through all addresses starting at the last instructions in the loop until 
      // we loop back around to the loop head.
      do{
	lines.clear();
	mutateeImage->getSourceLines((unsigned long) (q) , lines );
	q++;
      }while(lines[0].lineNumber() != row1);
      
      // Goes through all the addresses that are located in the loop head.
      do{
	lines.clear();
	mutateeImage->getSourceLines((unsigned long) (q) , lines );
	q++;
      }while(lines[0].lineNumber() == row1);
      if( lines[0].lineNumber() != row1 ) info2 = true;
#endif /* OLD */



      if (info2) {
	row2 = linesEnd[0].lineNumber(); 
	col2 = linesEnd[0].lineOffset();
	if (col2 < 0) col2 = 0;
	sprintf(newname, "Loop: %s %s() [{%s} {%d,%d}-{%d,%d}]", typeName, fname, filename, row1, col1, row2, col2);
      } else {
	sprintf(newname, "Loop: %s %s() [{%s} {%d,%d}]", typeName, fname, filename, row1, col1);
      }
    }
  else
    {
      strcpy(newname, fname);	
    }
    
}


// InsertTrace function for loop-level instrumentation.
// Bug exists at the moment that the second line number is 
// the last command at the outermost loop's level. So, if the outer
// loop has a nested loop inside, with blank lines afterwards,
// only the lines from the beginning of the outer loop to the 
// beginning of the outer loop are counted. 
void insertTrace(BPatch_function* functionToInstrument, 
                 BPatch_addressSpace* mutatee, 
                 BPatch_function* traceEntryFunc, 
		 BPatch_function* traceExitFunc,
		 BPatch_flowGraph* cfGraph,
		 BPatch_basicBlockLoop* loopToInstrument)
{
  char name[1024];
  char modname[1024];
  int i;



  functionToInstrument->getModuleName(modname, 1024);

  getLoopFileLineInfo(mutatee->getImage(), cfGraph, loopToInstrument, functionToInstrument, name);

  BPatch_module *module = functionToInstrument->getModule();

  if (strstr(modname, "libdyninstAPI_RT"))
    return;

  //  functionToInstrument->getName(name, 1024);

  int id = addName(name);
  BPatch_Vector<BPatch_snippet *> traceArgs;
  traceArgs.push_back(new BPatch_constExpr(id));
      
  BPatch_Vector<BPatch_point*>* loopEntr = cfGraph->findLoopInstPoints(BPatch_locLoopEntry, loopToInstrument);
  BPatch_Vector<BPatch_point*>* loopExit = cfGraph->findLoopInstPoints(BPatch_locLoopExit, loopToInstrument);
    

  BPatch_Vector<BPatch_snippet *> entryTraceArgs;
  entryTraceArgs.push_back(new BPatch_constExpr(id));
  entryTraceArgs.push_back(new BPatch_constExpr(name));


  BPatch_funcCallExpr entryTrace(*traceEntryFunc, entryTraceArgs);
  BPatch_funcCallExpr exitTrace(*traceExitFunc, traceArgs);
      
  if (loopEntr->size() == 0) {
    printf("Failed to instrument loop entry in %s\n", name);
  }
  else {
    for (i =0; i < loopEntr->size(); i++) {
      mutatee->insertSnippet(entryTrace, loopEntr[i], BPatch_callBefore, BPatch_lastSnippet);
    }
  }

  if (loopExit->size() == 0) {
    printf("Failed to instrument loop exit in %s\n", name);
  }
  else {
    for (i =0; i < loopExit->size(); i++) {
      mutatee->insertSnippet(exitTrace, loopExit[i], BPatch_callBefore, BPatch_lastSnippet);
    }
  }
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

  //functionToInstrument->getName(name, 1024);
  getFunctionFileLineInfo(mutatee->getImage(), functionToInstrument, name);

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

BPatch_function * tauFindFunction (BPatch_image *appImage, const char * functionName)
{
  // Extract the vector of functions 
  BPatch_Vector<BPatch_function *> found_funcs;
  if ((NULL == appImage->findFunction(functionName, found_funcs, false, true, true)) || !found_funcs.size()) {
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
BPatchSnippetHandle *invokeRoutineInFunction(BPatch_process *appThread,
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
// invokeRoutineInFunction calls routine "callee" with no arguments when 
// Function "function" is invoked at the point given by location
//
BPatchSnippetHandle *invokeRoutineInFunction(BPatch_process *appThread,
					     BPatch_image *appImage, BPatch_Vector<BPatch_point *> points,
					     BPatch_function *callee, 
					     BPatch_Vector<BPatch_snippet *> *callee_args){
  
  // First create the snippet using the callee and the args 
  const BPatch_snippet *snippet = new BPatch_funcCallExpr(*callee, *callee_args);
  if (snippet == NULL) {
    fprintf(stderr, "Unable to create snippet to call callee\n");
    exit(1);
  }//if

  if(points.size()) {
    // Insert the given snippet at the given point 
    appThread->insertSnippet(*snippet, points, BPatch_callAfter);
  }//if
  delete snippet; // free up space
}//invokeRoutineInFunction()

// 
// Initialize calls TauInitCode, the initialization routine in the user
// application. It is executed exactly once, before any other routine.
//
void Initialize(BPatch_process *appThread, BPatch_image *appImage, 
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
	  //((fname[len-3] == '.') && (fname[len-2] == 's') && (fname[len-1] == 'o'))|| 
	  (strcmp(fname, "LIBRARY_MODULE") == 0)){
	/* It is ok to instrument this module. Constraint doesn't exist. */
	// Wait: first check if it has libTAU* in the name!
	if (strncmp(fname, "libTAU", 6) == 0)  {
	  return true;  /* constraint applies - do not instrument! */
	}
	else {
	  return false; /* ok to instrument */
	}
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

// Constraint for routines. The constraint returns true for those routines that 
// should not be instrumented.
int routineConstraint(char *fname){ // fname is the function name

  bool doNotInst = false;
  if ((strncmp(fname, "Tau", 3) == 0) ||
      (strncmp(fname, "Profiler", 8) == 0) ||
      (strstr(fname, "FunctionInfo") != 0) ||
      (strncmp(fname, "RtsLayer", 8) == 0) ||
      (strncmp(fname, "DYNINST", 7) == 0) ||
      (strncmp(fname, "PthreadLayer", 12) == 0) ||
      (strncmp(fname, "threaded_func", 13) == 0) ||
      (strncmp(fname, "targ8", 5) == 0) ||
      (strncmp(fname, "__intel_", 8) == 0) ||
      (strncmp(fname, "_intel_", 7) == 0) ||
      (strncmp(fname, "The", 3) == 0) ||
      // The following functions show up in static executables
      (strncmp(fname, "__mmap", 6) == 0) ||
      (strncmp(fname, "_IO_printf", 10) == 0) ||
      (strncmp(fname, "__write", 7) == 0) ||
      (strncmp(fname, "__munmap", 8) == 0) || 
      (strstr(fname, "_L_lock") != 0) ||
      (strstr(fname, "_L_unlock") != 0) ) {


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

bool findFuncOrCalls(std::vector<const char *> names, BPatch_Vector<BPatch_point *> &points,
		     BPatch_image *appImage, BPatch_procedureLocation loc = BPatch_locEntry)
{
  BPatch_function *func = NULL;
  for (std::vector<const char *>::iterator i = names.begin(); i != names.end(); i++)
    {
      BPatch_function *f = tauFindFunction(appImage, *i);
      if (f && f->getModule()->isSharedLib()) {
	func = f;
	break;
      }
    }
  if (func) {
    BPatch_Vector<BPatch_point *> *fpoints = func->findPoint(loc);
    BPatch_Vector<BPatch_point *>::iterator k;
    if (fpoints && fpoints->size()) {
      for (k = fpoints->begin(); k != fpoints->end(); k++) {
	points.push_back(*k);
      }
      return true;
    }
  }

  //Moderately expensive loop here.  Perhaps we should make a name->point map first
  // and just do lookups through that.
  BPatch_Vector<BPatch_function *> *all_funcs = appImage->getProcedures();
  int initial_points_size = points.size();
  for (std::vector<const char *>::iterator i = names.begin(); i != names.end(); i++) {
    BPatch_Vector<BPatch_function *>::iterator j;
    for (j = all_funcs->begin(); j != all_funcs->end(); j++)
      {
	BPatch_function *f = *j;
	if (f->getModule()->isSharedLib())
	  continue;
	BPatch_Vector<BPatch_point *> *fpoints = f->findPoint(BPatch_locSubroutine);
	if (!fpoints || !fpoints->size())
	  continue;
	BPatch_Vector<BPatch_point *>::iterator j;
	for (j = fpoints->begin(); j != fpoints->end(); j++) {
	  std::string callee = (*j)->getCalledFunctionName();
	  if (callee == std::string(*i)) {
	    points.push_back(*j);
	  }
	}
      }
    if (points.size() != initial_points_size)
      return true;
  }

  return false;
}

bool findFuncOrCalls(const char *name, BPatch_Vector<BPatch_point *> &points,
		     BPatch_image *image, BPatch_procedureLocation loc = BPatch_locEntry)
{
  std::vector<const char *> v;
  v.push_back(name);
  return findFuncOrCalls(v, points, image, loc);
}

// 
// check if the application has an MPI library routine MPI_Comm_rank
// 
int checkIfMPI(BPatch_image * appImage, BPatch_Vector<BPatch_point *> &mpiinit,
	       BPatch_function * & mpiinitstub, bool binaryRewrite)
{
  std::vector<const char *> init_names;
  init_names.push_back("MPI_Init");
  init_names.push_back("PMPI_Init");
  bool ismpi = findFuncOrCalls(init_names, mpiinit, appImage, BPatch_locExit);

  mpiinitstub 	= tauFindFunction(appImage, "TauMPIInitStubInt");
  if (mpiinitstub == (BPatch_function *) NULL)
    printf("*** TauMPIInitStubInt not found! \n");
  
  if (!ismpi) {
    dprintf("*** This is not an MPI Application! \n");
    return 0;  // It is not an MPI application
  }
  else
    return 1;   // Yes, it is an MPI application.
}//checkIfMPI()

/* We create a new name that embeds the file and line information in the name */
int getFunctionFileLineInfo(BPatch_image* mutateeAddressSpace, 
			    BPatch_function *f, char *newname)
{
  bool info1, info2;
  unsigned long baseAddr,lastAddr;
  char fname[1024];
  const char *filename;
  int row1, col1, row2, col2;
  char varTypes[256];
  char *tempString;
  BPatch_type *returnType;
  const char *typeName;

  baseAddr = (unsigned long)(f->getBaseAddr());
  lastAddr = baseAddr + f->getSize();
  BPatch_Vector< BPatch_statement > lines;
  f->getName(fname, 1024);
  
  returnType = f->getReturnType();

 
  if(returnType)
    {
      typeName = returnType->getName();
    }
      else
	typeName = "void";


  info1 = mutateeAddressSpace->getSourceLines((unsigned long) baseAddr, lines);

  if (info1) {
    filename = lines[0].fileName();
    row1 = lines[0].lineNumber();
    col1 = lines[0].lineOffset();
    if (col1 < 0) col1 = 0;
    info2 = mutateeAddressSpace->getSourceLines((unsigned long) (lastAddr -1), lines);
    if (info2) {
      row2 = lines[1].lineNumber();
      col2 = lines[1].lineOffset();
      if (col2 < 0) col2 = 0;
      sprintf(newname, "%s %s() [{%s} {%d,%d}-{%d,%d}]", typeName, fname, filename, row1, col1, row2, col2);
    } else {
      sprintf(newname, "%s %s() [{%s} {%d,%d}]", typeName, fname, filename, row1, col1);
    }
  }
  else
    strcpy(newname, fname);
}


char * getGCCHOME(void) {
  FILE *fp;
  char command[512];
  char home[512];
  char * ret;
  char *gcchome = getenv("TAU_GCC_HOME");
  if (gcchome != (char *) NULL) {
    return gcchome;
  }

  /* if TAU_GCC_HOME is not set use this */
  strcpy(command, " tau_cc.sh -show | awk '{c=split($0, s); for(n=1; n<=c; ++n) print s[n] }' | grep gcc | grep '^.L' | sed -e 's/-L//'");

  fp=popen(command, "r");

  if (fp == NULL) {
    perror("Error launching tau_cc.sh to get TAU_GCC_HOME");
    return 0;
  }

  if ((ret = fgets(home, 512, fp)) != NULL) {
    int len = strlen(home); 
    if (home[len - 1] == '\n') { 
      home[len - 1 ] = '\0';
    }
    return strdup(home);
  } else {
    perror("Error reading from pipe to get TAU_GCC_HOME:");
    return 0;
  }
  pclose(fp);
}

bool loadDependentLibraries(BPatch_binaryEdit *bedit) {
  //old:    const string GCCHOME = "/usr/lib/gcc/i586-redhat-linux/4.4.1";
  const string GCCHOME = string(getGCCHOME());

  // Order of load matters, just like command line arguments to a standalone linker

  // Load C++ Library
  string cpplib = GCCHOME + "/libstdc++.a";
  if( !bedit->loadLibrary(cpplib.c_str()) ) {
    return false;
  }

  // Load pthreads to be safe
  if( bedit->isMultiThreadCapable() ) {
    if( !bedit->loadLibrary("libpthread.a") ) {
      return false;
    }
  }

  // Load C library
  if( !bedit->loadLibrary("libc.a") ) {
    return false;
  }

  // Load GCC support library
  string suplib = GCCHOME + "/libgcc.a";
  if( !bedit->loadLibrary(suplib.c_str()) ) {
    return false;
  }

  // Load the GCC exception library
  string ehlib = GCCHOME + "/libgcc_eh.a";
  if( !bedit->loadLibrary(ehlib.c_str()) ) {
    return false;
  }

  return true;
}


int tauRewriteBinary(BPatch *bpatch, const char *mutateeName, char *outfile, char* libname, char *staticlibname, char *staticmpilibname)
{
  using namespace std;
  BPatch_Vector<BPatch_point *> mpiinit;
  BPatch_function *mpiinitstub;


  dprintf("Inside tauRewriteBinary, name=%s, out=%s\n", mutateeName, outfile);
  BPatch_binaryEdit* mutateeAddressSpace = bpatch->openBinary(mutateeName, false);

  if( mutateeAddressSpace == NULL ) {
    fprintf(stderr, "Failed to open binary %s\n",
	    mutateeName);
    return -1;
  }

  BPatch_image* mutateeImage = mutateeAddressSpace->getImage();
  BPatch_Vector<BPatch_function*>* allFuncs = mutateeImage->getProcedures();
  bool isStaticExecutable; 


#ifdef TAU_DYNINST_STATIC_REWRITING_UNSUPPORTED
  isStaticExecutable = false;
#else /* TAU_DYNINST_STATIC_REWRITING_UNSUPPORTED */
  isStaticExecutable = mutateeAddressSpace->isStaticExecutable();
#endif  /* TAU_DYNINST_STATIC_REWRITING_UNSUPPORTED */

  if( isStaticExecutable ) {
    bool result1 = mutateeAddressSpace->loadLibrary(staticlibname);
    bool result2 = mutateeAddressSpace->loadLibrary(staticmpilibname);
    assert(result1);
  }else{
    bool result = mutateeAddressSpace->loadLibrary(libname);
    assert(result);
  }

  BPatch_function* entryTrace = tauFindFunction(mutateeImage, "traceEntry");
  BPatch_function* exitTrace = tauFindFunction(mutateeImage, "traceExit");
  BPatch_function* setupFunc = tauFindFunction(mutateeImage, "tau_dyninst_init");
  BPatch_function* cleanupFunc = tauFindFunction(mutateeImage, "tau_dyninst_cleanup");
  BPatch_function* mainFunc = tauFindFunction(mutateeImage, "main");
  name_reg = tauFindFunction(mutateeImage, "trace_register_func");

  // This heuristic guesses that debugging info. is available if main
  // is not defined in the DEFAULT_MODULE
  bool hasDebuggingInfo = false;
  BPatch_module *mainModule = mainFunc->getModule();
  if( NULL != mainModule ) {
    char moduleName[MUTNAMELEN];
    mainModule->getName(moduleName, MUTNAMELEN);
    if( strcmp(moduleName, "DEFAULT_MODULE") != 0 ) hasDebuggingInfo = true;
  }

  if(!mainFunc)
    {
      fprintf(stderr, "Couldn't find main(), aborting\n");
      return -1;
    }
  if(!entryTrace || !exitTrace || !setupFunc || !cleanupFunc )
    {
      fprintf(stderr, "Couldn't find OTF functions, aborting\n");
      return -1;
    }

  BPatch_Vector<BPatch_point*>* mainEntry = mainFunc->findPoint(BPatch_entry);
  assert(mainEntry);
  assert(mainEntry->size());
  assert((*mainEntry)[0]);

  mutateeAddressSpace->beginInsertionSet();

  int ismpi = checkIfMPI(mutateeImage, mpiinit, mpiinitstub, true);
  BPatch_constExpr isMPI(ismpi);
  BPatch_Vector<BPatch_snippet*> init_params;
  init_params.push_back(&isMPI);
  BPatch_funcCallExpr setup_call(*setupFunc, init_params);
  funcNames.push_back(&setup_call);

  if (ismpi) {
    /*
      char *mpilib = "libTAUsh-icpc-mpi-pdt.so";
      if( isStaticExecutable ) {
      mpilib = "libtaumpihook.a";
      }
      bool result = mutateeAddressSpace->loadLibrary(mpilib);
      assert(result);
    */

    
    //Create a snippet that calls TauMPIInitStub with the rank after MPI_Init
    //   BPatch_function *mpi_rank = tauFindFunction(mutateeImage, "taumpi_getRank");
    BPatch_function *mpi_rank = tauFindFunction(mutateeImage, "TauGetMpiRank");
    assert(mpi_rank);
    BPatch_Vector<BPatch_snippet *> rank_args;
    BPatch_funcCallExpr getrank(*mpi_rank, rank_args);
    BPatch_Vector<BPatch_snippet *> mpiinitargs;
    mpiinitargs.push_back(&getrank);
    BPatch_funcCallExpr initmpi(*mpiinitstub, mpiinitargs);
    
    mutateeAddressSpace->insertSnippet(initmpi, mpiinit, BPatch_callAfter, BPatch_firstSnippet);
  }

  for (BPatch_Vector<BPatch_function*>::iterator it=allFuncs->begin();
       it != allFuncs->end(); it++)
    {
      char fname[FUNCNAMELEN];
      (*it)->getName(fname, FUNCNAMELEN);
      dprintf("Processing %s...\n", fname);

      bool okayToInstr = true;
      bool instRoutineAtLoopLevel = false;



      // Goes through the vector of tauInstrument to check that the 
      // current routine is one that has been passed in the selective instrumentation
      // file
      for(std::vector<tauInstrument*>::iterator instIt=instrumentList.begin();
	  instIt != instrumentList.end(); instIt++)
	{
	  if( (*instIt)->getRoutineSpecified())
	    {
	      const char * instRName = (*instIt)->getRoutineName().c_str();
	      dprintf("Examining %s... \n", instRName);

	      //if( strcmp((char *)instRName, fname) != 0 && strcmp(fname, "main") != 0 && strcmp(instRName, "#") != 0 && fname[0] != '_')
                if (matchName((*instIt)->getRoutineName(), string(fname)))
		{
		  instRoutineAtLoopLevel = true;
	          dprintf("True: instrumenting %s at the loop level\n", instRName);
		}
	    }
	}




      // STATIC EXECUTABLE FUNCTION EXCLUDE
      // Temporarily avoid some functions -- this isn't a solution 
      // -- it appears that something like moduleConstraint would work 
      // well here
      if( isStaticExecutable ) {
	// Always instrument _fini to ensure instrumentation disabled correctly
        if( hasDebuggingInfo && strcmp(fname, "_fini") != 0) {
	  BPatch_module *funcModule = (*it)->getModule();
	  if( funcModule != NULL ) {
	    char moduleName[MUTNAMELEN];
	    funcModule->getName(moduleName, MUTNAMELEN);
	    if( strcmp(moduleName, "DEFAULT_MODULE") == 0 ) okayToInstr = false;
	  }
        }
      }


      if (okayToInstr && !routineConstraint(fname) ) { // ok to instrument

	insertTrace(*it, mutateeAddressSpace, entryTrace, exitTrace);
      }
      else {
	dprintf("Not instrumenting %s\n", fname);

      }

      if(okayToInstr && !routineConstraint(fname) && instRoutineAtLoopLevel) // Only occurs when we've defined that the selective file is for loop instrumentation
	{
	  dprintf("Generating CFG at loop level: %s\n", fname);
          BPatch_flowGraph *flow = (*it)->getCFG();
          BPatch_Vector<BPatch_basicBlockLoop*> basicLoop;
	  dprintf("Generating outer loop info : %s\n", fname);
          flow->getOuterLoops(basicLoop);
	  dprintf("Before instrumenting at loop level: %s\n", fname);

	  for(BPatch_Vector<BPatch_basicBlockLoop*>::iterator loopIt = basicLoop.begin(); 
	      loopIt != basicLoop.end(); loopIt++)
	    {

	      dprintf("Instrumenting at the loop level: %s\n", fname);
	      insertTrace(*it, mutateeAddressSpace, entryTrace, exitTrace, flow, *loopIt);
	    }
	}


    }

  BPatch_sequence sequence(funcNames);
  mutateeAddressSpace->insertSnippet(sequence,
                                     *mainEntry,
                                     BPatch_callBefore,
                                     BPatch_firstSnippet);

  mutateeAddressSpace->finalizeInsertionSet(false, NULL);

  if( isStaticExecutable) {
    bool loadResult = loadDependentLibraries(mutateeAddressSpace);
    if( !loadResult ) {
      fprintf(stderr, "Failed to load dependent libraries need for binary rewrite\n");
      return -1;
    }
  }

  std::string modifiedFileName(outfile);
  chdir("result");
  mutateeAddressSpace->writeFile(modifiedFileName.c_str());
  if (!isStaticExecutable) {  
    unlink(libname); 
    /* remove libTAU.so in the current directory. It interferes */
  }
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
  char staticlibname[FUNCNAMELEN];
  char staticmpilibname[FUNCNAMELEN];
  BPatch_process *appThread;                      //application thread
  BPatch_Vector<BPatch_point *> mpiinit;                      
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



  //bpatch->setTrampRecursive(true); /* enable C++ support */
  bpatch->setSaveFPR(true);

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
     
    while ((c = getopt (argc, argv, "vX:o:f:d:")) != -1)
      switch (c)
	{
        case 'v':
          vflag = 1;
          debugPrint = 1; /* Verbose option set */
          break;
        case 'X':
          xvalue = optarg;
	  loadlib = true; /* load an external measurement library */
          break;
        case 'f':
          fvalue = optarg; /* choose a selective instrumentation file */
	  processInstrumentationRequests(fvalue,instrumentList);
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
    sprintf(staticlibname,"libtau-mpi-pdt.a");
    sprintf(staticlibname,"libTauMpi-mpi-pdt.a");
    sprintf(libname, "libTAU.so");
    loadlib=true;
  }//if
  else {
    sprintf(staticlibname,"lib%s.a", &xvalue[3]);
    sprintf(staticmpilibname,"libTauMpi%s.a", &xvalue[6]);
    dprintf("staticmpilibname = %s\n", staticmpilibname);
    sprintf(libname, "lib%s.so", &xvalue[3]);
    if (xvalue[3] == 'T') {
      fprintf(stderr, "%s> Loading %s ...\n", mutname, libname);
    } else {
      fprintf(stderr, "%s> Loading %s ...\n", mutname, staticlibname);
      fprintf(stderr, "%s> Loading %s ...\n", mutname, staticmpilibname);
    }
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
    tauRewriteBinary(bpatch, mutname, outfile, (char *)libname, (char *)staticlibname, (char *)staticmpilibname);
    return 0; // exit from the application 
  }
#ifdef TAU_DYNINST41PLUS
  appThread = bpatch->processCreate(argv[optind], (const char **)&argv[optind] , NULL);
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
	    // get full source information
	    getFunctionFileLineInfo(appImage, (*p)[i], fname);
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

    if (!mpiinit.size()) { 
      dprintf("*** MPI_Init not found. This is not an MPI Application! \n");
    }
    else { // we've found either MPI_Comm_rank or PMPI_Comm_rank! 
      dprintf("FOUND MPI_Comm_rank or PMPI_Comm_rank! \n");
      BPatch_Vector<BPatch_snippet *> *mpistubargs = new BPatch_Vector<BPatch_snippet *>();
      BPatch_paramExpr paramRank(1);
   
      mpistubargs->push_back(&paramRank);
      invokeRoutineInFunction(appThread, appImage, mpiinit, mpiinitstub, mpistubargs);
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
