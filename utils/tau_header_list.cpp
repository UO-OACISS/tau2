/****************************************************************************
 **                      TAU Portable Profiling Package                     **
 **                      http://www.cs.uoregon.edu/research/tau             **
 *****************************************************************************
 **    Copyright 2008                                                       **
 **    Department of Computer and Information Science, University of Oregon **
 **    Advanced Computing Laboratory, Los Alamos National Laboratory        **
 ****************************************************************************/
/****************************************************************************
 **      File            : tau_headerlist.cpp                               **
 **      Description     : TAU Profiling Package                            **
 **      Author          : Alan Morris                                      **
 **      Contact         : tau-bugs@cs.uoregon.edu                          **
 **      Documentation   : See http://www.cs.uoregon.edu/research/tau       **
 **                                                                         **
 **      Description     : This program outputs the headers included in     **
 **                        a PDB file, excluding system headers             **
 **                                                                         **
 ****************************************************************************/

#include "pdb.h"
#include "pdbRoutine.h"

int mode_print_id = 0;
int mode_show = 0;
int mode_show_ids = 0;

char *target_file = NULL;

static void printIncludes(const pdbFile *f, bool first) {
  bool show = true;
  if (mode_show) {
    show = false;
    if (strcmp(f->name().c_str(), target_file) == 0) {
      show = true;
    }
  }

  pdbFile::incvec i = f->includes();
  for (pdbFile::incvec::iterator it=i.begin(); it!=i.end(); ++it) {
    if (!first && !(*it)->isSystemFile()) { // exclude system files
      // skip over "./"
      const char *ptr = (*it)->name().c_str();
      
      if (*ptr == '.') {
	if (*(ptr+1) == '/') {
	  ptr+=2;
	}
      }
	
      // output the name
      if (strstr(ptr, "Profiler.h")) {
	// Do not list/instrument Profiler.h
      } else if (strstr(ptr, "TAU.h")) {
	// Do not list/instrument TAU.h
      } else {
	if (mode_print_id) {
	  if (strcmp(ptr, target_file) == 0) {
	    //	    cout << ptr << " " << (*it)->id() << endl;
	    cout << (*it)->id() << endl;
	    exit (0);
	  }
	} else {
	  if (show) {
	    if (mode_show_ids) {
	      cout << (*it)->id() << endl;
	    } else {
	      cout << ptr << endl;
	    }
	  }
	}
      }
    }
    printIncludes(*it, false);
  }
}

int main(int argc, char *argv[]) {
  bool errflag = argc < 2;
  if ( errflag ) {
    cerr << "Usage: " << argv[0] << " [--id <file>] [--show <file>] [--showids <file>] pdbfile" << endl;
    return 1;
  }

  int idx = 1;
  if (strcmp(argv[idx],"--id") == 0) {
    mode_print_id = 1;
    idx++;
    target_file = argv[idx];
    idx++;
  }

  if (strcmp(argv[idx],"--show") == 0) {
    mode_show = 1;
    idx++;
    target_file = argv[idx];
    idx++;
  }

  if (strcmp(argv[idx],"--showids") == 0) {
    mode_show_ids = 1;
    mode_show = 1;
    idx++;
    target_file = argv[idx];
    idx++;
  }
  
  PDB pdb(argv[idx]);

  if (!pdb) {
    cerr << "Unable to read PDB file: %s\n", argv[1];
    return 1;
  }

  pdbFile *topFile = pdb.fileTree();
  printIncludes(topFile, true);
  return 0;
}
