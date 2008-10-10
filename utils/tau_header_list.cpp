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

static void printIncludes(const pdbFile *f, bool first) {
  pdbFile::incvec i = f->includes();
  for (pdbFile::incvec::iterator it=i.begin(); it!=i.end(); ++it) {
    if (!(*it)->isSystemFile()) { // exclude system files
      // skip over "./"
      const char *ptr = (*it)->name().c_str();
      if (*ptr == '.') {
	if (*(ptr+1) == '/') {
	  ptr+=2;
	}
      }
      if (!first) {
	// output the name
	cout << ptr << endl;
      }
    }
    printIncludes(*it, false);
  }
}

int main(int argc, char *argv[]) {
  bool errflag = argc < 2;
  if ( errflag ) {
    cerr << "Usage: " << argv[0] << " pdbfile" << endl;
    return 1;
  }

  PDB pdb(argv[1]);

  if (!pdb) {
    cerr << "Unable to read PDB file: %s\n", argv[1];
    return 1;
  }

  pdbFile *topFile = pdb.fileTree();
  printIncludes(topFile, true);
  return 0;
}
