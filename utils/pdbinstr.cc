#include <fstream.h>
#include <ctype.h>
#include <string.h>
#include <stdlib.h>
#ifdef _OLD_HEADER_
# include <set.h>
# include <algo.h>
#else
# include <set>
  using std::set;
# include <algorithm>
  using std::sort;
#endif
#include "pdbAll.h"


struct itemRef {
  itemRef(const pdbItem *i, bool isT) : item(i), isTarget(isT) {
    line = i->location().line();
    col  = i->location().col();
  }
  itemRef(const pdbItem *i, bool isT, int l, int c)
         : item(i), isTarget(isT), line(l), col(c) {}
  const pdbItem *item;
  bool     isTarget;
  int      line;
  int      col;
};

static bool locCmp(const itemRef* r1, const itemRef* r2) {
  if ( r1->line == r2->line )
    return r1->col < r2->col;
  else
    return r1->line < r2->line;
}
 
static const char *toName(pdbItem::templ_t v) ;
void getReferences(vector<itemRef *>& itemvec, PDB& pdb, pdbFile *file) {
/* get routines, templates and member templates of classes */
  PDB::routinevec routines = pdb.getRoutineVec();
  for (PDB::routinevec::const_iterator rit=routines.begin();
       rit!=routines.end(); ++rit) 
  {
    if ( (*rit)->location().file() == file && !(*rit)->isCompilerGenerated() && 
	 ((*rit)->storageClass() != pdbItem::ST_EXT)) 
    {
      // It is a "target" so give it a second arg of true.
      itemvec.push_back(new itemRef(*rit, true));
#ifdef DEBUG
      cout << (*rit)->fullName() <<endl;
#endif
    }
  }

  /* templates are not included in this. Get them */
  PDB::templatevec u = pdb.getTemplateVec();
  for(PDB::templatevec::iterator te = u.begin(); te != u.end(); ++te)
  { 
    if ( (*te)->location().file() == file)
    {
      pdbItem::templ_t tekind = (*te)->kind();
      if ((tekind == pdbItem::TE_FUNC) || (tekind == pdbItem::TE_MEMFUNC) || 
	  (tekind == pdbItem::TE_STATMEM))
      { 
  	// templates need some processing. Give it a false for isTarget arg.
        itemvec.push_back(new itemRef(*te, false));
	// Once we get to the template search for "{" and then put in instr. 
	// to identify this processing, we give it a false arg. Functions are
	// straightforward and need no special processing. 
      }
    }
  }
  sort(itemvec.begin(), itemvec.end(), locCmp);
}


static char toUpper(const char ch) { return (ch - 'a' + 'A'); }
 
static const char *toName(pdbItem::access_t v) {
  switch (v) {
  case pdbItem::AC_PUB : return "public";
  case pdbItem::AC_PRIV: return "private";
  case pdbItem::AC_PROT: return "protected";
  case pdbItem::AC_NA  :
  default              : return "NA";
  }
}

static const char *toName(pdbItem::store_t v) {
  switch (v) {
  case pdbItem::ST_ASM : return "assembler";
  case pdbItem::ST_AUTO: return "automatic";
  case pdbItem::ST_EXT : return "extern";
  case pdbItem::ST_STAT: return "static";
  case pdbItem::ST_NA  :
  default              : return "NA";
  }
}

static const char *toName(pdbItem::func_t v) {
  switch (v) {
  case pdbItem::FU_CONV: return "conversionOperator";
  case pdbItem::FU_CTOR: return "constructor";
  case pdbItem::FU_DTOR: return "destructor";
  case pdbItem::FU_OP  : return "operator";
  case pdbItem::FU_NA  :
  default              : return "NA";
  }
}

static const char *toName(pdbItem::virt_t v) {
  switch (v) {
  case pdbItem::VI_PURE: return "pureVirtual";
  case pdbItem::VI_VIRT: return "virtual";
  case pdbItem::VI_NO  :
  default              : return "no";
  }
}

static const char *toName(pdbItem::macro_t v) {
  switch (v) {
  case pdbItem::MA_DEF  : return "define";
  case pdbItem::MA_UNDEF: return "undef";
  case pdbItem::MA_NA   :
  default               : return "NA";
  }
}

static const char *toName(pdbItem::templ_t v) {
  switch (v) {
  case pdbItem::TE_CLASS   : return "class";
  case pdbItem::TE_FUNC    : return "function";
  case pdbItem::TE_MEMCLASS: return "memberClass";
  case pdbItem::TE_MEMFUNC : return "memberFunction";
  case pdbItem::TE_STATMEM : return "staticDataMember";
  case pdbItem::TE_NA      :
  default                  : return "NA";
  }
}

static const char *toName(pdbItem::float_t v) {
  switch (v) {
  case pdbItem::FL_FLOAT  : return "float";
  case pdbItem::FL_DBL    : return "double";
  case pdbItem::FL_LONGDBL: return "longDouble";
  case pdbItem::FL_NA     :
  default                 : return "NA";
  }
}

static const char *toName(pdbItem::int_t v) {
  switch (v) {
  case pdbItem::I_CHAR     : return "character";
  case pdbItem::I_SCHAR    : return "signedCharacter";
  case pdbItem::I_UCHAR    : return "unsignedCharacter";
  case pdbItem::I_SHORT    : return "short";
  case pdbItem::I_USHORT   : return "unsignedShort";
  case pdbItem::I_INT      : return "integer";
  case pdbItem::I_UINT     : return "unsignedInteger";
  case pdbItem::I_LONG     : return "long";
  case pdbItem::I_ULONG    : return "uunsignedLong";
  case pdbItem::I_LONGLONG : return "longLong";
  case pdbItem::I_ULONGLONG: return "unsignedLongLong";
  case pdbItem::I_NA       :
  default                  : return "NA";
  }
}

static const char *toName(pdbItem::type_t v) {
  switch (v) {
  case pdbItem::TY_BOOL  : return "boolean";
  case pdbItem::TY_ENUM  : return "enumeration";
  case pdbItem::TY_ERR   : return "error";
  case pdbItem::TY_FUNC  : return "functionSignature";
  case pdbItem::TY_VOID  : return "void";
  case pdbItem::TY_INT   : return "integer";
  case pdbItem::TY_FLOAT : return "float";
  case pdbItem::TY_PTR   : return "pointerOrReference";
  case pdbItem::TY_ARRAY : return "array";
  case pdbItem::TY_TREF  : return "typeReference";
  case pdbItem::TY_PTRMEM: return "pointerToMember";
  case pdbItem::TY_TPARAM: return "templateParameter";
  case pdbItem::TY_WCHAR : return "wideCharacter";
  case pdbItem::TY_CLASS : return "classType";
  case pdbItem::TY_NA    :
  default                : return "NA";
  }
}

static const char *toName(pdbItem::qual_t v) {
  switch (v) {
  case pdbItem::QL_CONST   : return "const";
  case pdbItem::QL_VOLATILE: return "volatile";
  case pdbItem::QL_RESTRICT: return "restrict";
  default                  : return "NA";
  }
}

static const char *toName(pdbItem::class_t v) {
  switch (v) {
  case pdbItem::CL_CLASS : return "class";
  case pdbItem::CL_STRUCT: return "struct";
  case pdbItem::CL_UNION : return "union";
  case pdbItem::CL_NA    :
  default                : return "NA";
  }
}

static const char *toName(pdbItem::mem_t v) {
  switch (v) {
  case pdbItem::M_VAR    : return "variable";
  case pdbItem::M_STATVAR: return "staticVariable";
  case pdbItem::M_TYPE   : return "type";
  case pdbItem::M_NA     :
  default                : return "NA";
  }
}

void printLo(ostream& ostr, const pdbLoc& l) {
  if ( l.file() ) {
    ostr << "location:           SO#"
         << l.file()->id() << " " << l.file()->name()
         << " " << l.line() << " " << l.col() << "\n";
  } else {
    ostr << "location:           <UNKNOWN>\n";
  }
}

void printItem(ostream& ostr, const pdbItem *i) {
  ostr << toUpper(i->desc()[0]) << toUpper(i->desc()[1]) << "#" << i->id()
       << " " << i->fullName() << "\n";
  printLo(ostr, i->location());
  if ( const pdbClass* cptr = i->parentClass() ) {
    ostr << "class:              CL#"
         << cptr->id() << " " << cptr->name() << "\n";
    ostr << "access:             " << toName(i->access()) << "\n";
  }
  if ( const pdbNamespace* nptr = i->parentNSpace() ) {
    ostr << "namespace:          NA#"
         << nptr->id() << " " << nptr->name() << "\n";
  }

}

const int INBUF_SIZE = 2048;

/* to instrument the file */
int instrumentFile(PDB& pdb, pdbFile* f, string& outfile) 
{
  string file(f->name());
  static char inbuf[INBUF_SIZE]; // to read the line
  // open outfile for instrumented version of source file
  ofstream ostr(outfile.c_str());
  if (!ostr) {
    cerr << "Error: Cannot open '" << outfile << "'" << endl;
    return false;
  }
  // open source file
  ifstream istr(file.c_str());
  if (!istr) {
    cerr << "Error: Cannot open '" << file << "'" << endl;
    return false;
  }
#ifdef DEBUG
  cout << "Processing " << file << " ..." << endl;
#endif 


  // initialize reference vector
  vector<itemRef *> itemvec;
  getReferences(itemvec, pdb, f);

  // put in code to insert <Profile/Profiler.h> 
  ostr<< "#include <Profile/Profiler.h>"<<endl;
  
  int inputLineNo = 0;
  int lastInstrumentedLineNo = 0;
  for(vector<itemRef *>::iterator it = itemvec.begin(); it != itemvec.end();
	++it)
  {
    // Read one line each till we reach the desired line no. 
#ifdef DEBUG
    cout <<"S: "<< (*it)->item->fullName() << " line "<< (*it)->line << " col " << (*it)->col << endl;
#endif 
    bool instrumented = false;
    while((instrumented == false) && (istr.getline(inbuf, INBUF_SIZE)) )
    {
      inputLineNo ++;
      if (inputLineNo < (*it)->line) 
      {
	// write the input line in the output stream
        ostr << inbuf <<endl;
      }
      else 
      { 
	// we're at the desired line no. search for an open brace
	int inbufLength = strlen(inbuf);
	if (lastInstrumentedLineNo >= (*it)->line) 
	{ 
	  // Hey! This line has already been instrumented. Go to the next 
	  // entry in the func
#ifdef DEBUG
	  cout <<"Entry already instrumented! line = "<<(*it)->line <<endl;
#endif 
	  ostr << inbuf <<endl;
	  break; // takes you to the outermost for loop
	}

	for(int i=0; i< inbufLength; i++)
	{ 
	  if ((inbuf[i] == '{') && (instrumented == false))
	  {
#ifdef DEBUG
	    cout <<"found the *first* { on the line" <<endl;
#endif 
	    ostr << inbuf[i] <<endl; // write the open brace and '\n'
	    // put in instrumentation for the routine HERE
	    //ostr <<"/*** INSTRUMENTATION ***/\n"; 
#ifdef SPACES
	    for (int space = 0; space < (*it)->col ; space++) ostr << " " ; 
#endif
	    // leave some leading spaces for formatting...

	    ostr <<"  TAU_PROFILE(\"" << (*it)->item->fullName() ;
	    if (!(*it)->isTarget) 
	    { // it is a template. Help it by giving an additional ()
	      ostr <<"()" ;
	    } 
	    if (strstr((*it)->item->fullName().c_str(), "main(")) 
	    { /* it is main() */
	      ostr <<"\", \" \", TAU_DEFAULT);" <<endl; // give an additional line 
#ifdef SPACES
	      for (int space = 0; space < (*it)->col ; space++) ostr << " " ; 
#endif 
	      // leave some leading spaces for formatting...
	      ostr <<"  TAU_PROFILE_SET_NODE(0);" <<endl; // set node 0
	    }
	    else 
	    {
	      ostr <<"\", \" \", TAU_USER);" <<endl; // give an additional line 
	    }
	    // if its a function, it already has a ()
	    instrumented = true;
	    lastInstrumentedLineNo = inputLineNo; 
	    // keep track of which line was instrumented last
 	    // and then finish writing the rest of the line 
	    // this will be used for templates and instantiated functions
	  }
	  else 
	  { 
	    // ok to write to ostr 
	    ostr << inbuf[i]; 
          } 
	} // for i loop
	ostr <<endl;
 	// if we didn't find the open brace on the desired line, if its in the 
	// next line go on with the while loop till we return instrumented 
   	// becomes true. 
	
      } // else      

    } // while loop

  } // For loop
  // For loop is over now flush out the remaining lines to the output file
  while (istr.getline(inbuf, INBUF_SIZE) ) 
  { 
    ostr << inbuf <<endl;
  }
  // written everything. quit and debug!
  ostr.close();

  return 0;
}


int main(int argc, char **argv)
{
  string outFileName("out.ins.C");
  if (argc < 3) 
  { 
    cout <<"Usage : "<<argv[0] <<" <pdbfile> <sourcefile> [-o <outputfile>]"<<endl;
    return 1;
  }
  PDB p(argv[1]); if ( !p ) return 1;
  const char * filename = argv[2];  

  if (argc == 5) 
  {
#ifdef DEBUG
    cout <<"5 argc "<<endl;
#endif
    if (strcmp(argv[3], "-o") == 0) 
    { 
#ifdef DEBUG
      cout <<"checks out... -o option"<<endl;
#endif 
      outFileName = string(argv[4]);
    }
    else 
    {
      cout<<"Hey! 5 args but -o doesn't show up as 4th arg." <<endl;
      cout <<"argv[4] is "<<argv[4] <<endl;
    }
  }
  else 
  {
    outFileName = string(filename + string(".ins"));
  }


  for (PDB::filevec::const_iterator it=p.getFileVec().begin();
       it!=p.getFileVec().end(); ++it) 
  {
     if ((*it)->name() == string(filename)) 
     {
#ifdef DEBUG
       cout <<" *** FILE *** "<< (*it)->name()<<endl;
#endif
       instrumentFile(p, *it, outFileName);
     }
  }

  /* start with routines */
/* 
  for (PDB::routinevec::iterator r=p.getRoutineVec().begin();
       r != p.getRoutineVec().end(); r++)
  {
    
#ifdef DEBUG
    cout << (*r)->fullName() <<endl;
#endif

  }
*/
#ifdef DEBUG
  cout <<"Done with instrumentation!" << endl;
#endif 

  return 0;
} 

  
  
