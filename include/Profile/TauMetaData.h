/****************************************************************************
**            TAU Portable Profiling Package               **
**            http://www.cs.uoregon.edu/research/tau               **
*****************************************************************************
**    Copyright 2010                                    **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/****************************************************************************
**    File         : TauMetaData.h                     **
**    Description     : TAU Profiling Package                   **
**    Contact        : tau-bugs@cs.uoregon.edu                      **
**    Documentation    : See http://www.cs.uoregon.edu/research/tau       **
**                                                                         **
**      Description     : This file contains metadata related routines     **
**                                                                         **
****************************************************************************/


#ifndef _TAU_METADATA_H_
#define _TAU_METADATA_H_

#include <Profile/TauMetaDataTypes.h>
#include <Profile/TauUtil.h>
#include <map>
#include <string.h>
#include <sstream>
using namespace std;

// the actual metadata key structure, can be nested.
class Tau_metadata_key {
  public:
  char* name;
  char* timer_context;
  int call_number;
  x_uint64 timestamp;
  Tau_metadata_key() {
    name = NULL;
    timer_context = NULL;
    call_number = 0;
    timestamp = 0;
  }
  /*
  virtual ~Tau_metadata_key() {
    if (name) free(name);
    if (timer_context) free(timer_context);
  }
  */

};


struct Tau_Metadata_Compare: std::binary_function<Tau_metadata_key,Tau_metadata_key,bool>
{
  bool operator()(const Tau_metadata_key& lhs, const Tau_metadata_key& rhs) const {

    char *left;
    char *right;
    int allocate_left = 0;
    int allocate_right = 0;

	// we are using C methods, because the C++ methods didn't work with PGI on Cray XK6.

    if (lhs.timer_context == NULL) {
        left = lhs.name;
    } else {
	    allocate_left = strlen(lhs.name)+strlen(lhs.timer_context)+64;
        left = (char *) calloc(allocate_left, sizeof(char));
        snprintf(left, allocate_left,  "%s%s%d:%llu", lhs.name, lhs.timer_context, lhs.call_number, lhs.timestamp);
    }
    if (rhs.timer_context == NULL) {
        right = rhs.name;
    } else {
        allocate_right = strlen(rhs.name)+strlen(rhs.timer_context)+64;
        right = (char *) calloc(allocate_right, sizeof(char));
        snprintf(right, allocate_right,  "%s%s%d:%llu", rhs.name, rhs.timer_context, rhs.call_number, rhs.timestamp);
    }
    bool result = strcmp(left, right) < 0;
	if (allocate_left > 0) {
        free(left);
	}
	if (allocate_right > 0) {
        free(right);
	}
    return result;
  }
};

void Tau_destructor_trigger(void);

class MetaDataRepo : public map<Tau_metadata_key,Tau_metadata_value_t*,Tau_Metadata_Compare> {
private:
  void freeMetadata (Tau_metadata_value_t * tmv);
public :
  Tau_metadata_value_t*& operator[] (const Tau_metadata_key& k) = delete;
  void emptyRepo(void) {
	MetaDataRepo::iterator it = this->begin();
	while (it != this->end()) {
	  MetaDataRepo::iterator eraseme = it;
	  ++it;
      //if (eraseme->first.name) free(eraseme->first.name);
      if (eraseme->first.timer_context) free(eraseme->first.timer_context);
	  this->freeMetadata(eraseme->second);
	}
	this->clear();
  }
  /* can't delete everything, just the map pointers. */
  void shallowEmpty(void) {
	MetaDataRepo::iterator it = this->begin();
	while (it != this->end()) {
	  MetaDataRepo::iterator eraseme = it;
	  ++it;
	  this->erase(eraseme); // deletes the key, keeps the value in memory for now
	}
	this->clear();
  }
  ~MetaDataRepo(void) {
      Tau_destructor_trigger();
      this->shallowEmpty();
  }
};

MetaDataRepo &Tau_metadata_getMetaData(int tid);
int Tau_metadata_writeMetaData(Tau_util_outputDevice *out, int counter, int tid);
int Tau_metadata_writeMetaData(FILE *fp, int counter, int tid);
int Tau_metadata_writeMetaData(Tau_util_outputDevice *out, int tid);

int Tau_metadata_fillMetaData();
Tau_util_outputDevice *Tau_metadata_generateMergeBuffer();
void Tau_metadata_removeDuplicates(char *buffer, int buflen);

void Tau_metadata_register(const char *name, int value);
void Tau_metadata_register(const char *name, const char *value);
void Tau_metadata_register_task(const char *name, int value, int tid);
void Tau_metadata_register_task(const char *name, const char *value, int tid);

int Tau_metadata_mergeMetaData();
int Tau_write_metadata_records_in_scorep(int tid);
char* Tau_metadata_get(const char *name, int tid);

void Tau_metadata_push_to_plugins(void);

int Tau_metadata_fillOpenMPMetaData(void);

#endif /* _TAU_METADATA_H_ */
