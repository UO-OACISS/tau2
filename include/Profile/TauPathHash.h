#ifndef TAU_PATH_HASH_H_
#define TAU_PATH_HASH_H_
#ifndef _AIX

#include <utility>
#include <cstdio>
#include <cstring>

// Putting "using namespace" statements in header files can create ambiguity
// between user-defined symbols and std symbols, creating unparsable code
// or even changing the behavior of user codes.  This is also widely considered
// to be bad practice.  Here's a code PDT can't parse because of this line:
//   EX: #include <complex>
//   EX: typedef double real;
//
//using namespace std;

/* *CWL* - Uses the mmap memory manager for signal-safety. Note that allocating
   TauPathHashTable itself is signal-unsafe. So care must be taken to ensure
   that each FunctionInfo object that needs a hash table for EBS creates one
   outside of an interrupt-based sample. A good spot would be at the time of the
   FunctionInfo constructor/initializer.
 */
#include <Profile/TauMmapMemMgr.h>

/* *CWL* - Each FI will host its own hash table for EBS. It is expected that the
   vast majority of FIs will encounter fewer than 63 unique unwind paths
   into any given sample. The memory cost of a resize is limited to
   double the actual use. The trigger threshold for a resize-rehash
   operation could be made conservative to keep these operations to a
   minimum.
*/
#define TAU_PATHHASH_DEFAULT_SIZE 63

/* *CWL* - This is limited in scope for TAU path keys
   (EBS, CallSites) only. As such, the key will always be (unsigned long *).
   The value type can be templated.

   TODO - define new filters for 32-bit platforms?
 */
#include <limits.h>
#if WORD_BIT > 32
#define HALF_SIZEOF_PTR 4
#define HASH_FILTER 0xf000000000000000 /* 8-byte unsigned */
#else
#define HALF_SIZEOF_PTR 2
#define HASH_FILTER 0xf0000000 /* 4-byte unsigned */
#endif

// Kevin: This is the class that will be contained in the hash table.
class TauPathAccumulator {
 public:
  unsigned long count;
  // Kevin: when we update PAPI support, this should be an array
  double accumulator[TAU_MAX_COUNTERS];
  // destructor
  ~TauPathAccumulator() {}
  // constructor
  TauPathAccumulator(int inCount, double firstValues[TAU_MAX_COUNTERS]) {
    count = inCount;
    int i;
    for (i = 0; i < Tau_Global_numCounters ; i++) {
      accumulator[i] = firstValues[i];
    }
  }
  // default constructor
  TauPathAccumulator() {
    count = 0;
    int i;
    for (i = 0; i < Tau_Global_numCounters ; i++) {
      accumulator[i] = 0.0;
    }
  }
};


template <class T> class TauPathHashTable {
 private:
  int tid; /* This will be needed for accessing the right manager blocks without locks */
  int tableSize;
  int numElements;

  /* book keeping variables for decision to resize */
  bool disableGrow;

  struct KeyValuePair {
    unsigned long *key;
    T value;
  };

  struct HashElement {
    struct KeyValuePair *pair;
    struct HashElement *next;
  };

  // Iterator support
  struct HashElement *iterPtr;
  int iterCount;
  int iterTblIdx;

  struct HashElement **table;
  struct HashElement **oldTable; /* for resize/rehashing */

  unsigned long hashSequence(const unsigned long *keySequence);
  bool compareKey(const unsigned long *key1, const unsigned long *key2);
  unsigned long *copyKey(int tid, const unsigned long *key);

  struct HashElement *createNewEntry(int tid, const unsigned long *key, T val) {
    struct HashElement *newEntry =
      (struct HashElement *)Tau_MemMgr_malloc(tid, sizeof(struct HashElement));
    if (newEntry == NULL) {
      /* Memory allocation failed! */
      fprintf(stderr, "ERROR tid %d: Failed to create new hash element. Tau_MemMgr_malloc failed.\n",
	      tid);
      return NULL;
    }
    struct KeyValuePair *newPair =
      (struct KeyValuePair *)Tau_MemMgr_malloc(tid, sizeof(struct KeyValuePair));
    if (newPair == NULL) {
      /* Memory allocation failed! */
      fprintf(stderr, "ERROR tid %d: Failed to create new hash key-value pair. Tau_MemMgr_malloc failed.\n",
	      tid);
      return NULL;
    }
    newEntry->pair = newPair;
    newEntry->pair->key = copyKey(tid, key);
    newEntry->pair->value = val;
    newEntry->next = NULL;
    //    printf("next:%p key:%p value:%ld\n", newEntry->next, newEntry->pair->key, newEntry->pair->value);
    return newEntry;
  }

  bool growThreshold(int tid);
  void growTable(int tid);
  void insertNoAllocate(int tid, struct HashElement *element);

 public:
  /* returns a pointer for manipulation (eg. increment) */
  T* get(const unsigned long *key);
  bool insert(const unsigned long *key, T val);
  int size() { return numElements; }

  // Best-effort attempt at an iterator. Not intended to be used while the
  //   hash table is still being actively updated. Not signal-safe unlike
  //   the rest of the hash table.
  void resetIter();
  std::pair<unsigned long *, T> *nextIter();

  // for debugging
  void printTable();

  /* For now, the destructor will not attempt to free/mmunmap memory */
  ~TauPathHashTable() {}
  /* Only the constructor needs to know tid because the correct memory segments are needed */
  TauPathHashTable(int tid_, int size_=TAU_PATHHASH_DEFAULT_SIZE):tid(tid_),
    tableSize(size_),numElements(0),disableGrow(false),iterPtr(NULL),iterCount(0),iterTblIdx(0) {

    table = (struct HashElement **)Tau_MemMgr_malloc(tid,
						     sizeof(struct HashElement *)*tableSize);
    //TAU_VERBOSE("Table created %p at size %d\n", table, tableSize);
    for (int i=0; i<tableSize; i++) {
      table[i] = NULL;
    }
    oldTable = NULL;
  }
};

// *CWL* - This is hashing specialized for EBS's use. As such, no resizing and
//         rehashing is expected nor allowed. Chaining will provide the necessary
//         collision handling.

// *CWL* - Hash Function derived from PJW Hash as described here:
//         http://www.cs.hmc.edu/~geoff/classes/hmc.cs070.200101/homework10/hashfuncs.html
//
// The length of the sequence is found at keySequence[0]
//
template <class T>
unsigned long TauPathHashTable<T>::hashSequence(const unsigned long *keySequence) {
  unsigned long h = 0;
  unsigned long g = 0; // temp
  int length = (int)keySequence[0];
  // Convert bytes to bits. For the 4-bit shifts.
  int shiftOffset = sizeof(unsigned long)*2*HALF_SIZEOF_PTR - 2*HALF_SIZEOF_PTR;

  for (int i=0; i<length; i++) {
    // The top 4 bits of h are all zero
    h = (h << HALF_SIZEOF_PTR) + keySequence[i+1];  // shift h 4 bits left, add in ki
    g = h & HASH_FILTER;              // get the top 4 bits of h
    if (g != 0) {                     // if the top 4 bits aren't zero,
      h = h ^ (g >> shiftOffset);     //   move them to the low end of h
      h = h ^ g;                      // The top 4 bits of h are again all zero
    }
  }
  return h % tableSize;
}

// Returns true if equal, false otherwise.
template <class T>
bool TauPathHashTable<T>::compareKey(const unsigned long *key1, const unsigned long *key2) {
  int l1, l2;
  // key1 and key2 cannot be NULL
  if (key1 == NULL || key2 == NULL) {
    return false;
  }
  l1 = key1[0];
  l2 = key2[0];
  if (l1 != l2) {
    return false;
  }
  for (int i=0; i<l1; i++) {
    if (key1[i+1] != key2[i+1]) {
      return false;
    }
  }
  return true;
}

template <class T>
unsigned long *TauPathHashTable<T>::copyKey(int tid, const unsigned long *key) {
  unsigned long *newKey = NULL;
  if (key != NULL) {
    int length = key[0];
    newKey = (unsigned long *)Tau_MemMgr_malloc(tid, sizeof(unsigned long)*(length+1));
    for (int i=0; i<length; i++) {
      newKey[i+1] = key[i+1];
    }
    newKey[0] = length;
  }
  return newKey;
}

template <class T>
bool TauPathHashTable<T>::growThreshold(int tid) {
  (void)(tid); // unused
  if (disableGrow) {
    return false;
  }
  return false;
}

// Decision made to resize the table
template <class T>
void TauPathHashTable<T>::growTable(int tid) {
  int oldSize = tableSize;
  oldTable = table; // hang on to the old table for the rehash

  tableSize *= 2 + 1;
  table =
    (struct HashElement **)Tau_MemMgr_malloc(tid,
					     sizeof(struct HashElement *)*tableSize);
  for (int i=0; i<tableSize; i++) {
    table[i] = NULL;
  }

  // Now rehash all elements. Remember we are only going to move the pointers around.
  //   We should not be using the public interface for this!
  HashElement *currentElement = NULL;
  HashElement *nextElement = NULL;
  numElements = 0;
  for (int i=0; i<oldSize; i++) {
    currentElement = oldTable[i];
    if (currentElement != NULL) {
      // Go down the chain, inserting.
      do {
	nextElement = currentElement->next; // store the previous chain pointer
	currentElement->next = NULL; // reset the pointer in the rehash
	insertNoAllocate(tid, currentElement);
	numElements++;
	currentElement = nextElement;
      } while (currentElement != NULL);
    }
  }

  // drop the previously allocated memory into the nether ... we will not attempt
  //   to reclaim it, for now.
  oldTable = NULL;
}

template <class T>
void TauPathHashTable<T>::insertNoAllocate(int tid, struct HashElement *element) {
  (void)(tid); // unused
  unsigned long bucket = hashSequence(element->pair->key);
  //  printf("INFO: tid=%d bucket %d\n", tid, bucket);

  struct HashElement *currentElement = table[bucket];
  if (currentElement == NULL) {
    table[bucket] = element;
  } else {
    while (currentElement->next != NULL) {
      currentElement = currentElement->next;
    }
    currentElement->next = element;
  }
}

template <class T>
void TauPathHashTable<T>::printTable() {
  for (int i=0; i<tableSize; i++) {
    struct HashElement *currentElement = table[i];
    if (currentElement != NULL) {
      int count = 0;
      while (currentElement != NULL) {
	printf("[%d-%d] ", i, count++);
	for (int j=0; j<currentElement->pair->key[0]; j++) {
	  printf("%p ", currentElement->pair->key[j+1]);
	}
	printf("| Value = %d\n", currentElement->pair->value);
	currentElement = currentElement->next;
      }
    }
  }
}

// returns NULL or a reference to the element
template <class T>
T* TauPathHashTable<T>::get(const unsigned long *key) {
  unsigned long bucket = hashSequence(key);
  //  printf("INFO: tid=%d bucket %d\n", tid, bucket);
  struct HashElement *entryPtr = table[bucket];
  // table entry itself
  if (entryPtr != NULL) {
    do {
      if (compareKey(key, entryPtr->pair->key)) {
	// found it. Return a reference to the element so it may be manipulated or used.
	return &(entryPtr->pair->value);
      }
      entryPtr = entryPtr->next;
    } while (entryPtr != NULL);
    // nothing in the chain
    return NULL;
  } else { // ! entryPtr != NULL
    // No key matched
    return NULL;
  }
}

// returns true if successful.
template <class T>
bool TauPathHashTable<T>::insert(const unsigned long *key, T val) {
  unsigned long bucket = hashSequence(key);
  //  printf("INFO: tid=%d bucket %d\n", tid, bucket);
  struct HashElement *newEntry = NULL;
  struct HashElement *entryPtr = table[bucket];
  if (entryPtr != NULL) {
    // Go down the colision list. If key is matched, update the value.
    //   If we get to the end of the list without a match, insert a new
    //   element to the end.
    do {
      if (compareKey(key, entryPtr->pair->key)) {
	entryPtr->pair->value = val;
	return true;
      }
      // just processed the last element and failed to match
      if (entryPtr->next == NULL) {
	newEntry = createNewEntry(tid, key, val);
	if (newEntry == NULL) {
	  // This is bad, failed to create the new entry!
	  return false;
	} else {
	  entryPtr->next = newEntry;
	  numElements++;
	  return true;
	}
      }
      entryPtr = entryPtr->next;
    } while (entryPtr != NULL); // This condition is spurious
  } else { // ! entryPtr != NULL
    newEntry = createNewEntry(tid, key, val);
    if (newEntry == NULL) {
      // This is bad, failed to create the new entry!
      return false;
    } else {
      table[bucket] = newEntry;
      numElements++;
      return true;
    }
  }
  // Silence a compiler warning
  return false;
}

template <class T>
void TauPathHashTable<T>::resetIter() {
  iterPtr = NULL;
  iterCount = 0;
  iterTblIdx = 0;
}

template <class T>
std::pair<unsigned long *, T> *TauPathHashTable<T>::nextIter()
{
    typedef std::pair<unsigned long*, T> pair_t;

  //  printf("numElements = %d\n", numElements);
  //  printf("IterPtr starts at %p\n", iterPtr);
  if (iterCount == numElements) {
    //    printf("Iteration count of %d has hit limit of %d\n", iterCount, numElements);
    return NULL ;
  }
  // Need to search later table entries.
  //   This test relies on short-cicruiting.
  //   iterPtr == NULL means we just started.
  //   iterPtr->next == NULL means we are at the end of a chain.
  if ((iterPtr == NULL) || (iterPtr->next == NULL)) {
    bool found = false;
    iterTblIdx++; // go to next table
    while (iterTblIdx < tableSize) {
      if (table[iterTblIdx] != NULL) {
	// only increment iterTblIdx if false
	found = true;
	break;
      }
      iterTblIdx++;
    }
    if (found) {
      iterPtr = table[iterTblIdx];
      //      printf("IterPtr found in table entry at %p\n", iterPtr);
      pair_t * item = new pair_t(iterPtr->pair->key, iterPtr->pair->value);
      //      printf("Found key %p value %d\n", item->first, item->second);
      iterCount++;
      return item;
    } else {
      //      printf("Nothing found in table!\n");
      return NULL;
    }
  } else { // somewhere in a chain
    // This condition must be true:
    //   iterPtr != NULL && iterPtr->next != NULL
    iterPtr = iterPtr->next;
    //    printf("IterPtr found in chain at %p\n", iterPtr);
    pair_t * item = new pair_t(iterPtr->pair->key, iterPtr->pair->value);
    //      printf("Found key %p value %d\n", item->first, item->second);
    iterCount++;
    return item;
  }
}

#endif /* _AIX */
#endif /* TAU_PATH_HASH_H_ */
