// *CWL* - This is hashing specialized for EBS's use. As such, no resizing and
//         rehashing is expected nor allowed. Chaining will provide the necessary
//         collision handling.

#include <stdio.h>
#include <string.h>

#include <Profile/Profiler.h>
#include <Profile/TauMmapMemMgr.h>
#include <Profile/TauPathHash.h>

typedef struct HashTableInfo {
  int size;
} table_info_t;

static hash_entry_t *table[TAU_MAX_THREADS][TAU_PATHHASH_MAX_TABLES];
static table_info_t tableInfo[TAU_MAX_THREADS][TAU_PATHHASH_MAX_TABLES];
static int nextTableHandle[TAU_MAX_THREADS];

// *CWL* - Hash Function derived from PJW Hash as described here:
//         http://www.cs.hmc.edu/~geoff/classes/hmc.cs070.200101/homework10/hashfuncs.html
//
//         TAU_PATHHASH_KEY_TYPE is assumed to be some type of integer. The length of the sequence
//         is found at keySequence[0]
int hashSequence(TAU_PATHHASH_KEY_TYPE keySequence) {
  TAU_PATHHASH_HASH_TYPE h = 0;
  TAU_PATHHASH_HASH_TYPE g = 0; // temp
  int length = (int)keySequence[0];
  // Convert bytes to bits. For the 4-bit shifts.
  int shiftOffset = sizeof(TAU_PATHHASH_HASH_TYPE)*8 - 8; 

  for (int i=0; i<length; i++) {
    // The top 4 bits of h are all zero
    h = (h << 4) + keySequence[i+1];  // shift h 4 bits left, add in ki
    g = h & HASH_FILTER;              // get the top 4 bits of h
    if (g != 0) {                     // if the top 4 bits aren't zero,
      h = h ^ (g >> shiftOffset);     //   move them to the low end of h
      h = h ^ g;                      // The top 4 bits of h are again all zero
    }
  }
  return h % TAU_PATHHASH_DEFAULT_SIZE;
}

// Returns true if equal, false otherwise.
bool compareKey(TAU_PATHHASH_KEY_TYPE key1, TAU_PATHHASH_KEY_TYPE key2) {
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

void TauPathHash_initIfNecessary() {
  static bool initialized = false;
  static bool thrInitialized[TAU_MAX_THREADS];
  if (!initialized) {
    for (int i=0; i<TAU_MAX_THREADS; i++) {
      thrInitialized[i] = false;
      nextTableHandle[i] = 0;
      for (int j=0; j<TAU_PATHHASH_MAX_TABLES; j++) {
	tableInfo[i][j].size = 0;
      }
    }
    initialized = true;
  }

  // *CWL* - stub. TODO: Replace with RtsThread:MyThread() later
  int myTid = RtsLayer::myThread();

  if (!thrInitialized[myTid]) {
    // acquire enough mmapped memory for the hash table and some extra
    //   space for chained overflows. The keys themselves will be allocated
    //   using the memory management unit but outside of the hashing
    //   framework.
    void *addr = Tau_MemMgr_mmap(myTid, TAU_MEMMGR_DEFAULT_BLOCKSIZE*8);
    if (addr == NULL) {
      fprintf(stderr, "ERROR tid %d: Hash failed to initialize by acquiring [%lld] bytes\n",
	      myTid,
	      TAU_MEMMGR_DEFAULT_BLOCKSIZE*8);
    }
    thrInitialized[myTid] = true;
  }
}

char *keyName(TAU_PATHHASH_KEY_TYPE key) {
  char buffer[4096];
  char temp[100];
  if (key == NULL) {
    return strdup("");
  }
  int length = key[0];
  sprintf(buffer, "");
  for (int i=0; i<length; i++) {
    sprintf(temp, "%ld ", key[i+1]);
    strcat(buffer, temp);
  }
  //  printf("%s\n", buffer);
  return strdup(buffer);
}

void TauPathHash_printTable(int tid, tau_hash_handle_t handle) {
  int length = tableInfo[tid][handle].size;
  if (length > 0) {
    hash_entry_t *temp;
    for (int i=0; i<length; i++) {
      hash_entry_t *entry = &(table[tid][handle][i]);
      if (entry->hasValue) {
	printf("Bucket %d: Key [%s] Value: %ld\n", i, keyName(entry->key), entry->value);
	temp = entry->chainPtr;
	int count = 0;
	while (temp != NULL) {
	  count++;
	  printf("Bucket %d-%d: Key [%s] Value: %ld\n", i, count,
		 keyName(temp->key), temp->value);
	  temp = temp->chainPtr;
	}
      }
    }
  }
}

// size = number of elements.
// Returns a handle to the table or -1 if fail.
tau_hash_handle_t TauPathHash_createHashTable(int tid, int size) { // defaults to TAU_PATHHASH_DEFAULT_SIZE
  if (nextTableHandle[tid] > TAU_PATHHASH_MAX_TABLES) {
    fprintf(stderr, "ERROR tid %d: Hit max hash table limit of %d", tid, TAU_PATHHASH_MAX_TABLES);
    return -1;
  }
  table[tid][nextTableHandle[tid]] = 
    (hash_entry_t *)Tau_MemMgr_malloc(tid, sizeof(hash_entry_t)*size);
  if (table[tid][nextTableHandle[tid]] != NULL) {
    for (int i=0; i<size; i++) {
      table[tid][nextTableHandle[tid]][i].hasValue = false;
    }
    tableInfo[tid][nextTableHandle[tid]].size = size;
    return nextTableHandle[tid]++;
  } else {
    fprintf(stderr, "ERROR tid %d: Failed to acquire memory for Hash table.", tid);
    return -1;
  }
}

// returns NULL or a reference to the element
hash_entry_t *TauPathHash_hashGet(int tid, tau_hash_handle_t handle, TAU_PATHHASH_KEY_TYPE key) {
  int bucket = hashSequence(key);
  //  printf("INFO: tid=%d bucket %d\n", tid, bucket);

  hash_entry_t *entryPtr = &(table[tid][handle][bucket]);
  // table entry itself
  if (entryPtr->hasValue) {
    if (compareKey(key, entryPtr->key)) {
      // found it. Return a reference to the element so it may be manipulated or used.
      return entryPtr;
    } else {
      // There is no need to check hasValue. All chains have value.
      entryPtr = entryPtr->chainPtr;
      while (entryPtr != NULL) {
	if (compareKey(key, entryPtr->key)) {
	  return entryPtr;
	}
	entryPtr = entryPtr->chainPtr;
      }
      return NULL;
    }
  } else {
    // Not found
    return NULL;
  }
}

TAU_PATHHASH_KEY_TYPE copyKey(int tid, TAU_PATHHASH_KEY_TYPE key) {
  TAU_PATHHASH_KEY_TYPE newKey = NULL;
  if (key != NULL) {
    int length = key[0];
    newKey = (TAU_PATHHASH_KEY_TYPE)Tau_MemMgr_malloc(tid, sizeof(TAU_PATHHASH_HASH_TYPE)*(length+1));
    for (int i=0; i<length; i++) {
      newKey[i+1] = key[i+1];
    }
    newKey[0] = length;
  }
  return newKey;
}

hash_entry_t *createNewChainEntry(int tid, TAU_PATHHASH_KEY_TYPE key, TAU_PATHHASH_VAL_TYPE val) {
  hash_entry_t *newEntry = (hash_entry_t *)Tau_MemMgr_malloc(tid, sizeof(hash_entry_t));
  if (newEntry == NULL) {
    // Memory allocation failed!
    fprintf(stderr, "ERROR tid %d: Failed to create new hash chain entry. malloc failed.\n",
	    tid);
    return NULL;
  }
  newEntry->hasValue = true;
  newEntry->key = copyKey(tid, key);
  newEntry->value = val;
  newEntry->chainPtr = NULL;
  return newEntry;
}

// returns true if successful.
bool TauPathHash_hashInsert(int tid, tau_hash_handle_t handle, TAU_PATHHASH_KEY_TYPE key, TAU_PATHHASH_VAL_TYPE val) {
  int bucket = hashSequence(key);
  //  printf("INFO: tid=%d bucket %d\n", tid, bucket);

  hash_entry_t *entryPtr = &(table[tid][handle][bucket]);
  hash_entry_t *tempPtr = NULL;
  if (entryPtr->hasValue) {
    if (compareKey(key, entryPtr->key)) {
      entryPtr->value = val;
      return true;
    } else { //  ! compareKey(key, entryPtr->key)
      printf("INFO: tid=%d Collision at bucket %d!\n", tid, bucket);
      if (entryPtr->chainPtr == NULL) {
	// no chains at the top level. Just insert.
	hash_entry_t *newEntry = createNewChainEntry(tid, key, val);
	if (newEntry == NULL) {
	  return false;
	}
	entryPtr->chainPtr = newEntry;
	return true;
      } else { // ! entryPtr->chainPtr == NULL
	tempPtr = entryPtr->chainPtr;
	do {
	  if (compareKey(key, tempPtr->key)) {
	    // found!
	    tempPtr->value = val;
	    return true;
	  } else {
	    tempPtr = tempPtr->chainPtr;
	  }
	} while (tempPtr->chainPtr != NULL);
	// chain has ended
	hash_entry_t *newEntry = createNewChainEntry(tid, key, val);
	if (newEntry == NULL) {
	  return false;
	}
	tempPtr->chainPtr = newEntry;
	return true;
      } // entryPtr->chainPtr == NULL
    } // compareKey(key, entryPtr->key)
  } else { // ! entryPtr->hasValue
    // Top level entry has no value. Set it up.
    entryPtr->hasValue = true;
    entryPtr->key = copyKey(tid, key);
    entryPtr->value = val;
    entryPtr->chainPtr = NULL;
    return true;
  }
}
