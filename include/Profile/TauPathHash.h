#ifndef TAU_PATH_HASH_H_
#define TAU_PATH_HASH_H_

#define TAU_PATHHASH_MAX_TABLES 5
#define TAU_PATHHASH_DEFAULT_SIZE 131071 /* 2^17 - 1 */

#define tau_hash_handle_t int

/* *CWL* - This is limited in scope for TAU EBS purposes only. 
           As such:
	   HASH_TYPE = some kind of integer.
           KEY_TYPE = some sequence of HASH_TYPE.
	   VAL_TYPE = some kind of integer.

	   TODO - find a more elegant way to derive template the functionality.
	          We still want to restrict any templating to just mapping to
		  different key value types.
 */
#define TAU_PATHHASH_VAL_TYPE unsigned long
#define TAU_PATHHASH_HASH_TYPE unsigned long
#define TAU_PATHHASH_KEY_TYPE TAU_PATHHASH_HASH_TYPE *
#define HASH_FILTER 0xf000000000000000 


typedef struct HashEntry {
  TAU_PATHHASH_KEY_TYPE key;
  TAU_PATHHASH_VAL_TYPE value;
  bool hasValue;
  struct HashEntry *chainPtr;
} hash_entry_t;

void TauPathHash_initIfNecessary();
tau_hash_handle_t TauPathHash_createHashTable(int tid, int size=TAU_PATHHASH_DEFAULT_SIZE);
hash_entry_t *TauPathHash_hashGet(int tid, tau_hash_handle_t handle, TAU_PATHHASH_KEY_TYPE key);
bool TauPathHash_hashInsert(int tid, tau_hash_handle_t handle, TAU_PATHHASH_KEY_TYPE key, TAU_PATHHASH_VAL_TYPE val);

void TauPathHash_printTable(int tid, tau_hash_handle_t handle);

#endif /* TAU_PATH_HASH_H_ */
