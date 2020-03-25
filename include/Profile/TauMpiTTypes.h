#ifndef TAU_MPIT_TYPES_H
#define TAU_MPIT_TYPES_H

struct StringPair {
  char first[TAU_NAME_LENGTH];
  char second[TAU_NAME_LENGTH];
};

struct ListStringPair {
  struct StringPair pair;
  struct ListStringPair *link;
};

typedef struct StringPair StringPair;
typedef struct ListStringPair ListStringPair;

struct ScalarControlVariable {
  char name[TAU_NAME_LENGTH];
  char value[TAU_NAME_LENGTH];
};

typedef struct ScalarControlVariable ScalarControlVariable;

struct VectorControlVariable {
  char name[TAU_NAME_LENGTH];
  ListStringPair *list;
  int number_of_elements;
};

typedef struct VectorControlVariable VectorControlVariable;

#endif //TAU_MPIT_TYPES_H
