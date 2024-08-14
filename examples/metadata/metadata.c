#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <TAU.h>

#ifdef _OPENMP
#  include <omp.h>
#endif

void bar (int x) {
  int tid = 0;
  int tcount = 1;
  TAU_PROFILE_TIMER(bt, "bar()", "void bar (int x)" , TAU_DEFAULT);
  TAU_PROFILE_START(bt);
#ifdef _OPENMP
  tid = omp_get_thread_num();
  tcount = omp_get_num_threads();
#endif
  printf("Thread %d of %d in bar(x = %d)\n", tid, tcount, x);
  sleep(1);
  TAU_PROFILE_STOP(bt);
}

void foo (int x) {
  int tid = 0;
  int tcount = 1;
  TAU_PROFILE_TIMER(bt, "foo()", "void foo (int x)" , TAU_DEFAULT);
  TAU_PROFILE_START(bt);

#ifdef _OPENMP
  tid = omp_get_thread_num();
  tcount = omp_get_num_threads();
#endif

  // a metadata field with context
  char message[256];
  snprintf(message, sizeof(message), "Thread %d of %d in foo(x = %d)", tid, tcount, x);
  TAU_CONTEXT_METADATA("Foo Parameters", message);
  TAU_METADATA("Foo Parameters", message);

  printf("%s\n", message);
  sleep(1);
  bar(x);
  TAU_PROFILE_STOP(bt);
}

void makeMetaData() {
  // a metadata field with no context
  TAU_METADATA("Test Metadata Name", "Test Metadata Value");
  // a metadata string
  TAU_METADATA_STRING(stringTest, "This is a test string");
  TAU_STRUCTURED_METADATA("Test Metadata String", stringTest);
  // a metadata integer
  TAU_METADATA_INTEGER(intTest, 1234567);
  TAU_STRUCTURED_METADATA("Test Metadata Integer", intTest);
  // a metadata double
  TAU_METADATA_DOUBLE(doubleTest, 1234.567);
  TAU_STRUCTURED_METADATA("Test Metadata Double", doubleTest);
  // a metadata null
  TAU_METADATA_NULL(nullTest);
  TAU_STRUCTURED_METADATA("Test Metadata Null", nullTest);
  // a metadata true
  TAU_METADATA_TRUE(trueTest);
  TAU_STRUCTURED_METADATA("Test Metadata True", trueTest);
  // a metadata false
  TAU_METADATA_FALSE(falseTest);
  TAU_STRUCTURED_METADATA("Test Metadata False", falseTest);
  // a metadata object
  TAU_METADATA_OBJECT(objectTest, "Test Metadata Object", doubleTest);
  TAU_METADATA_OBJECT_PUT(objectTest, "Test Metadata String", stringTest);
  TAU_METADATA_OBJECT_PUT(objectTest, "Test Metadata Integer", intTest);
  TAU_METADATA_OBJECT_PUT(objectTest, "Test Metadata Double", doubleTest);
  TAU_METADATA_OBJECT_PUT(objectTest, "Test Metadata Null", nullTest);
  TAU_METADATA_OBJECT_PUT(objectTest, "Test Metadata True", trueTest);
  TAU_METADATA_OBJECT_PUT(objectTest, "Test Metadata False", falseTest);
  TAU_STRUCTURED_METADATA("Test Metadata Object", objectTest);
  // a metadata array of values of different types!
  TAU_METADATA_ARRAY(arrayTest, 7);
  TAU_METADATA_ARRAY_PUT(arrayTest, 0, stringTest);
  TAU_METADATA_ARRAY_PUT(arrayTest, 1, intTest);
  TAU_METADATA_ARRAY_PUT(arrayTest, 2, doubleTest);
  TAU_METADATA_ARRAY_PUT(arrayTest, 3, nullTest);
  TAU_METADATA_ARRAY_PUT(arrayTest, 4, trueTest);
  TAU_METADATA_ARRAY_PUT(arrayTest, 5, falseTest);
  TAU_METADATA_ARRAY_PUT(arrayTest, 6, objectTest);
  TAU_STRUCTURED_METADATA("Test Metadata Array", arrayTest);
  return;
}

int main (int argc, char** argv) {
  int x = 0;
  TAU_PROFILE_TIMER(mt, "main()", "int (int, char **)", TAU_DEFAULT);
  TAU_PROFILE_SET_NODE(0);
  TAU_PROFILE_START(mt);

  makeMetaData();

#pragma omp parallel shared(x)
  {
	int max = 4;
    TAU_PROFILE_TIMER(pt, "Parallel Region", " " , TAU_DEFAULT);
    TAU_PROFILE_START(pt);
#ifdef _OPENMP
    max = max * omp_get_num_threads();
#endif
#pragma omp for schedule(static,1)
    for (x = 0 ; x < max ; x++) {
      TAU_PROFILE_TIMER(fl, "For loop", " " , TAU_DEFAULT);
      TAU_PROFILE_START(fl);
      foo (x);
      bar (x);
      TAU_PROFILE_STOP(fl);
    }
    TAU_PROFILE_STOP(pt);
  }
  TAU_PROFILE_STOP(mt);
  exit(0);
}
