#include <TAU.h>
#include <Profile/TauMetaData.h>
#include <Profile/TauMetaDataMerge.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void checkMetadata(const char *name, const char *expected) {
  char *actual = Tau_metadata_get(name, 0);
  if (expected == NULL) {
    if (actual != NULL) {
      fprintf(stderr, "Expected metadata %s to be absent, got '%s'\n", name, actual);
      free(actual);
      exit(EXIT_FAILURE);
    }
  } else if (actual == NULL || strcmp(actual, expected) != 0) {
    fprintf(stderr, "Unexpected metadata value for %s: expected '%s', got '%s'\n",
            name, expected, actual == NULL ? "(absent)" : actual);
    free(actual);
    exit(EXIT_FAILURE);
  }
  free(actual);
}

static void clearSlurmEnvironment(void) {
  unsetenv("SLURM_TOPOLOGY_ADDR");
  unsetenv("SLURM_TOPOLOGY_ADDR_PATTERN");
  unsetenv("SLURM_LOCALID");
}

int main(int argc, char **argv) {
  TAU_PROFILE_TIMER(timer, "slurm_metadata_test", "void()", TAU_DEFAULT);
  if (argc != 2) {
    fprintf(stderr, "Usage: %s absent|complete|partial|repeat|equivalent\n", argv[0]);
    return EXIT_FAILURE;
  }

  clearSlurmEnvironment();
  const char *address = NULL;
  const char *pattern = NULL;
  const char *localid = NULL;
  bool repeated = false;
  bool equivalent_local_rank = false;

  if (strcmp(argv[1], "complete") == 0) {
    address = "root.s2610.frontier10247";
    pattern = "switch.switch.node";
    localid = "12";
  } else if (strcmp(argv[1], "partial") == 0) {
    address = "root.s2610.frontier10248";
    pattern = "";
    localid = "";
  } else if (strcmp(argv[1], "repeat") == 0) {
    address = "root.s2610.frontier10247";
    pattern = "switch.switch.node";
    localid = "12";
    repeated = true;
  } else if (strcmp(argv[1], "equivalent") == 0) {
    localid = "12";
    equivalent_local_rank = true;
  } else if (strcmp(argv[1], "absent") != 0) {
    fprintf(stderr, "Unknown test case: %s\n", argv[1]);
    return EXIT_FAILURE;
  }

  if (address != NULL) {
    setenv("SLURM_TOPOLOGY_ADDR", address, 1);
  }
  if (pattern != NULL) {
    setenv("SLURM_TOPOLOGY_ADDR_PATTERN", pattern, 1);
  }
  if (localid != NULL) {
    setenv("SLURM_LOCALID", localid, 1);
  }

  TAU_PROFILE_SET_NODE(0);
  TAU_PROFILE_START(timer);

  if (equivalent_local_rank) {
    Tau_metadata_register("CRAY_PMI_LOCAL_RANK", "3");
  }
  writeMetaDataAfterMPI_Init();
  if (repeated) {
    writeMetaDataAfterMPI_Init();
  }

  checkMetadata("SLURM_TOPOLOGY_ADDR", address);
  checkMetadata("SLURM_TOPOLOGY_ADDR_PATTERN",
                pattern != NULL && pattern[0] != '\0' ? pattern : NULL);
  checkMetadata("SLURM_LOCALID",
                equivalent_local_rank || localid == NULL || localid[0] == '\0'
                    ? NULL
                    : localid);

  TAU_PROFILE_STOP(timer);
  return EXIT_SUCCESS;
}
