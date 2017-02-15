
#include <mpi.h>
#include <Profile/Profiler.h>
#include <Profile/TauEnv.h>
#include <Profile/TauMpiTTypes.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#define MAX_BUF 128
#define MAX_SIZE_FIELD_VALUE 64
#define MAX_SIZE_RULE 32
#define MAX_NB_RULES 16
#define MAX_NB_VALUES 16

typedef enum leftop_enum_t
{
  pvar,
  sign,
  number
} leftop_enum;

typedef struct mpit_pvar_t
{
 char *name;
 int is_array;
 int size;
} mpit_pvar;

typedef struct mpit_cvar_t
{
 char *name;
 int is_array;
 int size;
} mpit_cvar;

typedef struct mpit_var_t
{
 char *name;
 int is_array;
 int size;
 int is_pvar;
} mpit_var;

typedef struct leftop_t
{
  char *value;
  leftop_enum type;
} leftop;

typedef struct tuning_policy_rule_
{
 mpit_var *pvars;
 mpit_var *cvars;
 int num_pvars;
 char *condition;
 leftop *leftoperand;
 char *rightoperand;
 char *operator;
 char *value;
 char *logicop;
 char *resleftoperand;
 char *resoperator;
 char *resrightoperand;
} tuning_policy_rule;

//void plugin_tuning_policies(int argc, void **args)

tuning_policy_rule rules[MAX_NB_RULES];

/* Detect if given PVAR or CVAR is an array */
int detect_array(char *value, char *separator, mpit_var *var, int is_pvar)
{
  char *token;
  char *rightpart;
  char *name;
  int size;
  int is_array = 0;
  int i = 0;

  // Check if considered PVAR/CVAR is an array
  while(i < MAX_SIZE_FIELD_VALUE) {
    if(value[i] == '[') { is_array = 1; }
    i++;
  } 

  if(is_array) {
    if(strcmp(separator,"[") == 0) 
    {
      // Get field name 
      token = strtok(value, separator); 
      fprintf(stdout, "Name of array PVAR/CVAR: %s\n", token); 
      strcpy(name,token);
      // Get field name 
      token = strtok(NULL, separator);
      strcpy(rightpart,token);
      //strcpy(size,token);
    }  

    token = strtok(rightpart, "]");
    fprintf(stdout, "Size of PVAR/CVAR array:%s\n", token);
    size = atoi(token); 
  } else {
    strcpy(name,value);
    fprintf(stdout, "Name of name PVAR/CVAR: %s\n", name); 
    size = 0;
  } 
  
  strcpy(var->name, name);
  var->is_array = is_array;
  var->size = size;
  var->is_pvar = is_pvar; 

  return is_array;
}

int analyze_leftoperand(char *leftoperand, leftop *op)
{

  if(strncmp(leftoperand, "+", 1) == 0 || strncmp(leftoperand, "-", 1) == 0 || strncmp(leftoperand, "*", 1) == 0) {
   op->type = sign;
  } else if(atol(leftoperand) != 0L) {
   op->type = number;
  } else {
    op->type = pvar;
  }

  strcpy(op->value,leftoperand);
  
  return 1;
}

/* Parse list of values for each leftoperand list */
int parse_list_leftop(char *value, char *separator, leftop *listleftops, int size)
{
  int i = 0;
  char *token = strtok(value, separator);

  while(token != NULL)
  {
    leftop lop;
    token = strtok(NULL, separator);

    analyze_leftoperand(token, &lop);
    strcpy(lop.value, token);
    listleftops[i] = lop; 
    i += 1;
  }

  return 1;
}

/* Parse list of values for each field */
int parse_list_values(char *value, char *separator, mpit_var *listvars, int is_pvar)
{
  int i = 0;
  char *token = strtok(value, separator);
   
  while (token != NULL)
  {
    mpit_var var; 
    printf("%s\n", token);
    token = strtok(NULL, separator);
   
    detect_array(token, "[", &var, is_pvar); 
    
    listvars[i] = var; 
    i += 1;
  }

  return 1;
}

/* Parse field into 2 components: key and value */
int parse_rule_field(char *line, char *separator, char *key, char *value)
{
  char *token;
  //char separator[2] = ":";

  if(strcmp(separator,":") == 0) 
  {
   /* Get field name */
   token = strtok(line, separator); 
   strcpy(key,token);
   /* Get field name */
   token = strtok(NULL, separator);
   strcpy(value,token);
  }

  return 1;
}

/* Load policy rules from config file and populate dedicated structure */
void load_policy_rules(int argc, void **args)
{
 FILE *fp;
 char line[MAX_BUF];
 char fieldname[16];
 char fieldvalue[MAX_SIZE_FIELD_VALUE];
 mpit_var *pvars = NULL;
 mpit_var *cvars = NULL;
 char *token;
 char *key = NULL;
 //char key[16];
 //char value[16];
 char *value = NULL;
 char separator[2] = ":";
 int irule = -1;

 fprintf(stdout, "Tuning policies DSO init.....\n");

 fp=fopen("policy.conf","r");

 if(fp != NULL) 
 {
   // Read configuration file, parse lines, and populate rule structure
   while(fgets(line, sizeof(line), fp) != NULL) 
   {
     if(strncmp(line,"RULE",4) == 0) {    
       irule += 1; 
     }
     if(strncmp(line,"PVARS",5) == 0) {
       parse_rule_field(line, separator, key, value);
       parse_list_values(value, ",", pvars, 0);
       rules[irule].pvars = pvars;
       //strcpy(rules[irule].pvars,pvars);
     } 
     if(strncmp(line,"CVARS",5) == 0) {
       parse_rule_field(line, separator, key, value);
       parse_list_values(value, ",", cvars, 1);
       rules[irule].cvars = cvars;
       //strcpy(rules[irule].cvars,cvars);
     }
     if(strncmp(line,"STMT",4) == 0) {
       parse_rule_field(line, separator, key, value);
       strcpy(rules[irule].condition,value);
     } 
     if(strncmp(line,"LEFTOPERAND",11) == 0) {
       parse_rule_field(line, separator, key, value);
       strcpy(rules[irule].leftoperand,value);
     }
     if(strncmp(line,"RIGHTOPERAND",12) == 0) {
       parse_rule_field(line, separator, key, value);
       strcpy(rules[irule].rightoperand,value);
     }
     if(strncmp(line,"OPERATOR",8) == 0) {
       parse_rule_field(line, separator, key, value);
       strcpy(rules[irule].operator,value);
     }
     if(strncmp(line,"LOGICOP",7) == 0) {
       parse_rule_field(line, separator, key, value);
     }
     if(strncmp(line,"RESLEFTOPERAND",14) == 0) {
       parse_rule_field(line, separator, key, value);
       strcpy(rules[irule].resleftoperand,value);
     }
     if(strncmp(line,"RESRIGHTOPERAND",15) == 0) {
       parse_rule_field(line, separator, key, value);
       strcpy(rules[irule].resrightoperand,value);
     }
     if(strncmp(line,"RESOPERATOR",11) == 0) {
       parse_rule_field(line, separator, key, value);
       strcpy(rules[irule].resoperator,value);
     }

   } // End while 
 } // End if 

 fclose(fp);
}

/* Generic function for tuning policies */
void generic_tuning_policy(int argc, void **args)
{
  int return_val, i, j, namelen, verb, varclass, bind, threadsup;
  int index;
  int readonly, continuous, atomic;
  char event_name[TAU_NAME_LENGTH + 1] = "";
  char metric_string[TAU_NAME_LENGTH], value_string[TAU_NAME_LENGTH];
  int desc_len;
  char description[TAU_NAME_LENGTH + 1] = "";
  MPI_Datatype datatype;
  MPI_T_enum enumtype;
  static int firsttime = 1;
  static unsigned long long int *reduced_value_array = NULL;
  static char *reduced_value_cvar_string = NULL;
  static char *reduced_value_cvar_value_string = NULL;
 
  assert(argc=3);

  const int num_pvars 				= (const int)			(args[0]);
  int *tau_pvar_count 				= (int *)			(args[1]);
  unsigned long long int **pvar_value_buffer 	= (unsigned long long int **)	(args[2]);

  if(firsttime) {
    firsttime = 0;
    for(i = 0; i < num_pvars; i++){
      namelen = desc_len = TAU_NAME_LENGTH;
      return_val = MPI_T_pvar_get_info(i/*IN*/,
      event_name /*OUT*/,
      &namelen /*INOUT*/,
      &verb /*OUT*/,
      &varclass /*OUT*/,
      &datatype /*OUT*/,
      &enumtype /*OUT*/,
      description /*description: OUT*/,
      &desc_len /*desc_len: INOUT*/,
      &bind /*OUT*/,
      &readonly /*OUT*/,
      &continuous /*OUT*/,
      &atomic/*OUT*/);

      for(j=0; j<rules[0].num_pvars; j++) {
        //if(strcmp(event_name, rules[0].pvars[j])) {

        //}
      }
    }
  }

/*
  if((pvar_max_vbuf_usage_index == -1) || (pvar_vbuf_allocated_index == -1)) {
    printf("Unable to find the indexes of PVARs required for tuning\n");
    return;
  } else {
    dprintf("Index of %s is %d and index of %s is %d\n", PVAR_MAX_VBUF_USAGE, pvar_max_vbuf_usage_index, PVAR_VBUF_ALLOCATED, pvar_vbuf_allocated_index);
  }
 }
*/

}

void plugin_generic_tuning_policy(int argc, void **args)
{
  int return_val, i, namelen, verb, varclass, bind, threadsup;
  int index;
  int readonly, continuous, atomic;
  char event_name[TAU_NAME_LENGTH + 1] = "";
  char metric_string[TAU_NAME_LENGTH], value_string[TAU_NAME_LENGTH];
  int desc_len;
  char description[TAU_NAME_LENGTH + 1] = "";
  MPI_Datatype datatype;
  MPI_T_enum enumtype;
  static int firsttime = 1;
  static unsigned long long int *reduced_value_array = NULL;
  static char *reduced_value_cvar_string = NULL;
  static char *reduced_value_cvar_value_string = NULL;
  
  fprintf(stdout, "plugin tuning policy ...\n");

  assert(argc=3);

  const int num_pvars 				= (const int)			(args[0]);
  int *tau_pvar_count 				= (int *)			(args[1]);
  unsigned long long int **pvar_value_buffer 	= (unsigned long long int **)	(args[2]);

  /*MVAPICH specific thresholds and names*/
  char PVAR_MAX_VBUF_USAGE[TAU_NAME_LENGTH] = "mv2_vbuf_max_use_array";
  char PVAR_VBUF_ALLOCATED[TAU_NAME_LENGTH] = "mv2_vbuf_allocated_array";
  int PVAR_VBUF_WASTED_THRESHOLD = 10; //This is the threshold above which we will be free from the pool

  char CVAR_ENABLING_POOL_CONTROL[TAU_NAME_LENGTH] = "MPIR_CVAR_VBUF_POOL_CONTROL";
  char CVAR_SPECIFYING_REDUCED_POOL_SIZE[TAU_NAME_LENGTH] = "MPIR_CVAR_VBUF_POOL_REDUCED_VALUE";

  int pvar_max_vbuf_usage_index, pvar_vbuf_allocated_index, has_threshold_been_breached_in_any_pool;
  pvar_max_vbuf_usage_index = -1;
  pvar_vbuf_allocated_index = -1;
  has_threshold_been_breached_in_any_pool = 0;

  if(firsttime) {
    firsttime = 0;
    for(i = 0; i < num_pvars; i++){
      namelen = desc_len = TAU_NAME_LENGTH;
      return_val = MPI_T_pvar_get_info(i/*IN*/,
      event_name /*OUT*/,
      &namelen /*INOUT*/,
      &verb /*OUT*/,
      &varclass /*OUT*/,
      &datatype /*OUT*/,
      &enumtype /*OUT*/,
      description /*description: OUT*/,
      &desc_len /*desc_len: INOUT*/,
      &bind /*OUT*/,
      &readonly /*OUT*/,
      &continuous /*OUT*/,
      &atomic/*OUT*/);

      if(strcmp(event_name, PVAR_MAX_VBUF_USAGE) == 0) {
        pvar_max_vbuf_usage_index = i;
      } else if (strcmp(event_name, PVAR_VBUF_ALLOCATED) == 0) {
        pvar_vbuf_allocated_index = i;
      }
      reduced_value_array = (unsigned long long int *)calloc(sizeof(unsigned long long int), tau_pvar_count[pvar_max_vbuf_usage_index]);
      reduced_value_cvar_string = (char *)malloc(sizeof(char)*TAU_NAME_LENGTH);
      strcpy(reduced_value_cvar_string, "");
      reduced_value_cvar_value_string = (char *)malloc(sizeof(char)*TAU_NAME_LENGTH);
      strcpy(reduced_value_cvar_value_string, "");
  }

  if((pvar_max_vbuf_usage_index == -1) || (pvar_vbuf_allocated_index == -1)) {
    printf("Unable to find the indexes of PVARs required for tuning\n");
    return;
  } else {
    dprintf("Index of %s is %d and index of %s is %d\n", PVAR_MAX_VBUF_USAGE, pvar_max_vbuf_usage_index, PVAR_VBUF_ALLOCATED, pvar_vbuf_allocated_index);
  }
 }
  /*Tuning logic: If the difference between allocated vbufs and max use vbufs in a given
 *   * vbuf pool is higher than a set threshhold, then we will free from that pool.*/
  for(i = 0 ; i < tau_pvar_count[pvar_max_vbuf_usage_index]; i++) {
    if(pvar_value_buffer[pvar_max_vbuf_usage_index][i] > 1000) pvar_value_buffer[pvar_max_vbuf_usage_index][i] = 0; /*HACK - we are getting garbage values for pool2. Doesn't seem to be an issue in TAU*/

    if((pvar_value_buffer[pvar_vbuf_allocated_index][i] - pvar_value_buffer[pvar_max_vbuf_usage_index][i]) > PVAR_VBUF_WASTED_THRESHOLD) {
      has_threshold_been_breached_in_any_pool = 1;
      reduced_value_array[i] = pvar_value_buffer[pvar_max_vbuf_usage_index][i];
      dprintf("Threshold breached: Max usage for %d pool is %llu but vbufs allocated are %llu\n", i, pvar_value_buffer[pvar_max_vbuf_usage_index][i], pvar_value_buffer[pvar_vbuf_allocated_index][i]);
    } else {
      reduced_value_array[i] = pvar_value_buffer[pvar_vbuf_allocated_index][i] + 10; //Some value higher than current allocated
    }

    if(i == (tau_pvar_count[pvar_max_vbuf_usage_index])) {
      sprintf(metric_string,"%s[%d]", CVAR_SPECIFYING_REDUCED_POOL_SIZE, i);
      sprintf(value_string,"%llu", reduced_value_array[i]);
    } else {
      sprintf(metric_string,"%s[%d],", CVAR_SPECIFYING_REDUCED_POOL_SIZE, i);
      sprintf(value_string,"%llu,", reduced_value_array[i]);
    }
    
    strcat(reduced_value_cvar_string, metric_string);
    strcat(reduced_value_cvar_value_string, value_string);

  }

  if(has_threshold_been_breached_in_any_pool) {
    sprintf(metric_string,"%s,%s", CVAR_ENABLING_POOL_CONTROL, reduced_value_cvar_string);
    sprintf(value_string,"%d,%s", 1, reduced_value_cvar_value_string);
    dprintf("Metric string is %s and value string is %s\n", metric_string, value_string);
    Tau_mpi_t_parse_and_write_cvars(metric_string, value_string);
  } else {
    sprintf(metric_string,"%s", CVAR_ENABLING_POOL_CONTROL);
    sprintf(value_string,"%d", 0);
    dprintf("Metric string is %s and value string is %s\n", metric_string, value_string);
    Tau_mpi_t_parse_and_write_cvars(metric_string, value_string);
  }

}

/*Implement user based CVAR tuning policy based on a policy file (?)
 * TODO: This tuning logic should be in a separate module/file. Currently implementing hard-coded policies for MVAPICH meant only for experimentation purposes*/
//void Tau_enable_user_cvar_tuning_policy(const int num_pvars, int *tau_pvar_count, unsigned long long int **pvar_value_buffer) {
void plugin_tuning_policy(int argc, void **args) {

  int return_val, i, namelen, verb, varclass, bind, threadsup;
  int index;
  int readonly, continuous, atomic;
  char event_name[TAU_NAME_LENGTH + 1] = "";
  char metric_string[TAU_NAME_LENGTH], value_string[TAU_NAME_LENGTH];
  int desc_len;
  char description[TAU_NAME_LENGTH + 1] = "";
  MPI_Datatype datatype;
  MPI_T_enum enumtype;
  static int firsttime = 1;
  static unsigned long long int *reduced_value_array = NULL;
  static char *reduced_value_cvar_string = NULL;
  static char *reduced_value_cvar_value_string = NULL;
  
  fprintf(stdout, "plugin tuning policy ...\n");

  assert(argc=3);

  const int num_pvars 				= (const int)(args[0]);
  int *tau_pvar_count 				= (int *)(args[1]);
  unsigned long long int **pvar_value_buffer 	= (unsigned long long int **)(args[2]);

  /*MVAPICH specific thresholds and names*/
  char PVAR_MAX_VBUF_USAGE[TAU_NAME_LENGTH] = "mv2_vbuf_max_use_array";
  char PVAR_VBUF_ALLOCATED[TAU_NAME_LENGTH] = "mv2_vbuf_allocated_array";
  int PVAR_VBUF_WASTED_THRESHOLD = 10; //This is the threshold above which we will be free from the pool

  char CVAR_ENABLING_POOL_CONTROL[TAU_NAME_LENGTH] = "MPIR_CVAR_VBUF_POOL_CONTROL";
  char CVAR_SPECIFYING_REDUCED_POOL_SIZE[TAU_NAME_LENGTH] = "MPIR_CVAR_VBUF_POOL_REDUCED_VALUE";

  int pvar_max_vbuf_usage_index, pvar_vbuf_allocated_index, has_threshold_been_breached_in_any_pool;
  pvar_max_vbuf_usage_index = -1;
  pvar_vbuf_allocated_index = -1;
  has_threshold_been_breached_in_any_pool = 0;

 if(firsttime) {
  firsttime = 0;
  for(i = 0; i < num_pvars; i++){
      namelen = desc_len = TAU_NAME_LENGTH;
      return_val = MPI_T_pvar_get_info(i/*IN*/,
      event_name /*OUT*/,
      &namelen /*INOUT*/,
      &verb /*OUT*/,
      &varclass /*OUT*/,
      &datatype /*OUT*/,
      &enumtype /*OUT*/,
      description /*description: OUT*/,
      &desc_len /*desc_len: INOUT*/,
      &bind /*OUT*/,
      &readonly /*OUT*/,
      &continuous /*OUT*/,
      &atomic/*OUT*/);

      if(strcmp(event_name, PVAR_MAX_VBUF_USAGE) == 0) {
        pvar_max_vbuf_usage_index = i;
      } else if (strcmp(event_name, PVAR_VBUF_ALLOCATED) == 0) {
        pvar_vbuf_allocated_index = i;
      }
      reduced_value_array = (unsigned long long int *)calloc(sizeof(unsigned long long int), tau_pvar_count[pvar_max_vbuf_usage_index]);
      reduced_value_cvar_string = (char *)malloc(sizeof(char)*TAU_NAME_LENGTH);
      strcpy(reduced_value_cvar_string, "");
      reduced_value_cvar_value_string = (char *)malloc(sizeof(char)*TAU_NAME_LENGTH);
      strcpy(reduced_value_cvar_value_string, "");
  }

  if((pvar_max_vbuf_usage_index == -1) || (pvar_vbuf_allocated_index == -1)) {
    printf("Unable to find the indexes of PVARs required for tuning\n");
    return;
  } else {
    dprintf("Index of %s is %d and index of %s is %d\n", PVAR_MAX_VBUF_USAGE, pvar_max_vbuf_usage_index, PVAR_VBUF_ALLOCATED, pvar_vbuf_allocated_index);
  }
 }
  /*Tuning logic: If the difference between allocated vbufs and max use vbufs in a given
 *   * vbuf pool is higher than a set threshhold, then we will free from that pool.*/
  for(i = 0 ; i < tau_pvar_count[pvar_max_vbuf_usage_index]; i++) {
    if(pvar_value_buffer[pvar_max_vbuf_usage_index][i] > 1000) pvar_value_buffer[pvar_max_vbuf_usage_index][i] = 0; /*HACK - we are getting garbage values for pool2. Doesn't seem to be an issue in TAU*/

    if((pvar_value_buffer[pvar_vbuf_allocated_index][i] - pvar_value_buffer[pvar_max_vbuf_usage_index][i]) > PVAR_VBUF_WASTED_THRESHOLD) {
      has_threshold_been_breached_in_any_pool = 1;
      reduced_value_array[i] = pvar_value_buffer[pvar_max_vbuf_usage_index][i];
      dprintf("Threshold breached: Max usage for %d pool is %llu but vbufs allocated are %llu\n", i, pvar_value_buffer[pvar_max_vbuf_usage_index][i], pvar_value_buffer[pvar_vbuf_allocated_index][i]);
    } else {
      reduced_value_array[i] = pvar_value_buffer[pvar_vbuf_allocated_index][i] + 10; //Some value higher than current allocated
    }

    if(i == (tau_pvar_count[pvar_max_vbuf_usage_index])) {
      sprintf(metric_string,"%s[%d]", CVAR_SPECIFYING_REDUCED_POOL_SIZE, i);
      sprintf(value_string,"%llu", reduced_value_array[i]);
    } else {
      sprintf(metric_string,"%s[%d],", CVAR_SPECIFYING_REDUCED_POOL_SIZE, i);
      sprintf(value_string,"%llu,", reduced_value_array[i]);
    }
    
    strcat(reduced_value_cvar_string, metric_string);
    strcat(reduced_value_cvar_value_string, value_string);

  }

  if(has_threshold_been_breached_in_any_pool) {
    sprintf(metric_string,"%s,%s", CVAR_ENABLING_POOL_CONTROL, reduced_value_cvar_string);
    sprintf(value_string,"%d,%s", 1, reduced_value_cvar_value_string);
    dprintf("Metric string is %s and value string is %s\n", metric_string, value_string);
    Tau_mpi_t_parse_and_write_cvars(metric_string, value_string);
  } else {
    sprintf(metric_string,"%s", CVAR_ENABLING_POOL_CONTROL);
    sprintf(value_string,"%d", 0);
    dprintf("Metric string is %s and value string is %s\n", metric_string, value_string);
    Tau_mpi_t_parse_and_write_cvars(metric_string, value_string);
  }
}
