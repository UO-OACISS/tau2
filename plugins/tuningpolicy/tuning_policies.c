
#include <mpi.h>
#include <Profile/Profiler.h>
#include <Profile/TauEnv.h>
#include <Profile/TauMpiTTypes.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/types.h>
#include <assert.h>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include "json.h"
#include "json_util.h"

#define MAX_BUF 128
#define MAX_SIZE_FIELD_VALUE 64
#define MAX_SIZE_RULE 32
#define MAX_NB_RULES 16
#define MAX_NB_VALUES 16

#define MAX_TREE_DEPTH 32

#if 0
enum node_enum_e
{
  ID = 0,
  NUMPVARS = 1,
  LOGIC = 2,
  OPERATION = 3,
  STMT = 4,
  CONDITION = 5,
  LOPERAND = 6,
  ROPERAND = 7,
  OPERATOR = 8,
  RESULT = 9,
  ELSE = 10
};
#endif

enum node_enum_e
{
  OPEQUALS = 0,
  OPUPPER = 1,
  OPLOWER = 2,
  OPUPPEREQUAL = 3,
  OPLOWEREQUAL = 4,
  LOPERAND = 5,
  ROPRAND = 6
};

typedef enum node_enum_e node_enum_t;

enum operand_enum_e
{
  pvar,
  number
};

typedef enum operand_enum_e operand_enum_t;

enum stmt_enum_e
{
  IF,
  WHILE
};

typedef enum stmt_enum_e stmt_enum_t;

enum operator_enum_e
{
  EQUALS,
  UPPER,
  LOWER,
  UPPEREQUAL,
  LOWEREQUAL
};

typedef enum operator_enum_e operator_enum_t;

enum res_operator_enum_e
{
  EQUAL,
  INCR,
  DECR
};

typedef enum res_operator_enum_e res_operator_enum_t;

struct mpit_pvar_s
{
 char *name;
 int is_array;
 int size;
} mpit_pvar;

typedef struct mpit_pvar_s mpit_pvar_t;

struct mpit_cvar_s
{
 char *name;
 int is_array;
 int size;
} mpit_cvar;

typedef struct mpit_cvar_s mpit_cvar_t;

struct mpit_var_s
{
 char *name;
 int is_array;
 int size;
 int is_pvar;
} mpit_var;

typedef struct mpit_var_s mpit_var_t;

struct operand_s
{
  //char *value;
  int value;
  operand_enum_t type;
};

typedef struct operand_s operand_t;

struct groupoperand_s
{
 struct operand_s leftop;
 struct operand_s rightop;
 enum operator_enum_e op; 
};

typedef struct groupoperand_s groupoperand_t;

struct node_s
{
 node_enum_t nodeType;
 
 struct node_s *loperand;
 struct node_s *roperand;
 
 //operand_t *loperand;
 //operand_t *roperand;
 
 operator_enum_t ope;
};

typedef struct node_s node_t;

struct condition_s
{
  //char *stmt;
  enum stmt_enum_e stmt;

  node_t *root; // Tree containing operands and operators

#if 0  
  union {
    struct groupoperand_s leftgroupoperand;
    struct operand_s leftop; 
  };
#endif

#if 0
  union {
    struct groupoperand_s rightgroupoperand;
    struct operand_s rightop;
  };
#endif

  enum operator_enum_e ope;  
};

typedef struct condition_s condition_t;

struct res_s
{

#if 0
  union {
    struct groupoperand_s resleftgroupoperand;
    struct operand_s resleftop; 
  };

  union {
    struct groupoperand_s resrightgroupoperand;
    struct operand_s resrightop;
  };
#endif

  node_t *root; // Tree containing operands and operators

  enum operator_enum_e resoperator;  
};

typedef struct res_s res_t;

struct loop_s
{
  int size;
};

typedef struct loop_s loop_t;

struct op_s
{
 struct loop_s loop; 
 int is_pvar_array;
 int array_size;
 condition_t *cond;
 int num_pvars;
 node_t *result;
 node_t *elseresult;
};

typedef struct op_s op_t;

struct logic_s
{ 
  int is_pvar_array;
  int array_size;
  int num_ops;
  int num_pvars;
  op_t *op;
};

typedef struct logic_s logic_t;

struct tuning_policy_rule_s
{
  struct mpit_var_s *pvars;
  struct mpit_var_s *cvars;
  int num_pvars;
  int is_array_pvar;
  struct op_s *op;
};

typedef struct tuning_policy_rule_s tuning_policy_rule_t;


#define LEFTOPPLUS(leftop,rightop) \
	return leftop + rightop

#define LEFTOPMINUS(leftop,rightop) \
	return leftop - rightop

#define LEFTOPTIMES(leftop,rightop) \
	return leftop * rightop

#define LEFTOPDIV(leftop,rightop) \
	return leftop / rightop

#define OPEQ(leftop,rightop) \
        leftop == rightop	

#define OPLOWER(leftop,rightop) \
	leftop < rightop	

#define OPUPPER(leftop,rightop) \
	leftop > rightop	

#define OPLOWEQ(leftop,rightop) \
	leftop <= rightop	

#define OPUPEQ(leftop,rightop) \
	leftop >= rightop	

#define RESULT(root) \
        printf("RESULT\n")
 	//resleftop = root->loperand;

#define EVALEQ(leftop,op,rightop) leftop = rightop ? 1 : 0

#define EVALUPEQ(leftop,op,rightop) leftop >= rightop ? 1: 0

#define EVALUPPER(leftop,op,rightop) leftop > rightop ? 1 : 0

#define EVALLOWER(leftop,op,rightop) leftop < rightop ? 1 : 0

#define EVALLOWEQ(leftop,op,rightop) leftop <= rightop ? 1 : 0

#define EVAL(node) \
        int curr_depth = 0; \
 	while((node->loperand != NULL) && (node->roperand != NULL)i && (curr_depth < MAX_TREE_DEPTH)) { \
          /* Store tree for operation */ \ 
        }

#define IFSTMT(leftop,op,rightop) \
	if(EVALEQ(leftop->value,op,rightop->value)) { \
	  return 1; \
        } \
        else { \
          return 0; \
        }

#define WHILESTMT(leftop,op,rightop) \
	while(EVALEQ(leftop,op,rightop)) { \
          return 1; \
       } else { \
         return 0; \
       }

#define RESOPEQ(leftop,rightop) \
	leftop = rightop

#define RESOPPLUSEQ(leftop,rightop) \
	leftop += rightop

#define RESOPMINUSEQ(leftop,rightop) \
	leftop -= rightop

#define RESOPTIMESEQ(leftop,rightop) \
	leftop *= rightop

/*
#define FOREACH(limit) \
	for(i=0;i<limit;i++) {	\
       	
	}
*/

#define LEFTOPERAND(leftop,rightop,operator) \
	leftop operator rightop

//#define POPTREE(root) \

#define CONDITION3(stmt,leftoperand,rightoperand,ope) \
 	IFSTMT(leftoperand,ope,rightoperand)
//	stmt(leftoperand operator rightoperand)

#define CONDITION(stmt,node) \
	printf("CONDITION: POP tree\n");

#define CONDITION2(stmt,root) \
	IFSTMT(root->loperand,root->ope,root->roperand)

#define RESULT2(resleftop,resrightop,operator) \
	resleftop operator resrightop;

#define ELSE2(resleftop,resrightop,operator) \
	resleftop operator resrightop;

#define ELSE(resleftop,resrightop,operator) \
        resleftop = resrightop;

#define FOR(bound) \
	for(i=0; i<tau_pvar_count[bound]; i++)

#if 0
#define WRITECVARS(op,metric_string,value_string) \
        condition_t *cond = op->cond; \
	CONDITION(cond->stmt,cond->root) { \
          res_t *res = op->result; \
          RESULT(res->root); \
          sprintf(metric_string,"%d",res->root); \
        } \
        if(op->elseresult != NULL) { \
          res_t *elseres = op->elseresult; \
        }

#define INNEROP(op) \
        condition_t *cond = op->cond; \
	CONDITION(cond->stmt,cond->root) { \
          node_t *res = op->result; \
          RESULT(res); \
        } \
        if(op->elseresult != NULL) { \
          node_t *elseres = op->elseresult; \
          RESULT(elseres); \
 	} 

#define INNERLOGIC(op) \
	if(op->is_pvar_array == 1) { \
          unsigned long long int *value_array = (unsigned long long int *)calloc(tau_pvar_count[op->array_size],sizeof(unsigned long long int)); \
          char *value_cvar_string = (char *)malloc(sizeof(char)*TAU_NAME_LENGTH); \
          strcpy(value_cvar_string,""); \
          char *value_cvar_value_string = (char *)malloc(sizeof(char)*TAU_NAME_LENGTH); \
          strcpy(value_cvar_value_string,""); \
          FOR(op->array_size) { \
            INNEROP(op); \
            for(j=0; j<tau_pvar_count[op->num_pvars]; j++) { \
              if(i == (tau_pvar_count[j])) { \
                sprintf(metric_string, "%s[%d]", op->result, i); \
                sprintf(value_string, "%llu", value_array[i]); \
              } \
            } \
            strcat(value_cvar_string,metric_string); \
            strcat(value_cvar_value_string,value_string); \
          } \
        } else { \
          unsigned long long int value; \
	  char *value_cvar_string = (char *)malloc(sizeof(char)*TAU_NAME_LENGTH); \
          strcpy(value_cvar_string,""); \
          char *value_cvar_value_string = (char *)malloc(sizeof(char)*TAU_NAME_LENGTH); \
          strcpy(value_cvar_value_string,""); \
          INNEROP(op); \
          for(j=0; j<tau_pvar_count[op->num_pvars]; j++) { \
            if(i == (tau_pvar_count[j]))  { \
              sprintf(metric_string,"%s", op->result); \
              sprintf(value_string, "%llu", value); \
            } \
          } \
          strcat(value_cvar_string,metric_string); \
          strcat(value_cvar_value_string,value_string); \
        } 
#endif

/*
struct tuning_policy_rule_s
{
 struct mpit_var_s *pvars;
 struct mpit_var_s *cvars;
 int num_pvars;
 char *condition;
 struct leftop_s *leftoperand;
 char *rightoperand;
 char *operator;
 char *value;
 char *logicop;
 struct mpit_cvar_s *resleftoperand;
 //char *resleftoperand;
 char *resoperator;
 char *resrightoperand;
};
*/

typedef struct tuning_policy_rule_s tuning_policy_rule_t;
//void plugin_tuning_policies(int argc, void **args)

tuning_policy_rule_t rules[MAX_NB_RULES];

//static json_object *jso = NULL;

void innerop(struct op_s *op)
{
        condition_t *cond = op->cond; 
	CONDITION(cond->stmt,cond->root) { 
          node_t *res = op->result; 
          RESULT(res); 
        } 
        if(op->elseresult != NULL) { 
          node_t *elseres = op->elseresult; 
          RESULT(elseres); 
 	} 
}

void outerop(struct op_s *op)
{
  char metric_string[TAU_NAME_LENGTH], value_string[TAU_NAME_LENGTH]; 
  int *tau_pvar_count = NULL;
  int i=0, j=0;

  if(op->is_pvar_array == 1) { 
    unsigned long long int *value_array = (unsigned long long int *)calloc(tau_pvar_count[op->array_size],sizeof(unsigned long long int)); 
    char *value_cvar_string = (char *)malloc(sizeof(char)*TAU_NAME_LENGTH); 
    strcpy(value_cvar_string,""); 
    char *value_cvar_value_string = (char *)malloc(sizeof(char)*TAU_NAME_LENGTH); 
    strcpy(value_cvar_value_string,""); 
    for(i=0; i<op->array_size; i++) {
      innerop(op); 
      for(j=0; j<tau_pvar_count[op->num_pvars]; j++) { 
        if(i == (tau_pvar_count[j])) { 
          sprintf(metric_string, "%s[%d]", op->result, i); 
          sprintf(value_string, "%llu", value_array[i]); 
        } 
      } 
      strcat(value_cvar_string,metric_string); 
      strcat(value_cvar_value_string,value_string); 
   } 
  } else { 
    unsigned long long int value; 
    char *value_cvar_string = (char *)malloc(sizeof(char)*TAU_NAME_LENGTH); 
    strcpy(value_cvar_string,""); 
    char *value_cvar_value_string = (char *)malloc(sizeof(char)*TAU_NAME_LENGTH); 
    strcpy(value_cvar_value_string,""); 
    innerop(op); 
    for(j=0; j<tau_pvar_count[op->num_pvars]; j++) { 
      if(i == (tau_pvar_count[j]))  { 
        sprintf(metric_string,"%s", op->result); 
        sprintf(value_string, "%llu", value); 
      } 
   } 
   strcat(value_cvar_string,metric_string); 
   strcat(value_cvar_value_string,value_string); 
  }  

}


extern "C" int Tau_mpi_t_parse_and_write_cvars(const char *cvar_metrics, const char *cvar_values);

int parse_logic(node_t *tree)
{

 return 1;
}

/* Detect if given PVAR or CVAR is an array */
int detect_array(char *value, char *separator, mpit_var_t *var, int is_pvar)
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

#if 0
/* Analyze each element of leftoperand field */
int analyze_leftoperand(char *leftoperand, leftop_t *op)
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
int parse_list_leftop(char *value, char *separator, leftop_t *listleftops)
{
  int i = 0;
  char *token = strtok(value, separator);

  while(token != NULL)
  {
    leftop_t lop;
    token = strtok(NULL, separator);

    analyze_leftoperand(token, &lop);
    strcpy(lop.value, token);
    listleftops[i] = lop; 
    i += 1;
  }

  return 1;
}
#endif

/* Parse list of values for each field */
int parse_list_values(char *value, char *separator, mpit_var_t *listvars, int is_pvar)
{
  int i = 0;
  char *token = strtok(value, separator);
   
  while (token != NULL)
  {
    mpit_var_t var; 
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

#if 0
void json_parse_array( json_object *jobj, char *key) 
{
  void json_parse(json_object * jobj); /*Forward Declaration*/
  enum json_type type;

  json_object *jarray = jobj; /*Simply get the array*/

  if(key) {
    jarray = json_object_object_get(jobj, key); /*Getting the array if it is a key value pair*/
  }

  int arraylen = json_object_array_length(jarray); /*Getting the length of the array*/
  printf("Array Length: %dn",arraylen);
  int i;
  json_object * jvalue;

  for (i=0; i< arraylen; i++){
    jvalue = json_object_array_get_idx(jarray, i); /*Getting the array element at position i*/
    type = json_object_get_type(jvalue);
    if (type == json_type_array) {
      json_parse_array(jvalue, NULL);
    }
    else if (type != json_type_object) {
      printf("value[%d]: ",i);
      print_json_value(jvalue);
    }
    else {
      json_parse(jvalue);
    }
  }
}

/*Parsing the json object*/
void json_parse(json_object * jobj) 
{

  enum json_type type;

  json_object_object_foreach(jobj, key, val) { /*Passing through every array element*/

    printf("type: ",type);
    type = json_object_get_type(val);

    switch (type) {
      case json_type_boolean: 
      case json_type_double: 
      case json_type_int: 
      case json_type_string: print_json_value(val);
                           break; 
      case json_type_object: printf("json_type_objectn");
                           jobj = json_object_object_get(jobj, key);
                           json_parse(jobj); 
                           break;
      case json_type_array: printf("type: json_type_array, ");
                          json_parse_array(jobj, key);
                          break;
    }
  }
} 
#endif

/* Load JSON file and store string into a JSON object  
 * DEPRECATED
 * */
void read_json_rules()
{
  const char *filename = "./policy.json";

  int d = open(filename, O_RDONLY, 0);

  if (d < 0)
  {
    fprintf(stderr,
            "FAIL: unable to open %s: %s\n",
            filename, strerror(errno));

    exit(EXIT_FAILURE);
 }

 json_object *jso = json_object_from_fd(d);

 if (jso != NULL)
 {
   printf("OK: json_object_from_fd(%s)=%s\n",
                       filename, json_object_to_json_string(jso));
   json_object_put(jso);
 }
 else
 {
   fprintf(stderr,
           "FAIL: unable to parse contents of %s: %s\n",
           filename, json_util_get_last_err());
 }
 close(d);

}

/*
 * Load JSON file using C++ Boost and store into proper structures
 */
void tuningpolicies_load_rules()
{

 
}

#if 0
/* Load policy rules from config file and populate dedicated structure */
void load_policy_rules(int argc, void **args)
{
 FILE *fp;
 char line[MAX_BUF];
 char fieldname[16];
 char fieldvalue[MAX_SIZE_FIELD_VALUE];
 mpit_var_t *pvars = NULL;
 mpit_var_t *cvars = NULL;
 leftop_t *listleftops = NULL;
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
       parse_list_leftop(value, ",", listleftops);
       strcpy(rules[irule].leftoperand,listleftops);
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
#endif


/* Generic function for tuning policies */
int generic_tuning_policy(int argc, void **args)
{
  int i, j, namelen, verb, varclass, bind;
  int return_val;
  //int threadsup;
  //int index;
  int rule_id = 0;
  int readonly, continuous, atomic;
  char event_name[TAU_NAME_LENGTH + 1] = "";
  char metric_string[TAU_NAME_LENGTH], value_string[TAU_NAME_LENGTH];
  int desc_len;
  char description[TAU_NAME_LENGTH + 1] = "";
  MPI_Datatype datatype;
  MPI_T_enum enumtype;

  static int firsttime = 1;
  static int is_pvar_array = 0;

  //static unsigned long long int *cvar_value_array = NULL;
  //static char *cvar_string = NULL;
  //static char *cvar_value_string = NULL;
 
  assert(argc=3);

  const int num_pvars 				= (intptr_t)			(args[0]);
  int *tau_pvar_count 				= (int *)			(args[1]);
  //unsigned long long int **pvar_value_buffer 	= (unsigned long long int **)	(args[2]);

  int pvar_index[num_pvars];

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

      // Check pvar name match
      for(j=0; j<rules[rule_id].num_pvars; j++) { 
        if(strcmp(event_name, rules[rule_id].pvars[j].name) == 0) {
          pvar_index[j] = j;
          // Is considered pvar an array ?
          if(rules[rule_id].pvars[j].is_array == 1)
            is_pvar_array = 1;
        } 
      } //for
    } //for

    for(j=0; j<rules[rule_id].num_pvars; j++) {
      if(pvar_index[j] == -1) {
        printf("Unable to find the indexes of PVARs required for tuning\n");
        return -1;
      }
    }

  }

  /* Call the inner logic */  
  op_t *op = rules[rule_id].op;
  //logic_t logic = rules[rule_id].logic;

  outerop(op);
  //INNERLOGIC(op);

#if 0
  if(rules[rule_id].is_array_pvar == 1) {
    
    for(i=0; i<op.loop.size; i++) {
      INNERLOGIC(op);

      if(i == op.loop.size) {
        sprintf(metric_string,"%s[%d]", rules[rule_id].cvars, i);
        //sprintf(value_string,"%llu", reduced_value_array[i]);
      } else {
        sprintf(metric_string,"%s[%d],", rules[rule_id].cvars, i);
        //sprintf(value_string,"%llu,", reduced_value_array[i]);
      }

    }

  } else {
      INNERLOGIC(op);
      //sprintf(metric_string,"%s[%d]", rules[rule_id].cvar, i);
      //printf(value_string,"%llu", reduced_value_array[i]);
  } 
 
  for(j=0; j<rules[rule_id].num_pvars; j++) {
    if(j == (tau_pvar_count[j])) {}
  }
#endif 

  return 1;

}

/*Implement user based CVAR tuning policy based on a policy file (?)
 * TODO: This tuning logic should be in a separate module/file. Currently implementing hard-coded policies for MVAPICH meant only for experimentation purposes*/
//void Tau_enable_user_cvar_tuning_policy(const int num_pvars, int *tau_pvar_count, unsigned long long int **pvar_value_buffer) {
int plugin_tuning_policy(int argc, void **args) {

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

  const int num_pvars 				= (intptr_t)			(args[0]);
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
    return -1;
  } else {
    printf("dprintf\n");
    //dprintf("Index of %s is %d and index of %s is %d\n", PVAR_MAX_VBUF_USAGE, pvar_max_vbuf_usage_index, PVAR_VBUF_ALLOCATED, pvar_vbuf_allocated_index);
  }
 }

  /*Tuning logic: If the difference between allocated vbufs and max use vbufs in a given
 *   * vbuf pool is higher than a set threshhold, then we will free from that pool.*/
  for(i = 0 ; i < tau_pvar_count[pvar_max_vbuf_usage_index]; i++) {
    if(pvar_value_buffer[pvar_max_vbuf_usage_index][i] > 1000) pvar_value_buffer[pvar_max_vbuf_usage_index][i] = 0; /*HACK - we are getting garbage values for pool2. Doesn't seem to be an issue in TAU*/

    if((pvar_value_buffer[pvar_vbuf_allocated_index][i] - pvar_value_buffer[pvar_max_vbuf_usage_index][i]) > PVAR_VBUF_WASTED_THRESHOLD) {
      has_threshold_been_breached_in_any_pool = 1;
      reduced_value_array[i] = pvar_value_buffer[pvar_max_vbuf_usage_index][i];
      //dprintf("Threshold breached: Max usage for %d pool is %llu but vbufs allocated are %llu\n", i, pvar_value_buffer[pvar_max_vbuf_usage_index][i], pvar_value_buffer[pvar_vbuf_allocated_index][i]);
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
    //dprintf("Metric string is %s and value string is %s\n", metric_string, value_string);
    Tau_mpi_t_parse_and_write_cvars(metric_string, value_string);
  } else {
    sprintf(metric_string,"%s", CVAR_ENABLING_POOL_CONTROL);
    sprintf(value_string,"%d", 0);
    //dprintf("Metric string is %s and value string is %s\n", metric_string, value_string);
    Tau_mpi_t_parse_and_write_cvars(metric_string, value_string);
  }
 
  return return_val;
}
