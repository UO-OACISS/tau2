
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


#include <json/json.h>

//#include <boost/property_tree/ptree.hpp>
//#include <boost/property_tree/json_parser.hpp>

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

using namespace std;

enum node_content_e
{
  OPEQUALS 	= 0,
  OPUPPER 	= 1,
  OPLOWER 	= 2,
  OPUPPEREQUAL 	= 3,
  OPLOWEREQUAL	= 4,
  OPEQUALASSIGN = 5,
  OPINCR 	= 6,
  OPDECR	= 7,
  LOPERAND 	= 8,
  ROPERAND 	= 7
};

typedef enum node_content_e node_content_t;

enum node_type_e
{
  NODE  = 0,
  LEAF  = 1
};

typedef enum node_type_e nodeType;

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
};

typedef struct mpit_pvar_s mpit_pvar_t;

struct mpit_cvar_s
{
 char *name;
 int is_array;
 int size;
};

typedef struct mpit_cvar_s mpit_cvar_t;

struct mpit_var_s
{
 char *name;
 int is_array;
 int size;
 int is_pvar;
};

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
 nodeType type;
 
 struct node_s *loperand;
 struct node_s *roperand;
 
 //operand_t *loperand;
 //operand_t *roperand;
 string data;
 
 operator_enum_t ope;
};

typedef struct node_s node_t;

struct condition_s
{
  string stmt;
  //stmt_enum_t stmt;

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

//tuning_policy_rule_t rules[MAX_NB_RULES];
<<<<<<< HEAD

static int rule_idx = 0;

class Op
{
public:
  Op(){}
  struct loop_s loop; 
  int is_pvar_array;
  int array_size;
  condition_t *cond;
  int num_pvars;
  node_t *result;
  node_t *elseresult;

private:

};

class Rule
{
public:
  Rule(){}
  int index;
  int num_pvars;  
  int is_array_pvar;
  struct mpit_var_s *pvars;
  Op op;

private:

};

Rule rules[MAX_NB_RULES];

static int rule_idx = 0;

class Op
{
public:
  Op(){}
  struct loop_s loop; 
  int is_pvar_array;
  int array_size;
  condition_t *cond;
  int num_pvars;
  node_t *result;
  node_t *elseresult;

private:

};

class Rule
{
public:
  Rule(){}
  int index;
  int num_pvars;  
  int is_array_pvar;
  struct mpit_var_s *pvars;
  Op op;

private:

};

Rule rules[MAX_NB_RULES];

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

#define STRTOINT(str,num) \
	for (int i=0; i<str.length();  i++) \
	  num = num*10 + (int(str[i])-48); \


#if 0
#define EVALEQ(node) \
        int intlop = 0, introp = 0; \
        STRTOINT(intlop,node->loperand->data); \
        STRTOINT(introp,node->roperand->data); \
        intlop = introp ? 1 : 0

#define EVALLOWER(node) \
        int intlop = 0, introp = 0; \
        STRTOINT(intlop,node->loperand->data); \
        STRTOINT(introp,node->roperand->data); \
        intlop = introp ? 1 : 0

#define EVALGREATER(node) \
        int intlop = 0, introp = 0; \
	STRTOINT(intlop,node->loperand->data); \
	STRTOINT(introp,node->roperand->data); \
	intlop = introp ? 1 : 0

#define EVALLOWEREQ(node) \
        int intlop = 0, introp = 0; \
        STRTOINT(intlop,node->loperand->data); \
        STRTOINT(introp,node->roperand->data); \
        intlop = introp ? 1 : 0

#define EVALGREATEREQ(node) \
        int intlop = 0, introp = 0; \
        STRTOINT(intlop,node->loperand->data); \
        STRTOINT(introp,node->roperand->data); \
        intlop = introp ? 1 : 0


#define EVALLOWER(node) \
	TOINT(node->loperand->data) < TOINT(node->roperand->data) ? 1 : 0

#define EVALGREATER(node) \
	TOINT(node->loperand->data) > TOINT(node->roperand->data) ? 1 : 0

#define EVALLOWEREQ(node) \
	TOINT(node->loperand->data) <= TOINT(node->roperand->data) ? 1 : 0

#define EVALGREATEREQ(node) \
	TOINT(node->loperand->data) >= TOINT(node->roperand->data) ? 1 : 0

#endif

#define IFSTMT(returnval) if(returnval)

#define WHILESTMT(returnval) while(returnval)

#define IFSTMT2(node) \
        if(node->data == "==") { \
          if(EVALEQ(node)) { \
            return 1; \
          } else { \
            return 0; \
          } \
        } else if(node->data == "<") { \
          if(EVALLOWER(node)) { \
            return 1; \
          } else { \
            return 0; \
          } \
        } else if(node->data == ">") { \
          if(EVALGREATER(node)) { \
            return 1; \
          } else { \
            return 0; \
          } \
        } else if(node->data == "<=") { \
          if(EVALLOWEREQ(node)) { \
            return 1; \
          } else { \
            return 0; \
          } \
        } else if(node->data == ">=") { \
           if(EVALGREATEREQ(node)) { \
            return 1; \
          } else { \
            return 0; \
          } \
        }

#define WHILESTMT2(node) \
        if(node->data == "==") { \
          while(EVALEQ(node)) { \
            return 1; \
          } else { \
            return 0; \
          } \
        } else if(node->data == "<") { \
          while(EVALLOWER(node)) { \
            return 1; \
          } else { \
            return 0; \
          } \
        } else if(node->data == ">") { \
          while(EVALGREATER(node)) { \
            return 1; \
          } else { \
            return 0; \
          } \
        } else if(node->data == "<=") { \
          while(EVALLOWEREQ(node)) { \
            return 1; \
          } else { \
            return 0; \
          } \
        } else if(node->data == ">=") { \
           while(EVALGREATEREQ(node)) { \
            return 1; \
          } else { \
            return 0; \
          } \
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

#define CONDITION2(stmt,node) \
        if(stmt == IF) { \
          IFSTMT(node); \
        }
        //} else if(stmt == WHILE) { \
           //WHILESTMT(node); \
        }

#define CONDITION(root) \
	 if(node->data == "==") { \
          if(EVALEQ(node)) { \
            return 1; \
          } else { \
            return 0; \
          } \
        } else if(node->data == "<") { \
          if(EVALLOWER(node)) { \
            return 1; \
          } else { \
            return 0; \
          } \
        } else if(node->data == ">") { \
          if(EVALGREATER(node)) { \
            return 1; \
          } else { \
            return 0; \
          } \
        } else if(node->data == "<=") { \
          if(EVALLOWEREQ(node)) { \
            return 1; \
          } else { \
            return 0; \
          } \
        } else if(node->data == ">=") { \
           if(EVALGREATEREQ(node)) { \
            return 1; \
          } else { \
            return 0; \
          } \
        }
        

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
          strncpy(value_cvar_string, "", sizeof(char)*TAU_NAME_LENGTH);  \
          char *value_cvar_value_string = (char *)malloc(sizeof(char)*TAU_NAME_LENGTH); \
          strncpy(value_cvar_value_string, "", sizeof(char)*TAU_NAME_LENGTH);  \
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
          strncpy(value_cvar_string, "", sizeof(value_cvar_string));  \
          char *value_cvar_value_string = (char *)malloc(sizeof(char)*TAU_NAME_LENGTH); \
          strncpy(value_cvar_value_string, "", sizeof(value_cvar_value_string));  \
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

//void plugin_tuning_policies(int argc, void **args)
//static json_object *jso = NULL;

int toInt(string s)
{
    int num = 0;
    for (int i=0; i<s.length();  i++)
        num = num*10 + (int(s[i])-48);
    return num;
}

int printInOrder(node_t *node)
{
  if(node->type == LEAF)
    return -1;

  printInOrder(node->loperand);
   
  printInOrder(node->roperand);

  return 1;
}

/* Recursive evaluation of a tree based expression */
int evalRes(node_t *root)
{
  // empty tree
  if (!root)
    return 0;

  // leaf node, value
  if(!(root->loperand) && !(root->roperand)) {
    return toInt(root->data);    
  }

  int l_val = evalRes(root->loperand); // Evaluate left operand
  int r_val = evalRes(root->roperand); // Evaluate right operand

  if(root->data == "+") {
    return l_val + r_val;
  } else if(root->data == "-") {
    return l_val - r_val; 
  } else if(root->data == "*") {
    return l_val * r_val;
  } else if(root->data == "/") {
    return l_val / r_val;
  }
  
  return 1;
}

int pop_tree(node_t *node)
{

 if (node->type == NODE) { 
   pop_tree(node->loperand);
   pop_tree(node->roperand);
   
 } else if(node->type == LEAF) {

 }

  return 1;
}

/* Evaluate condition expression */
int evalCond(node_t *node) 
{
  int loperand = toInt(node->loperand->data);
  int roperand = toInt(node->roperand->data);

  if(node->data == "==") { 
    if(loperand == roperand)
      return 1;
    else 
      return 0;
  } else if(node->data == "<") { 
    if(loperand < roperand)
      return 1;
    else 
      return 0;
  } else if(node->data == ">") { 
    if(loperand > roperand)
      return 1;
    else 
      return 0;
  } else if(node->data == "<=") { 
    if(loperand <= roperand)
      return 1;
    else 
      return 0;
  } else if(node->data == ">=") { 
    if(loperand >= roperand)
      return 1;
    else 
      return 0;
 }

 return -1;
        
}

/* Recursive evaluation of a tree based expression */
int evalExpr(node_t *root)
{
  int resVal = 0;

  // empty tree
  if (!root)
    return 0;

  // leaf node, value
  if(!(root->loperand) && !(root->roperand)) {
    return toInt(root->data);    
  }

  int l_val = evalExpr(root->loperand); // Evaluate left operand
  int r_val = evalExpr(root->roperand); // Evaluate right operand

  if(root->data == "+") {
    return l_val + r_val;
  } else if(root->data == "-") {
    return l_val - r_val; 
  } else if(root->data == "*") {
    return l_val * r_val;
  } else if(root->data == "/") {
    return l_val / r_val;
  } else {
    resVal = evalCond(root);
    return resVal;
  }
  
  return -1;
}


/*
int pop_tree(node_t *node)
{

 if (node->type == NODE) { 
   pop_tree(node->loperand);
   pop_tree(node->roperand);
   
 } else if(node->type == LEAF) {

 }

  return 1;
}
*/

int eval_condition(stmt_enum_t *stmt, node_t *root)
{
  
 return 1;
}

int result(node_t *root)
{
  if((root->loperand->type == LEAF) && (root->roperand->type == LEAF)) { 
   
    /* Evaluate result */
   
  
  } else {

  }

  return 1;
}

//void innerop(struct op_s *op)
void innerop(Op op)
{
        int resVal = 0;
        condition_t *cond = op.cond; 
        resVal = evalExpr(cond->root); 
        if(cond->stmt == "if") {
          IFSTMT(resVal) {
            node_t *res = op.result;
            evalExpr(res);
          }
        } else if(cond->stmt == "while") {
          WHILESTMT(resVal) {
            node_t *res = op.result;
            evalExpr(res);
          }
        }
        
        if(op.elseresult != NULL) { 
          node_t *elseres = op.elseresult; 
          evalExpr(elseres); 
 	} 
}

//void outerop(struct op_s *op)
void outerop(Op op)
{
  char metric_string[TAU_NAME_LENGTH], value_string[TAU_NAME_LENGTH]; 
  int *tau_pvar_count = NULL;
  int i=0, j=0;

  if(op.is_pvar_array == 1) { 
    unsigned long long int *value_array = (unsigned long long int *)calloc(tau_pvar_count[op.array_size],sizeof(unsigned long long int)); 
    char *value_cvar_string = (char *)malloc(sizeof(char)*TAU_NAME_LENGTH); 
    strncpy(value_cvar_string, "", sizeof(char)*TAU_NAME_LENGTH);  
    char *value_cvar_value_string = (char *)malloc(sizeof(char)*TAU_NAME_LENGTH); 
    strncpy(value_cvar_value_string, "", sizeof(char)*TAU_NAME_LENGTH);  
    for(i=0; i<op.array_size; i++) {
      innerop(op); 
      for(j=0; j<tau_pvar_count[op.num_pvars]; j++) { 
        if(i == (tau_pvar_count[j])) { 
          snprintf(metric_string, sizeof(metric_string),  "%s[%d]", op.result, i); 
          snprintf(value_string, sizeof(value_string),  "%llu", value_array[i]); 
        } 
      } 
      strcat(value_cvar_string,metric_string); 
      strcat(value_cvar_value_string,value_string); 
   } 
  } else { 
    unsigned long long int value; 
    char *value_cvar_string = (char *)malloc(sizeof(char)*TAU_NAME_LENGTH); 
    strncpy(value_cvar_string, "", sizeof(char)*TAU_NAME_LENGTH);  
    char *value_cvar_value_string = (char *)malloc(sizeof(char)*TAU_NAME_LENGTH); 
    strncpy(value_cvar_value_string, "", sizeof(char)*TAU_NAME_LENGTH);  
    innerop(op); 
    for(j=0; j<tau_pvar_count[op.num_pvars]; j++) { 
      if(i == (tau_pvar_count[j]))  { 
        snprintf(metric_string, sizeof(metric_string), "%s", op.result); 
        snprintf(value_string, sizeof(value_string),  "%llu", value); 
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

static JSONCPP_STRING readInputFile(const char* path) {
  FILE* file = fopen(path, "rb");
  if (!file)
    return JSONCPP_STRING("");
  fseek(file, 0, SEEK_END);
  long const size = ftell(file);
  unsigned long const usize = static_cast<unsigned long>(size);
  fseek(file, 0, SEEK_SET);
  JSONCPP_STRING text;
  char* buffer = new char[size + 1];
  buffer[size] = 0;
  if (fread(buffer, 1, usize, file) == usize)
    text = buffer;
  fclose(file);
  delete[] buffer;
  return text;
}

static JSONCPP_STRING removeSuffix(const JSONCPP_STRING& path,
                                const JSONCPP_STRING& extension) {
  if (extension.length() >= path.length())
    return JSONCPP_STRING("");
  JSONCPP_STRING suffix = path.substr(path.length() - extension.length());
  if (suffix != extension)
    return JSONCPP_STRING("");
  return path.substr(0, path.length() - extension.length());
}

/* Populate rule structure based on input json tree */
void store_json_tree(Json::Value& value, const JSONCPP_STRING& path)
{

  if(path == ".rule") {
    rule_idx++;
  } 
   
  if(path == ".rule.num_pvars") {
    rules[rule_idx].num_pvars = value.asLargestInt();
  }

  if(path == ".rule.operation") {
    Op op;
    rules[rule_idx].op = op; 
    //op_t *op = (struct op_s)malloc(sizeof(struct op_s));
    //rules[rule_idx].op = op; 
  }

  if(path == "rule.operation.condition") {
    condition_t *cond = (struct condition_s *)malloc(sizeof(struct condition_s));
    rules[rule_idx].op.cond = cond;
  }
 
  if(path == "rule.operation.condition.leftoperand") {
    node_t *root = (struct node_s *)malloc(sizeof(struct node_s));
    node_t *loperand = (struct node_s *)malloc(sizeof(struct node_s));
   
    if(value.type() == Json::stringValue) {
      loperand->data = value.asString().c_str();
      loperand->type = LEAF;
      root->loperand = loperand;    
      rules[rule_idx].op.cond->root = root;
      
    } else if (value.type() == Json::objectValue) {
      loperand->data = "";
      loperand->type = NODE;
      root->loperand = loperand; 
      rules[rule_idx].op.cond->root = root;
    }
  }

  if(path == "rule.operation.condition.rightoperand") {
    node_t *root = (struct node_s *)malloc(sizeof(struct node_s));
    node_t *roperand = (struct node_s *)malloc(sizeof(struct node_s));
 
    if(value.type() == Json::stringValue) {
      roperand->data = value.asString().c_str();
      roperand->type = LEAF;
      root->roperand = roperand;    
      rules[rule_idx].op.cond->root = root;
      
    } else if (value.type() == Json::objectValue) {
      roperand->data = "";
      roperand->type = NODE;
      root->roperand = roperand; 
      rules[rule_idx].op.cond->root = root;
    }
  }
 
  if(path == "rule.operation.condition.operator") {
    node_t *root = (struct node_s *)malloc(sizeof(struct node_s));
    operator_enum_t ope;  
    root->data = value.asString().c_str();
    //root->ope = ope;
    rules[rule_idx].op.cond->root = root;
  }

  if(path == "rule.operation.condition.stmt") {
    //stmt_enum_t stmt;
    //rules[rule_idx].op.cond->stmt = stmt;
    rules[rule_idx].op.cond->stmt = value.asString().c_str();
  }

  if(path == "rule.operation.result") {
    node_t *res = (struct node_s *)malloc(sizeof(struct node_s));
    rules[rule_idx].op.result = res; 
  }

  if(path == "rule.operation.else") {
    node_t *elseresult;
    rules[rule_idx].op.elseresult = elseresult;
  }

  if(path == "rule.operation.result.leftoperand") {
    node_t *loperand = (struct node_s *)malloc(sizeof(struct node_s));

    if(value.type() == Json::stringValue) {
      loperand->data = value.asString().c_str();
      loperand->type = LEAF;
      rules[rule_idx].op.result->loperand = loperand;
      
    } else if (value.type() == Json::objectValue) {
      loperand->data = "";
      loperand->type = NODE;
      rules[rule_idx].op.result->loperand = loperand;
    }

  }

  if(path == "rule.operation.result.rightoperand") {
    node_t *roperand = (struct node_s *)malloc(sizeof(struct node_s));
 
    if(value.type() == Json::stringValue) {
      roperand->data = value.asString().c_str();
      roperand->type = LEAF;
      rules[rule_idx].op.result->roperand = roperand;
      
    } else if (value.type() == Json::objectValue) {
      roperand->data = "";
      roperand->type = NODE;
      rules[rule_idx].op.result->roperand = roperand;
    }

    rules[rule_idx].op.result->roperand = roperand;  
  }

  if(path == "rule.operation.result.operator") {
    operator_enum_t ope;
    //root->data = value.asString().c_str();
    //rules[rule_idx].op.result->ope = ope;
    rules[rule_idx].op.result->data = value.asString().c_str();
  }

  if(path == "rule.operation.else.leftoperand") {
    node_t *loperand = (struct node_s *)malloc(sizeof(struct node_s));

    if(value.type() == Json::stringValue) {
      loperand->data = value.asString().c_str();
      loperand->type = LEAF;
      rules[rule_idx].op.elseresult->loperand = loperand;
      
    } else if (value.type() == Json::objectValue) {
      loperand->data = "";
      loperand->type = NODE;
      rules[rule_idx].op.elseresult->loperand = loperand;
    }

  }

  if(path == "rule.operation.else.rightoperand") {
    node_t *roperand = (struct node_s *)malloc(sizeof(struct node_s));
 
     if(value.type() == Json::stringValue) {
      roperand->data = value.asString().c_str();
      roperand->type = LEAF;
      rules[rule_idx].op.elseresult->roperand = roperand;
      
    } else if (value.type() == Json::objectValue) {
      roperand->data = "";
      roperand->type = NODE;
      rules[rule_idx].op.elseresult->roperand = roperand;
    }

  }

  if(path == "rule.operation.else.operator") {
    operator_enum_t ope;
    //rules[rule_idx].op.elseresult->ope = ope;
    rules[rule_idx].op.elseresult->data = value.asString().c_str();
  }

  switch(value.type()) {
  
    case Json::objectValue: {
     
     Json::Value::Members members(value.getMemberNames());

     std::sort(members.begin(), members.end());
     JSONCPP_STRING suffix = *(path.end() - 1) == '.' ? "" : ".";
     for (Json::Value::Members::iterator it = members.begin();
         it != members.end();
         ++it) {
       const JSONCPP_STRING name = *it;
       store_json_tree(value[name], path + suffix + name);
     }
 
    } 
    break;
    case Json::nullValue:
    case Json::intValue:
    case Json::uintValue:
    case Json::realValue:
    case Json::stringValue:
    case Json::booleanValue:
    case Json::arrayValue:
    break;

  }
}

/* Parse JSON tree */
void parse_json_tree(Json::Value& value, JSONCPP_STRING path = ".")
{

  switch(value.type()) {
    case Json::nullValue:
    case Json::intValue:
    case Json::uintValue:
    case Json::realValue:
    case Json::stringValue:
    case Json::booleanValue:
    case Json::arrayValue:
    case Json::objectValue: {
      Json::Value::Members members(value.getMemberNames());
      std::sort(members.begin(),members.end());
      JSONCPP_STRING suffix = *(path.end() - 1) == '.' ? "": ".";
      for(Json::Value::Members::iterator it = members.begin(); it != members.end(); ++it) {
        const JSONCPP_STRING name = *it;
        parse_json_tree(value[name], path + suffix + name);
      }
    } break;

    default:
    break;  
  }

}

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

void read_json_rules(const JSONCPP_STRING& path)
{
  Json::Value root;
  //JSONCPP_STRING path = "";
  FILE* fpolicy = fopen(path.c_str(), "r");
  
  store_json_tree(root, path);
  fclose(fpolicy);

}

/*
 * Load JSON file and store into proper structures
 */
int tuningpolicies_load_rules()
{
  Json::Value root;
  //JSONCPP_STRING path = "";
  FILE* fpolicy = fopen(path.c_str(), "r");
  
  store_json_tree(root, path);
  fclose(fpolicy);

  JSONCPP_STRING path = "policy.json";

  JSONCPP_STRING input = readInputFile(path.c_str());
  if (input.empty()) {
    printf("Failed to read input or empty input: %s\n", path.c_str());
    return 3;
  }
 
  JSONCPP_STRING basePath = removeSuffix(path, ".json"); 
  Json::Features features = Json::Features::strictMode();
  Json::Value root;
 
  Json::Reader reader(features);
  bool parsingSuccessful = reader.parse(input.data(), input.data() + input.size(), root);
  //bool parsingSuccessful = reader.parse(NULL, NULL, NULL);
  if (!parsingSuccessful) {
    printf("Failed to parse policy file: \n%s\n",
           reader.getFormattedErrorMessages().c_str());
    return 1;
  }

  store_json_tree(root, path);
  //read_json_rules(basePath);
 
}

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
  //op_t *op = rules[rule_id].op;
  Op op = rules[rule_id].op;
  //logic_t logic = rules[rule_id].logic;

 // Call logic 
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
      strncpy(reduced_value_cvar_string,  "", sizeof(char)*TAU_NAME_LENGTH); 
      reduced_value_cvar_value_string = (char *)malloc(sizeof(char)*TAU_NAME_LENGTH);
      strncpy(reduced_value_cvar_value_string,  "", sizeof(char)*TAU_NAME_LENGTH); 
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
      snprintf(metric_string, sizeof(metric_string), "%s[%d]", CVAR_SPECIFYING_REDUCED_POOL_SIZE, i);
      snprintf(value_string, sizeof(value_string), "%llu", reduced_value_array[i]);
    } else {
      snprintf(metric_string, sizeof(metric_string), "%s[%d],", CVAR_SPECIFYING_REDUCED_POOL_SIZE, i);
      snprintf(value_string, sizeof(value_string), "%llu,", reduced_value_array[i]);
    }
    
    strcat(reduced_value_cvar_string, metric_string);
    strcat(reduced_value_cvar_value_string, value_string);

  }

  if(has_threshold_been_breached_in_any_pool) {
    snprintf(metric_string, sizeof(metric_string), "%s,%s", CVAR_ENABLING_POOL_CONTROL, reduced_value_cvar_string);
    snprintf(value_string, sizeof(value_string), "%d,%s", 1, reduced_value_cvar_value_string);
    //dprintf("Metric string is %s and value string is %s\n", metric_string, value_string);
    Tau_mpi_t_parse_and_write_cvars(metric_string, value_string);
  } else {
    snprintf(metric_string, sizeof(metric_string), "%s", CVAR_ENABLING_POOL_CONTROL);
    snprintf(value_string, sizeof(value_string), "%d", 0);
    //dprintf("Metric string is %s and value string is %s\n", metric_string, value_string);
    Tau_mpi_t_parse_and_write_cvars(metric_string, value_string);
  }
 
  return return_val;
}
