#include <stdio.h>
#include <stdint.h>
#include <iostream>
#include <string>
#include <map>
#include <list>
#include <vector>
#include <sstream>
#include <set>

using namespace std;

/* BEGIN: Disassem Structs */
// Look up ops
enum InstructionMix { FloatingPoint = 0, Integer = 1, Conversion = 2, Move = 3, Predicate = 4, 
		      Texture = 5, LoadStore = 6, Surface = 7, Control = 8, SIMD = 9, Misc = 10};

struct ImixStats {
public:
  std::string kernel;
  int flops_raw;
  int ctrlops_raw;
  int memops_raw;
  int totops_raw;
  float flops_pct;
  float ctrlops_pct;
  float memops_pct;
};

struct CudaOps 
{
 public:
  std::string kernel;
  std::string filename;
  int lineno;
  std::string instruction;
  int pcoffset;
};

static std::set<std::string> s_FP;
static std::set<std::string> s_Int;
static std::set<std::string> s_Conv;
static std::set<std::string> s_Move;
static std::set<std::string> s_Pred;
static std::set<std::string> s_Tex;
static std::set<std::string> s_LdSt;
static std::set<std::string> s_Surf;
static std::set<std::string> s_Ctrl;
static std::set<std::string> s_SIMD;
static std::set<std::string> s_Misc;
static std::vector<CudaOps> v_cudaOps;
static bool init_instruction = false;
/* END:  Disassem Structs */

/* BEGIN: SASS Structs */
class InstrSampling
{
 public:
  uint32_t sourceLocatorId;
  uint32_t functionId;
  uint32_t pcOffset;
  uint32_t correlationId;
  uint32_t executed;
  uint32_t threadsExecuted;
  double timestamp_delta;
  double timestamp_current;
};

class FuncSampling
{
 public:
  uint32_t fid;
  uint32_t contextId;
  uint32_t moduleId;
  uint32_t functionIndex;
  const char* name;
  const char* demangled;
  double timestamp;
  uint32_t deviceId;
};

class SourceSampling
{
 public:
  uint32_t sid;
  const char* fileName;
  uint32_t lineNumber;
  double timestamp;
};
/* END: SASS Structs */

/* // routines for calculating kernel level stats */
/* void printInstrMap(std::map<uint32_t, std::list<InstrSampling> > instructionMap); */
void printSourceMap(std::map<uint32_t, SourceSampling> srcLocMap);
void printFuncMap(std::map<uint32_t, FuncSampling> funcMap);
double getKernelExecutionTimes(uint32_t functionIndex, std::map<uint32_t, std::list<InstrSampling> > instructionMap);
void resetKernelExecutionTimes(uint32_t functionIndex, std::map<uint32_t, std::list<InstrSampling> > instructionMap);
uint32_t getFunctionId(const char* kernel, std::map<uint32_t, FuncSampling> funcMap);
uint32_t getKernelSamples(uint32_t functionIndex, std::map<uint32_t, std::list<InstrSampling> > instructionMap);
uint32_t getUniqueKernelLaunches(uint32_t functionIndex, std::map<uint32_t, std::list<InstrSampling> > instructionMap);
const char* getKernelFilePath(uint32_t functionIndex, std::map<uint32_t, std::list<InstrSampling> > instructionMap, std::map<uint32_t, SourceSampling> srcLocMap);
uint32_t getKernelLineNo(uint32_t functionIndex, std::map<uint32_t, std::list<InstrSampling> > instructionMap, std::map<uint32_t, SourceSampling> srcLocMap);
ImixStats write_runtime_imix(uint32_t functionId, std::list<InstrSampling> instrFunc_list, std::map<std::pair<int, int>, CudaOps> map_disassem, std::map<uint32_t, SourceSampling> srcLocMap, std::string kernel);
std::vector<std::string> get_disassem_from_out(std::string cmd);
std::map<std::pair<int, int>, CudaOps> parse_cubin(char* cubin_file);
std::map<std::pair<int, int>, CudaOps> parse_disassem(std::vector<std::string> vec);
std::map<std::string, ImixStats> print_instruction_mixes();
std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems);
std::vector<std::string> split(const std::string &s, char delim);
void print_vector(std::vector<std::string> vec);
std::map<std::string, ImixStats> write_disassem();
void init_instruction_set();
int get_instruction_mix_category(string instr);
void insert_instructions(std::set<std::string> *s_set, std::string Ins[], int size);
std::string sanitize_instruction(std::string instr);


