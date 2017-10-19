#include <Profile/CudaSass.h>
#include <iostream>
#include <set>
#include <string.h>
#include <iomanip>
#include <fcntl.h>
#include <cstdio>
#include <cstdlib>

using namespace std;

//#define TAU_DEBUG_SASS 1

/* BEGIN: Disassem helpers */
std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    elems.push_back(item);
  }
  return elems;
}

std::vector<std::string> split(const std::string &s, char delim) {
  std::vector<std::string> elems;
  split(s, delim, elems);
  return elems;
}

void print_vector(std::vector<std::string> vec)
{
  for(std::vector<std::string>::const_iterator iter = vec.begin();
      iter != vec.end(); iter++) {
    std::cout << *iter << endl;
  }
}

/**
 * Execute a command and get the result.
 *
 * @param   cmd - The system command to run.
 * @return  The string command line output of the command.
 */

std::vector<std::string> get_disassem_from_out(std::string cmd) 
{
  // string data;
  FILE * stream;
  const int max_buffer = 256;
  char buffer[max_buffer];
  std::vector<std::string> v_out;

  cmd.append(" 2>&1"); // Do we want STDERR?

  stream = popen(cmd.c_str(), "r");
  if (stream) {
    int i = 0;
    while (!feof(stream)) {
      if (fgets(buffer, max_buffer, stream) != NULL) {
	v_out.push_back(buffer);
      }
    }
    pclose(stream);
  }
  
  return v_out;
}

std::map<std::pair<int, int>, CudaOps> parse_disassem(std::vector<std::string> vec, int device_id) 
{
  int n = vec.size()-1;
  int i = 0;
  std::map<std::pair<int, int>, CudaOps> map_disassem;

  while (i < n) {
    string prefix = "//---";
    if (vec[i].substr(0, prefix.size()) ==  prefix) {
#ifdef TAU_DEBUG_DISASM
      cout << "//-- prefix found\n";
#endif
      std::size_t pos = vec[i].find("SYMBOLS");
      if(pos > vec[i].size()) {
#ifdef TAU_DEBUG_DISASM
        cout << "SYMBOLS not found\n";
#endif
      }
      else {
#ifdef TAU_DEBUG_DISASM
        cout << "SYMBOLS found, break\n";
#endif
        break;
      }
      string global = ".global";
      std::size_t g_pos = vec[i].find(global);
      while(g_pos > vec[i].size()) {
	i++;
	g_pos = vec[i].find(global);
      }
      // at this point g_pos is at .global
      std::string kernel_name = "null";
      std::string file_path = "null";
      int line_number = -1;
      std::string instrs = "null";
      std::vector<std::string> v_out = split(vec[i], ' ');
      string underscore = "_";
      for(int j = 0; j < v_out.size(); j++) {
	if (v_out[j].substr(0, underscore.size()) == underscore) {
	  kernel_name = split(v_out[j], ' ')[0];
	  //kernel_name = v_out[j].substr(0, v_out[j].length() - 2);
	}
      }
      while (vec[i].substr(0, prefix.size()) != prefix && i < n) {
	if(vec[i].find("//## File") != std::string::npos) {
	  v_out = split(vec[i], ' ');
	  for (int i = 0; i < v_out.size(); i++) {
	    if(v_out[i].find("\"") != std::string::npos) {
	      file_path = v_out[i].substr(1, v_out[i].length() - 3);
	      std::string line_number2 = split(v_out[i+2], ' ')[0];
	      line_number = atoi(line_number2.c_str());
	    }
	  }
	}
	else if (vec[i].find("*/") != std::string::npos) {
	  int pc_offset = -1;
	  v_out = split(vec[i], '*');
	  if (v_out[1].length() == 4) {
	    // make sure it's not a register
	    pc_offset = strtoul(v_out[1].c_str(), NULL, 16);
	    instrs = sanitize_instruction(v_out[2]);
#ifdef TAU_DEBUG_DISASM
	    cout << "About to insert cudaOpsMap\n";
	    cout << "Kernel name: " << kernel_name;
	    cout << "File path " << file_path << ", Line number " << line_number << endl;
	    cout << "PC Offset " << pc_offset  << ", Instrs " << instrs << endl;
#endif
	    CudaOps cuOps;
	    cuOps.kernel = kernel_name;
	    cuOps.filename = file_path;
	    cuOps.lineno = line_number;
	    cuOps.pcoffset = pc_offset;
	    cuOps.instruction = instrs;
	    cuOps.deviceid = device_id;
	    v_cudaOps.push_back(cuOps);

	    typedef std::pair<int, int> my_key_type;
	    typedef std::map<my_key_type, CudaOps> my_map_type;
	    map_disassem.insert(my_map_type::value_type(my_key_type(line_number, pc_offset), cuOps));
	  }
	}
	i++;
      }
    }
    else {
      i++;
    }
  }
  return map_disassem;
}

std::map<std::pair<int, int>, CudaOps> parse_cubin(char* cubin_file, int device_id)
{
  init_instruction_set();
  std::map<std::pair<int, int>, CudaOps> map_disassem;
  string command1 = "nvdisasm -g -c -hex -sf " + (string)cubin_file;
  std::vector<std::string> res1 = get_disassem_from_out(command1);
  map_disassem = parse_disassem(res1, device_id);
  return map_disassem;
}


std::map<std::string, ImixStats> print_instruction_mixes()
{
  std::map<std::string, ImixStats> map_imixStats;
  map_imixStats = write_disassem();
  return map_imixStats;
}

void init_instruction_set() 
{
  if (!init_instruction) {
#ifdef TAU_DEBUG_DISASM
    cout << "Initializing instruction set\n";
#endif
    std::string FPIns[] = {"FADD", "FCHK", "FCMP", "FFMA", "FMAD", "FMAX", "FMIN", "FMNMX",
			   "FMUL", "FSET", "FSETP", "FSWZ", "FSWZADD", "MUFU", "RRO", "DADD",
			   "DFMA", "DMAX", "DMIN", "DMNMX", "DMUL", "DSET", "DSETP", "LG2", "RCP"};
    std::string IntIns[] = {"BFE","BFI","FLO","IADD","IADD3","ICMP","IMAD","IMADSP","IMNMX",
			    "IMAX","IMIN","IMUL","ISAD","ISCADD","ISET","ISETP","LEA","LOP",
			    "LOP3","POPC","SHF","SHL","SHR","XMAD","COS","EX2","RSQ","SIN","ISUB"};
    std::string ConvIns[] = {"F2F","F2I","I2F","I2I"};
    std::string MoveIns[] = {"MOV","PRMT","SEL","SHFL","ADA","A2R","G2R","MVC","MVI","R2A",
			     "R2C","R2G","SQR"};
    std::string PredIns[] = {"CSET","CSETP","PSET","PSETP","P2R","R2P"};
    std::string TexIns[] = {"TEX","TLD","TLD4","TXQ","TEXS","TLD4S","TLDS"};
    std::string LdStIns[] = {"LD","LDC","LDG","LDL","LDLK","LDS","LDSLK","LD_LDU","LDS_LDU",
			     "LDU","ST","STG","STL","STS","STSCUL","STSUL","STUL","ATOM","ATOMS",
			     "RED","CCTL","CCTLL","MEMBAR","CCTLT","GATOM","GLD","GRED","GST",
			     "LLD","LST"};
    std::string SurfIns[] = {"SUATOM","SUBFM","SUCLAMP","SUEAU","SULEA","SULD","SULDGA",
			     "SUQ","SURED","SUST","SUSTGA"};
    
    std::string CtrlIns[] = {"BRA","BRX","JMP","JMX","SSY","SYNC","CAL","JCAL","PRET","RET",
			     "BRK","PBK","CONT","LONGJMP","PLONGJMP","PCNT","EXIT","PEXIT","BPT",
			     "C2R"};
    std::string SIMDIns[] = {"VADD","VADD2","VADD4","VSUB","VSUB2","VSUB4","VMAD","VAVRG2","VAVRG4",
			     "VABSDIFF","VABSDIFF2",
			     "VABSDIFF4","VMIN","VMIN2","VMIN4","VMAX","VMAX2","VMAX4","VSHL",
			     "VSHR","VSET","VSET2","VSET4"};
    std::string MiscIns[] = {"NOP","CS2R","S2R","B2R","BAR","R2B","LEPC","VOTE"};

    insert_instructions(&s_FP, FPIns, 25);
    insert_instructions(&s_Int, IntIns, 29);
    insert_instructions(&s_Conv, ConvIns, 4);
    insert_instructions(&s_Move, MoveIns, 13);
    insert_instructions(&s_Pred, PredIns, 6);
    insert_instructions(&s_Tex, TexIns, 7);
    insert_instructions(&s_LdSt, LdStIns, 30);
    insert_instructions(&s_Surf, SurfIns, 11);
    insert_instructions(&s_Ctrl, CtrlIns, 20);
    insert_instructions(&s_SIMD, SIMDIns, 23);
    insert_instructions(&s_Misc, MiscIns, 8);

    init_instruction = true;
  }
}

void insert_instructions(std::set<std::string> *s_set, std::string Ins[], int size)
{

  for (int i = 0; i < size; i ++) {
    s_set->insert(Ins[i]);
  }
#ifdef TAU_DEBUG_DISASM
  cout << "Inside insert_instructions\n";
  for(set<string>::iterator it = s_set->begin(); it != s_set->end(); it++) {
    cout << "s_set[i]: " << *it << endl;
  }
#endif

}

int get_instruction_mix_category(string instr)
{
#ifdef TAU_DEBUG_DISASM
  cout << "Inside get_instruction_mix_category\n";
#endif
  if(s_FP.count(instr))
    return FloatingPoint;
  else if(s_Int.count(instr))
    return Integer;
  else if(s_Conv.count(instr))
    return Conversion;
  else if(s_Move.count(instr))
    return Move;
  else if(s_Pred.count(instr))
    return Predicate;
  else if(s_Tex.count(instr))
    return Texture;
  else if(s_LdSt.count(instr))
    return LoadStore;
  else if(s_Surf.count(instr))
    return Surface;
  else if(s_Ctrl.count(instr))
    return Control;
  else if(s_SIMD.count(instr))
    return SIMD;
  else if(s_Misc.count(instr))
    return Misc;
  return -1;
}

std::map<std::string, ImixStats> write_disassem()
{
  // break down by kernel, then iterate, do count of each instruction type, write to file
  std::map<std::string, ImixStats> map_imixStats;
  string current_kernel = "";
  string kernel_iter = "";
  int flops_raw = 0;
  int ctrlops_raw = 0;
  int memops_raw = 0;
  int totops_raw = 0;
  double flops_pct = 0;
  double ctrlops_pct = 0;
  double memops_pct = 0;
  for(int i = 0; i < v_cudaOps.size(); i++) {

    kernel_iter = v_cudaOps[i].kernel.erase(v_cudaOps[i].kernel.size() - 1) ; // strip '\n'

    // cout << "[CudaDisassembly]:  demangleName: " << kernel_iter << endl;
    if (current_kernel == "") {
      current_kernel = kernel_iter;
    }
    else if (current_kernel != kernel_iter) {
      flops_pct = ((float)flops_raw/totops_raw) * 100;
      memops_pct = ((float)memops_raw/totops_raw) * 100;
      ctrlops_pct = ((float)ctrlops_raw/totops_raw) * 100;
      ImixStats imix_stats;
      // push onto map
      imix_stats.flops_raw = flops_raw;
      imix_stats.ctrlops_raw = ctrlops_raw;
      imix_stats.memops_raw = memops_raw;
      imix_stats.totops_raw = totops_raw;
      imix_stats.flops_pct = flops_pct;
      imix_stats.ctrlops_pct = ctrlops_pct;
      imix_stats.memops_pct = memops_pct;
      imix_stats.kernel = current_kernel;
      map_imixStats[current_kernel] = imix_stats;

#ifdef TAU_DEBUG_DISASM
      cout << "[CudaDisassembly]:  current_kernel: " << current_kernel << endl;
      cout << "  FLOPS: " << flops_raw << ", MEMOPS: " << memops_raw 
	   << ", CTRLOPS: " << ctrlops_raw << ", TOTOPS: " << totops_raw << "\n";
      cout << setprecision(2) << "  FLOPS_pct: " << flops_pct << "%, MEMOPS_pct: " 
	   << memops_pct << "%, CTRLOPS_pct: " << ctrlops_pct << "%\n";
#endif
      current_kernel = kernel_iter;
      flops_raw = 0;
      ctrlops_raw = 0;
      memops_raw = 0;
      totops_raw = 0;
      flops_pct = 0;
      ctrlops_pct = 0;
      memops_pct = 0;
    }
    else {
      // cout << "[CudaDisassembly]: do nothing\n";
    }

    // counting ops for new kernel sample (fix!)
    int instr_type = get_instruction_mix_category(v_cudaOps[i].instruction);
    switch(instr_type) {
      // Might be non-existing ops, don't count those!
      case FloatingPoint: case Integer:
      case SIMD: case Conversion: {
	flops_raw++;
	totops_raw++;
	break;
      }
      case LoadStore: case Texture:
      case Surface: {
	memops_raw++;
	totops_raw++;
	break;
      }
      case Control: case Move:
      case Predicate: {
	ctrlops_raw++;
	totops_raw++;
	break;
      }
      case Misc: {
        totops_raw++;
	break;
      }
    }

  }
  // do one last time for corner case
  flops_pct = ((float)flops_raw/totops_raw) * 100;
  memops_pct = ((float)memops_raw/totops_raw) * 100;
  ctrlops_pct = ((float)ctrlops_raw/totops_raw) * 100;
  ImixStats imix_stats;
  // push onto map
  imix_stats.flops_raw = flops_raw;
  imix_stats.ctrlops_raw = ctrlops_raw;
  imix_stats.memops_raw = memops_raw;
  imix_stats.totops_raw = totops_raw;
  imix_stats.flops_pct = flops_pct;
  imix_stats.ctrlops_pct = ctrlops_pct;
  imix_stats.memops_pct = memops_pct;
  imix_stats.kernel = current_kernel;
  map_imixStats[current_kernel] = imix_stats;
  
#ifdef TAU_DEBUG_DISASM
  cout << "[CudaDisassembly]:  current_kernel: " << current_kernel << endl;
  cout << "  FLOPS: " << flops_raw << ", MEMOPS: " << memops_raw 
       << ", CTRLOPS: " << ctrlops_raw << ", TOTOPS: " << totops_raw << "\n";
  cout << setprecision(2) << "  FLOPS_pct: " << flops_pct << "%, MEMOPS_pct: " 
       << memops_pct << "%, CTRLOPS_pct: " << ctrlops_pct << "%\n";
#endif
  flops_raw = 0;
  ctrlops_raw = 0;
  memops_raw = 0;
  totops_raw = 0;
  flops_pct = 0;
  ctrlops_pct = 0;
  memops_pct = 0;
  
  return map_imixStats;
}

std::string sanitize_instruction(std::string instr) 
{
  string op = "";
  
  int i = 0;
  int n = instr.length();
  bool done = false;
  while (!done) {
    if(instr.find("@") != std::string::npos) {
      std::size_t pos1 = instr.find("@");
      i = pos1;
      while(instr[i] != ' ') {
	i++;
      }
      // at space
      i++;
      while(isalpha(instr[i]) || isdigit(instr[i])) {
	op += instr[i];
	i++;
      }
      done = true;
      // break;
    }
    else if (instr.find("ISETP") != std::string::npos) {
      op += "ISETP";
      done = true;
      // break;
    }
    else if (instr[i] == ' ' || instr[i] == '/' || instr[i] == '{') {
      i++;
    }
    else {
      // at an op, read until space or .
      op += instr[i];
      i++;
      while (isalpha(instr[i]) || isdigit(instr[i])) {
	op += instr[i];
	i++;
      }
      done = true;
      // break;
    }
  } // while

  return op;
}
/* END: Disassem helpers */

/* BEGIN: SASS Helper Functions */
double getKernelExecutionTimes(uint32_t functionIndex, std::map<uint32_t, std::list<InstrSampling> > instructionMap)
{
  double kernel_exec_time;
  kernel_exec_time = 0;
  
  std::list<InstrSampling> instrFunc_list = instructionMap.find(functionIndex)->second;

  for (std::list<InstrSampling>::iterator it2 = instrFunc_list.begin();
       it2 != instrFunc_list.end(); ++it2) {
    kernel_exec_time += it2->timestamp_delta;
  }

  return kernel_exec_time;	
}

void resetKernelExecutionTimes(uint32_t functionIndex, std::map<uint32_t, std::list<InstrSampling> > instructionMap)
{
#ifdef TAU_DEBUG_SASS
  printf("[TauGpu]:  About to remove items on list\n");
#endif
  //InstrSampling is_temp = instrFuncMap.find(functionIndex)->second.back();
  instructionMap.find(functionIndex)->second.clear();
  //instrFuncMap[functionIndex].push_back(is_temp);

}

uint32_t getFunctionId(const char* kernel, std::map<uint32_t, FuncSampling> funcMap) 
{
  uint32_t retval = -1;
  for (std::map<uint32_t, FuncSampling>::iterator it=funcMap.begin(); it != funcMap.end(); ++it) {
    FuncSampling fTemp = it->second;
    // cout << "fTemp.Name: " << fTemp.name << ", kernel: " << kernel << endl;
    if (strcmp(fTemp.name, kernel) == 0) {
      // cout << "FOUND!!!\n"; 
      retval = fTemp.fid;
      break;
    }
  }  
  return retval;
}

uint32_t getKernelSamples(uint32_t functionIndex, std::map<uint32_t, std::list<InstrSampling> > instructionMap) 
{
  std::list<InstrSampling> instrFunc_list = instructionMap.find(functionIndex)->second;
  return instrFunc_list.size();	
}

uint32_t getUniqueKernelLaunches(uint32_t functionIndex, std::map<uint32_t, std::list<InstrSampling> > instructionMap)
{
  std::list<InstrSampling> instrFunc_list = instructionMap.find(functionIndex)->second;
  std::set<uint32_t> corrIdFromInstr_set;
  for (std::list<InstrSampling>::iterator it=instrFunc_list.begin(); 
       it != instrFunc_list.end(); ++it) {
    InstrSampling is = *it;
    corrIdFromInstr_set.insert(is.correlationId);
  }
  return corrIdFromInstr_set.size();
}


const char* getKernelFilePath(uint32_t functionIndex, std::map<uint32_t, std::list<InstrSampling> > instructionMap, std::map<uint32_t, SourceSampling> srcLocMap)
{
  // assuming first entry's filepath is same for all
  uint32_t srcId_temp = instructionMap.find(functionIndex)->second.front().sourceLocatorId;
  return srcLocMap.find(srcId_temp)->second.fileName;
}

uint32_t getKernelLineNo(uint32_t functionIndex, std::map<uint32_t, std::list<InstrSampling> > instructionMap, std::map<uint32_t, SourceSampling> srcLocMap)
{
  uint32_t lineNumber;
  lineNumber = 99999999;
  std::list<InstrSampling> instrFunc_list = instructionMap.find(functionIndex)->second;

  std::set<uint32_t> srcIdFromInstr_set;
  for (std::list<InstrSampling>::iterator it=instrFunc_list.begin(); 
       it != instrFunc_list.end(); ++it) {
    InstrSampling is = *it;
    srcIdFromInstr_set.insert(is.sourceLocatorId);
  }
  for (std::set<uint32_t>::iterator it2=srcIdFromInstr_set.begin();
       it2 != srcIdFromInstr_set.end(); it2++) { 
    uint32_t srcId = *it2;
    SourceSampling ss = srcLocMap.find(srcId)->second;
    if (lineNumber > ss.lineNumber) {
      lineNumber = ss.lineNumber;
    }
  }

  return lineNumber;	

}

// void printInstrMap(std::map<uint32_t, std::list<InstrSampling> > instructionMap)
// {
//   // we only care about each kernel, get lowest number line of source code reference
//   for(std::map<uint32_t, InstrSampling>::iterator iter = instructionMap.begin();
//       iter != instructionMap.end(); ++iter) {
//     InstrSampling instrTemp = iter->second;
    
//     printf("~~~ Instr Sampling MAP ~~~\n  tstamp_delta: %lu%, SourceLocatorID: %i\n  FunctionID: %i, pcOffset: %i\n",
// 	   instrTemp.timestamp_delta, instrTemp.sourceLocatorId, instrTemp.functionId, instrTemp.pcOffset);
//   }
// }

void printSourceMap(std::map<uint32_t, SourceSampling> srcLocMap)
{
  for (std::map<uint32_t, SourceSampling>::iterator it=srcLocMap.begin(); 
       it != srcLocMap.end(); ++it) {
    SourceSampling sTemp = it->second;
    printf("~~~ srcLocMap ~~~\n  id %u, fileName %s, lineNumber %u\n  timestamp %f\n~~~~~~\n",
	   sTemp.sid, sTemp.fileName, sTemp.lineNumber, sTemp.timestamp);
  }
}

void printFuncMap(std::map<uint32_t, FuncSampling> funcMap)
{
  // iterate map, print out
  for (std::map<uint32_t, FuncSampling>::iterator it=funcMap.begin(); it != funcMap.end(); ++it) {
    FuncSampling fTemp = it->second;
    printf("~~~ funcMap ~~~\n  id %u, ctx %u, moduleId %u\n  functionIndex %u, name %s, demangled %s\n~~~~~~\n",
	   fTemp.fid, fTemp.contextId, fTemp.moduleId,fTemp.functionIndex, fTemp.name,
	   fTemp.demangled);
  }
}

ImixStats write_runtime_imix(uint32_t functionId, std::list<InstrSampling> instrSamp_list, std::map<std::pair<int, int>, CudaOps> map_disassem, std::map<uint32_t, SourceSampling> srcLocMap, std::string kernel)
{

#ifdef TAU_DEBUG_SASS
  cout << "[CudaSass]: write_runtime_imix begin\n";
#endif

  // look up from map_imix_static
  ImixStats imix_stats;
  string current_kernel = "";
  int flops_raw = 0;
  int ctrlops_raw = 0;
  int memops_raw = 0;
  int totops_raw = 0;
  double flops_pct = 0;
  double ctrlops_pct = 0;
  double memops_pct = 0;

  // check if entries exist
  if (!instrSamp_list.empty()) {
    // cout << "[CuptiActivity]:  instrSamp_list not empty\n";
    for (std::list<InstrSampling>::iterator iter=instrSamp_list.begin();
	 iter != instrSamp_list.end(); iter++) {
      InstrSampling is = *iter;
      
      // TODO:  Get line info here...
      int sid = is.sourceLocatorId;
      // cout << "[CuptiActivity]:  is.sourceLocatorId: " << is.sourceLocatorId << endl;
      int lineno = -1;
      if ( srcLocMap.find(sid) != srcLocMap.end() ) {
	lineno = srcLocMap.find(sid)->second.lineNumber;
	// cout << "[CuptiActivity]:  lineno: " << lineno << endl;
	std::pair<int, int> p1 = std::make_pair(lineno, is.pcOffset);

	for (std::map<std::pair<int, int>,CudaOps>::iterator iter= map_disassem.begin();
	     iter != map_disassem.end(); iter++) { 
	  CudaOps cuops = iter->second;
	  // cout << "cuops pair(" << cuops.lineno << ", " << cuops.pcoffset << ")\n";
	  if (map_disassem.find(p1) != map_disassem.end()) {
	    CudaOps cuops = map_disassem.find(p1)->second;
	    // cout << "[CuptiActivity]:  cuops.instruction: " << cuops.instruction << endl;
	    // map to disassem
	    int instr_type = get_instruction_mix_category(cuops.instruction);
	    switch(instr_type) {
	      // Might be non-existing ops, don't count those!
	      case FloatingPoint: case Integer:
	      case SIMD: case Conversion: {
		flops_raw++;
		totops_raw++;
		break;
	      }
	      case LoadStore: case Texture:
	      case Surface: {
		memops_raw++;
		totops_raw++;
		break;
	      }
	      case Control: case Move:
	      case Predicate: {
		ctrlops_raw++;
		totops_raw++;
		break;
	      }
	      case Misc: {
		totops_raw++;
		break;
	      }
	    }
	  }
	  else {
#if TAU_DEBUG_DISASM
	    cout << "[CuptiActivity]:  map_disassem does not exist for pair(" 
	    	 << lineno << "," << is.pcOffset << ")\n";
#endif
	  }
	}
      }
      else {
#if TAU_DEBUG_DISASM
	cout << "[CuptiActivity]:  srcLocMap does not exist for sid: " << sid << endl;
#endif
      }
    }
  }
  else {
    cout << "[CuptiActivity]: instrSamp_list empty!\n";
  }
  
  string kernel_iter = kernel;

  flops_pct = ((float)flops_raw/totops_raw) * 100;
  memops_pct = ((float)memops_raw/totops_raw) * 100;
  ctrlops_pct = ((float)ctrlops_raw/totops_raw) * 100;
  // push onto map
  imix_stats.flops_raw = flops_raw;
  imix_stats.ctrlops_raw = ctrlops_raw;
  imix_stats.memops_raw = memops_raw;
  imix_stats.totops_raw = totops_raw;
  imix_stats.flops_pct = flops_pct;
  imix_stats.ctrlops_pct = ctrlops_pct;
  imix_stats.memops_pct = memops_pct;
  imix_stats.kernel = kernel_iter;

#ifdef TAU_DEBUG_DISASM
  cout << "[CudaDisassembly]:  current_kernel: " << kernel_iter << endl;
  cout << "  FLOPS: " << flops_raw << ", MEMOPS: " << memops_raw 
       << ", CTRLOPS: " << ctrlops_raw << ", TOTOPS: " << totops_raw << "\n";
  cout << setprecision(2) << "  FLOPS_pct: " << flops_pct << "%, MEMOPS_pct: " 
       << memops_pct << "%, CTRLOPS_pct: " << ctrlops_pct << "%\n";
#endif

  return imix_stats;
}
/* END: SASS Helper Functions */
