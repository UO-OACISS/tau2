#include <Profile/CudaDisassembly.h>

using namespace std;

// #define TAU_DEBUG_DISASM


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

void parse_disassem(std::vector<std::string> vec, int device_id) 
{
  int n = vec.size()-1;
  int i = 0;
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
      std::string line_number = "null";
      std::string instrs = "null";
      std::vector<std::string> v_out = split(vec[i], ' ');
      string underscore = "_";
      for(int j = 0; j < v_out.size(); j++) {
	if (v_out[j].substr(0, underscore.size()) == underscore) {
	  kernel_name = v_out[j].substr(0, v_out[j].length() - 2);
	}
      }
      while (vec[i].substr(0, prefix.size()) != prefix && i < n) {
	if(vec[i].find("//## File") != std::string::npos) {
	  v_out = split(vec[i], ' ');
	  for (int i = 0; i < v_out.size(); i++) {
	    if(v_out[i].find("\"") != std::string::npos) {
	      file_path = v_out[i].substr(1, v_out[i].length() - 3);
	      line_number = v_out[i+2].substr(0, v_out[i+2].length() - 2);
	    }
	  }
	}
	else if (vec[i].find("*/") != std::string::npos) {
	  string pc_offset = "null";
	  v_out = split(vec[i], '*');
	  if (v_out[1].length() == 4) {
	    // make sure it's not a register
	    pc_offset = v_out[1];
	    instrs = sanitize_instruction(v_out[2]);
#ifdef TAU_DEBUG_DISASM
	    cout << "About to insert cudaOpsMap\n";
	    cout << "Kernel name: " << kernel_name;
	    cout << "File path " << file_path << ", Line number " << line_number << endl;
	    cout << "PC Offset " << pc_offset  << ", Instrs " << instrs << endl;
#endif
	    CudaOps *cuOps = new CudaOps(kernel_name, file_path, line_number, instrs, 
					 pc_offset, device_id);
	    v_cudaOps.push_back(cuOps);
	  }
	}
	i++;
      }
    }
    else {
      i++;
    }
  }
}

void print_hello_world(char* cubin_file, int device_id)
{
  init_instruction_set();
  string command1 = "nvdisasm -g -c -hex -sf " + (string)cubin_file;
  std::vector<std::string> res1 = get_disassem_from_out(command1);
  parse_disassem(res1, device_id);
  //print_disassem();

  // test
  int six = get_instruction_mix_category("CCTLT"); //6
  int four = get_instruction_mix_category("PSETP");
  int ten = get_instruction_mix_category("NOP"); //10
  cout << "six:" << six << ", four:" << four << ", ten:" << ten << endl;

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

void print_disassem()
{
  for(int i = 0; i < v_cudaOps.size(); i++) {
    std::cout << v_cudaOps[i]->deviceid << ":" << v_cudaOps[i]->kernel << ":"
	      << v_cudaOps[i]->filename << ":" << v_cudaOps[i]->lineno << ":"
	      << v_cudaOps[i]->instruction << ":" << v_cudaOps[i]->pcoffset << std::endl;
  }
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



