/****************************************************************************
 ** tau_reduce.cpp
 ** Version 1.0
 ** Nick Trebon
 ** 7/2002
 ****************************************************************************
 ** tau_reduce will read in a pprof created dump file and store the 
 ** information.  tau_reduce will then generate a exclude list
 ** that can be used to exclude certain functions when pprof is again 
 ** implemented.
 ***************************************************************************/
#include "pprof_elem.h"

#define SIZE_OF_FILENAME 1024   // Include full path. 
#ifdef TAU_HP_GNU
#define SIZE_OF_LINE  2*1024   // Big line to accomodate long function names 
#else
#define SIZE_OF_LINE  64*1024   // Big line to accomodate long function names 
#endif


FILE* fp;
char line[SIZE_OF_LINE];           //input line of file
char rule[SIZE_OF_LINE];           //to store a full rule
string dumpfilename;               //filename of dump file
string rulefilename;               //filename of rule file
string outputfilename;             //filename of outputfile
int number_of_functions;           //number of functions
int hwcounters = 0;                //are hardware counters used? default 0
int profilestats = 0;              //are profile stats used? default 0  (standard deviation)
int excl=0;                        //is element exlusive?  default 0 
int number_excluded;               //number of functions excluded
int apply_default_rules=1;         //should we apply default rules?  default 1
int numstmts;                      //number of statements in a single rule
int verbose=0;                     //verbose mode?  default 0
int output=0;                      //do we output to a file? default 0
int groupnames_used=0;             //groupnames used? default 0;
int command_groupname=0;           //is this a command with a groupname? default 0;
pprof_elem** elemArray;            //array that holds the pprof_elem objects
string** excludedFunctions;        //to hold the function names that will be excluded
int** countExcluded;               //array to hold number of times a function is excluded


//putLine() takes a line and writes it out to the file that is currently opened.
//It prints out an error message and exits if an output error occurs.
void putLine(string s){
  if(fprintf(fp,"%s\n",s.c_str())<0){
    perror("Error:  fprintf reports an error\n");
    exit(0);
  }//if
}//putLine()

//getLine() simply reads in a line from the file that is currently opened and stores
//it in the line variable.  It prints out an error message and exits if an input error 
//occurs.
void getLine(){
  if(fgets(line, sizeof(line),fp)==NULL){
    perror("Error:  fgets returns NULL\n");
    exit(0);
  }//if
}//getLine();

//getFunctionName() finds the name of the function that is enclosed in quotation marks
//in the line.  It then sets the line buffer so it holds the substring that follows 
//the closing quotation mark.  Some names have spaces attached to the end, and others do 
//not.  Determine if there are any extra spaces, and remove them.
string getFunctionName(){
  string rest = line;
  rest = rest.substr(rest.find("\"")+1);
  string s = rest.substr(0,rest.find("\""));  
  //now, set rest to the remainder of the string after the last quote.  Copy to 
  //the line buffer.
  rest = rest.substr(s.length()+1);
  strncpy(line, rest.c_str(), rest.length()+1);
  line[rest.length()]='\0';
  //s now equals the string enclosed between the two quotes.  Now, remove any
  //extraneous spaces that may be attached at the end of s.  Then, return s.
  int i=s.rfind(" ",0);
  while(i==s.length()){
    s=s.substr(0,s.length()-1);
    i=s.rfind(" ",0);
  }//while
  return s;
}//getFunctionName()

//getGroupNamesFromLine() finds the GROUP="g1 | g2 ..." string in the line
//and will return a string containing only the names of the groups.  
string getGroupNamesFromLine(){
  string gnames = line;
  gnames=gnames.substr(gnames.find("\"")+1);
  gnames=gnames.substr(0,gnames.find("\""));
  //now string l is the string contained between the starting and ending 
  //quotation marks.
  return gnames;  
}//getGroupNamesFromLine()


//select()prints out the output that can be used to implement a select file.
//Currently it prints it out to the screen.
void select(){
  //if we are sending the output to a file, first open and write header.
  //then, cycle through array of function names and print out one per line.
  //write header, and close file.
  if(output){
    if((fp=fopen(outputfilename.c_str(), "w"))==NULL){
      string errormessage="Error: Could not open outputfile: ";
      errormessage+=outputfilename+"\n";
      perror(errormessage.c_str());
      exit(0);
    }//if    
    putLine("# Selective instrumentation: Specify an exclude/include list.\n\n");
    putLine("BEGIN_EXCLUDE_LIST\n");
    for(int i=0; i<number_excluded;i++){
      putLine((*excludedFunctions[i])+"\n");
    }//for
    putLine("END_EXCLUDE_LIST\n"); 
    fclose(fp);
  }//if
  else{ //if !output, print to screen in the same fashion that we printed to file
    printf("\n\n\n");
    printf("# Selective instrumentation: Specify an exclude/include list.\n\n");
    printf("BEGIN_EXCLUDE_LIST\n");
    for(int i=0; i<number_excluded;i++){
      printf("%s\n",(*excludedFunctions[i]).c_str());
    }//for
    printf("END_EXCLUDE_LIST\n");
  }//else
}//select()

//initialize() completes several preliminary tasks.  It starts reading in
//the first line and checks to see if it is in the correct format.
//The second line is then read in, and
//the number of functions and the version are extracted.  It also checks to see if
//profilestats should be turned on.  Next, the array of objects is initialized.  
//finally one more line is read in which contains the headers.  From this, we determine
//whether or not we use hardware counters
void initialize(){
  //read in first line
  getLine();

  //for now, check to make sure it includes a "dep" string
  //if(strstr(line,".dep")==NULL){
  //printf("Incorrect format (should contain \".dep\") in file %s: %s\n", dumpfilename, line);
  //exit(0);
  //}//if
  
  //read in second line, which contains the number of functions and version
  getLine();
  sscanf(line,"%d %*s", &number_of_functions);

  //check to see if there is standard deviation data in file
  if(strstr(line,"-stddev")!=NULL)
    profilestats=1;

  //initialize array of functions
  elemArray = new pprof_elem*[number_of_functions];

  for(int i=0; i<number_of_functions; i++){
    elemArray[i] = new pprof_elem;
  }//for
  
  //get the next line, which contains the format of the data
  getLine();

  //if the second heading in the line is counts, then we know our version uses
  //hardware counters
  if( strstr(line, "counts")!=NULL)
    hwcounters=1;
}//initialize()

//fillTable() reads in all the data from the file.  It is set to read in 2x the 
//number of fucntions, so it needs to be pre-pointed to either the mean or total
//section of the dump file.  Then it adds the fields to the objects in the array.
//this method ASSUMES that the section will contain 2*the number of functions of 
//entries.  It assumes that in the first half, each function is mentioned once, and
//in the second half, each function is mentioned once.
void fillTable(){
  double u;
  double c;
  double p;
  double co;
  double tc;
  double nc;
  double ns;
  double sd;
  double unitspercall;
  string name;
  string gnames;
  int i;

  //now we are at the total data.  Our first run will fill in most of the data
  //for each element.  we need to determine if the line being read in is
  //inclusive or exclusive, if there are hw counters or timing data, and if
  //there is standard deviation data.

  //determine if groupnames are used
  if(strstr(line, "GROUP=\"")!=NULL)
    groupnames_used=1;
 
  for(i=0; i<number_of_functions; i++){
    //there are two line for each element.  here we process the first line.
    //first determine whether or not it is inclusive or exclusive and if it uses group names
    if(strstr(line, "excl")!=NULL)
      excl=1;
    else
      excl=0;
    //parse the line
    name=getFunctionName();
    (*elemArray[i]).setName(name);
    if(hwcounters){
      if(excl){
	sscanf(line, "%*s %lG", &co);
	(*elemArray[i]).setCount(co);
      }//if	
      else{
	sscanf(line, "%*s %lG", &tc);
	(*elemArray[i]).setTotalCount(tc);
      }//else
      //check to see if groupnames are used in this file
      if(groupnames_used){
	gnames=getGroupNamesFromLine();
	(*elemArray[i]).setGroupNames(gnames);
      }//if
    }//if
    else{
      if(excl){
	sscanf(line, "%*s %lG", &u);
	(*elemArray[i]).setUsec(u);
      }//if	
      else{
	sscanf(line, "%*s %lG", &c);
	(*elemArray[i]).setCumusec(c);
      }//else
      //check to see if groupnames are used in this file
      if(groupnames_used){
	gnames=getGroupNamesFromLine();
	(*elemArray[i]).setGroupNames(gnames);
      }//if	
    }//else
    //now read in second line of pair
    getLine();
    if(hwcounters){
      //this line is in the following format: %time msec totalmsec #call #subrs counts/call name
      //some of these fields we already have, so we can ignore them.  Fill in the ones we don't   
      sscanf(line, "%lG %*s %*s %lG %lG %lG", &p, &nc, &ns, &unitspercall);
      (*elemArray[i]).setPercent(p);
      (*elemArray[i]).setNumCalls(nc);
      (*elemArray[i]).setNumSubrs(ns);
      (*elemArray[i]).setCountsPerCall(unitspercall);
    }//if
    else{
      //this line is in the following format: %time msec totalmsec #call #subrs usec/call name
      //if there is standard deviation, then it comes before the name attribute
      //some of these fields we already have, so we can ignore them.  Fill in the ones we don't   
      if(profilestats){
	sscanf(line, "%lG %*s %*s %lG %lG %lG %lG", &p, &nc, &ns, &unitspercall, &sd);
	(*elemArray[i]).setPercent(p);
	(*elemArray[i]).setNumCalls(nc);
	(*elemArray[i]).setNumSubrs(ns);
	(*elemArray[i]).setUsecsPerCall(unitspercall);
	(*elemArray[i]).setStdDeviation(sd);
      }//if
      else{
	sscanf(line, "%lG %*s %*s %lG %lG %lG", &p, &nc, &ns, &unitspercall);
	(*elemArray[i]).setPercent(p);
	(*elemArray[i]).setNumCalls(nc);
	(*elemArray[i]).setNumSubrs(ns);
	(*elemArray[i]).setUsecsPerCall(unitspercall);
      }//else
    }//else
    //now, try and get the next line
    getLine();
  }//for
  //now run through the second half of the functions listed.  this half will contain
  //either the excl or incl data that the previous run did not include.
  for(i=0;i<number_of_functions;i++){
    if(strstr(line,"excl")!=NULL)
      excl=1;
    else 
      excl=0;
    name=getFunctionName();
    for(int j=0;j<number_of_functions;j++){
      if(name==(*elemArray[j]).getName()){
	if(hwcounters){
	  if(excl){
	    sscanf(line, "%*s %lG", &co);
	    (*elemArray[j]).setCount(co);
	  }//if
	  else{
	    sscanf(line, "%*s %lG", &tc);
	    (*elemArray[j]).setTotalCount(tc);
	  }//else	
	}//if
	else{
	  if(excl){
	    sscanf(line, "%*s %lG", &u);
	    (*elemArray[j]).setUsec(u);
	  }//if	
	  else{
	    sscanf(line, "%*s %lG", &c);
	    (*elemArray[j]).setCumusec(c);
	  }//else	
	}//else
      }//if
    }//for
    //we can skip the second line of each pair, so do two getLine()'s to get the 
    //first of each pair of lines
    getLine();
    getLine();
  }//for
}//fillTable()

//printTable() for debugging.  Print out objects in array.
void printTable(){
  for(int i=0; i<number_of_functions; i++){
    (*elemArray[i]).printElem();
  }//for
}//printTable()

//addFunctionName() takes in a name and checks to see if it is already in the exclude array.
//If it is not, then it adds it to the array, and increments the count.
void addFunctionName(string s){
  for(int i=0;i<number_excluded;i++){
    if(*excludedFunctions[i]==s)
      return;
  }//for
  *excludedFunctions[number_excluded]=s;
  number_excluded++;
}//addFunctionName()


//processCommand() parses the command and processes it accordingly.
void processCommand(int more){
  char f[32];
  char op;
  double number;
  //zero out the field character array
  for(int k=0;k<32;k++){
    f[k]='\0';
  }//for
  sscanf(line, "%s %c %lG", f, &op, &number);
#ifdef DEBUG
  printf("f is %s \n", f);
#endif /* DEBUG */
  //did we read something in for the field?  If not give error and return
  if(strlen(f)==0){
    printf("Error: The field variable is empty.  Rule: '%s'\n",rule);
    return;
  }
  string field=string(f);
  if(field=="numcalls"){
    if(op=='<'){
      for(int i=0; i<number_of_functions; i++){
	if((*elemArray[i]).getNumCalls() < number)
	  //addFunctionName((*elemArray[i]).getName());
	  (*countExcluded[i])++;
      }//for
    }//if
    else if(op=='>'){
      for(int i=0; i<number_of_functions; i++){
	if((*elemArray[i]).getNumCalls() > number)
	  (*countExcluded[i])++;
      }//for	
    }//else if
    else if(op=='='){
      for(int i=0; i<number_of_functions; i++){
	if((*elemArray[i]).getNumCalls() == number)
	  (*countExcluded[i])++;
      }//for
    }//else if
    else{//unregonized op
      printf("Error:  unrecognized operator: '%c' in rule: '%s'\n", op,rule);
    }//else
  }//if
  else if(field=="numsubrs"){
    if(op=='<'){
      for(int i=0; i<number_of_functions; i++){
	if((*elemArray[i]).getNumSubrs() < number)
	  (*countExcluded[i])++;
      }//for
    }//if
    else if(op=='>'){
      for(int i=0; i<number_of_functions; i++){
	if((*elemArray[i]).getNumSubrs() > number)
	  (*countExcluded[i])++;
      }//for	  
    }//else if
    else if(op=='='){
      for(int i=0; i<number_of_functions; i++){
	if((*elemArray[i]).getNumSubrs() == number)
	  (*countExcluded[i])++;
      }//for
    }//else if
    else{//unregonized op
      printf("Error:  unrecognized operator: '%c' in rule: '%s'\n", op,rule);
    }//else
  }//else if
  else if(field=="percent"){
    if(op=='<'){
      for(int i=0; i<number_of_functions; i++){
	if((*elemArray[i]).getPercent() < number)
	  (*countExcluded[i])++;
      }//for
    }//if
    else if(op=='>'){
      for(int i=0; i<number_of_functions; i++){
	if((*elemArray[i]).getPercent() > number)
	  (*countExcluded[i])++;
      }//for	  
    }//else if
    else if(op=='='){
      for(int i=0; i<number_of_functions; i++){
	if((*elemArray[i]).getPercent() == number)
	  (*countExcluded[i])++;
      }//for
    }//else if
    else{//unregonized op
      printf("Error:  unrecognized operator: '%c' in rule: '%s'\n", op,rule);
    }//else
  }//else if
  else if(field=="usec"){
    if(hwcounters){
      printf("Error:  no timing data for this file -- use count or totalcount.  Rule: '%s'\n",rule);
    }//if
    else{
      if(op=='<'){
	for(int i=0; i<number_of_functions; i++){
	  if((*elemArray[i]).getUsec() < number)
	    (*countExcluded[i])++;
	}//for
      }//if
      else if(op=='>'){
	for(int i=0; i<number_of_functions; i++){
	  if((*elemArray[i]).getUsec() > number)
	    (*countExcluded[i])++;
	}//for	  
      }//else if
      else if(op=='='){
	for(int i=0; i<number_of_functions; i++){
	  if((*elemArray[i]).getUsec() == number)
	    (*countExcluded[i])++;
	}//for
      }//else if
      else{//unregonized op
	printf("Error:  unrecognized operator: '%c' in rule: '%s'\n", op,rule);
      }//else
    }//else
  }//else if
  else if(field=="cumusec"){
    if(hwcounters){
      printf("Error:  no timing data for this file -- try count and totalcount.  Rule: '%s'\n",rule);
    }//if
    else{
      if(op=='<'){
	for(int i=0; i<number_of_functions; i++){
	  if((*elemArray[i]).getCumusec() < number)
	    (*countExcluded[i])++;
	}//for
      }//if
      else if(op=='>'){
	for(int i=0; i<number_of_functions; i++){
	  if((*elemArray[i]).getCumusec() > number)
	    (*countExcluded[i])++;
	}//for	  
      }//else if
      else if(op=='='){
	for(int i=0; i<number_of_functions; i++){
	  if((*elemArray[i]).getCumusec() == number)
	    (*countExcluded[i])++;
	}//for
      }//else if
      else{//unregonized op
	printf("Error:  unrecognized operator: '%c' in rule: '%s'\n", op,rule);
      }//else
    }//else
  }//else if
  else if(field=="count"){
    if(hwcounters){
      if(op=='<'){
	for(int i=0; i<number_of_functions; i++){
	  if((*elemArray[i]).getCount() < number)
	    (*countExcluded[i])++;
	}//for
      }//if
      else if(op=='>'){
	for(int i=0; i<number_of_functions; i++){
	  if((*elemArray[i]).getCount() > number)
	    (*countExcluded[i])++;
	}//for	  
      }//else if
      else if(op=='='){
	for(int i=0; i<number_of_functions; i++){
	  if((*elemArray[i]).getCount() == number)
	    (*countExcluded[i])++;
	}//for
      }//else if
      else{//unregonized op
	printf("Error:  unrecognized operator: '%c' in rule: '%s'\n", op,rule);
      }//else
    }//if
    else{
      printf("Error:  no hardware counter data for this file -- use usec and cumusec.  Rule: '%s'\n",rule);
    }//else
  }//else if
  else if(field=="totalcount"){
    if(hwcounters){
      if(op=='<'){
	for(int i=0; i<number_of_functions; i++){
	  if((*elemArray[i]).getTotalCount() < number)
	    (*countExcluded[i])++;
	}//for
      }//if
      else if(op=='>'){
	for(int i=0; i<number_of_functions; i++){
	  if((*elemArray[i]).getTotalCount() > number)
	    (*countExcluded[i])++;
	}//for	  
      }//else if
      else if(op=='='){
	for(int i=0; i<number_of_functions; i++){
	  if((*elemArray[i]).getTotalCount() == number)
	    (*countExcluded[i])++;
	}//for
      }//else if
      else{//unregonized op
	printf("Error:  unrecognized operator: '%c' in rule: '%s'\n", op,rule);
      }//else
    }//if
    else{
      printf("Error:  no count data for this file -- try usec and cumusec.  Rule: '%s'\n",rule);
    }//else
  }//else if
  else if(field=="stddev"){
    if(!profilestats){
      printf("Error:  there is no standard deviation for this data.  Rule: '%s'\n",rule);
    }//if
    else{
      if(op=='<'){
	for(int i=0; i<number_of_functions; i++){
	  if((*elemArray[i]).getStdDeviation() < number)
	    (*countExcluded[i])++;
	}//for
      }//if
      else if(op=='>'){
	for(int i=0; i<number_of_functions; i++){
	  if((*elemArray[i]).getStdDeviation() > number)
	    (*countExcluded[i])++;
	}//for	  
      }//else if
      else if(op=='='){
	for(int i=0; i<number_of_functions; i++){
	  if((*elemArray[i]).getStdDeviation() == number)
	    (*countExcluded[i])++;
	}//for
      }//else if
      else{//unregonized op
	printf("Error:  unrecognized operator: '%c' in rule: '%s'\n", op,rule);
      }//else
    }//else
  }//else if
  else if(field=="usecs/call"){
    if(hwcounters){
      printf("Error:  no timing data for this file -- use counts/call.  Rule: '%s'\n",rule);
    }//if
    else{
      if(op=='<'){
	for(int i=0; i<number_of_functions; i++){
	  if((*elemArray[i]).getUsecsPerCall() < number)
	    (*countExcluded[i])++;
	}//for
      }//if
      else if(op=='>'){
	for(int i=0; i<number_of_functions; i++){
	  if((*elemArray[i]).getUsecsPerCall() > number)
	    (*countExcluded[i])++;
	}//for	  
      }//else if
      else if(op=='='){
	for(int i=0; i<number_of_functions; i++){
	  if((*elemArray[i]).getUsecsPerCall() == number)
	    (*countExcluded[i])++;
	}//for
      }//else if
      else{//unregonized op
	printf("Error:  unrecognized operator: '%c' in rule: '%s'\n", op,rule);
      }//else
    }//else
  }//else if
  else if(field=="counts/call"){
    if(!hwcounters){
      printf("Error:  no timing data for this file -- use usecs/call.  Rule: '%s'\n",rule);
    }//if
    else{
      if(op=='<'){
	for(int i=0; i<number_of_functions; i++){
	  if((*elemArray[i]).getCountsPerCall() < number)
	    (*countExcluded[i])++;
	}//for
      }//if
      else if(op=='>'){
	for(int i=0; i<number_of_functions; i++){
	  if((*elemArray[i]).getCountsPerCall() > number)
	    (*countExcluded[i])++;
	}//for	  
      }//else if
      else if(op=='='){
	for(int i=0; i<number_of_functions; i++){
	  if((*elemArray[i]).getCountsPerCall() == number)
	    (*countExcluded[i])++;
	}//for
      }//else if
      else{//unregonized op
	printf("Error:  unrecognized operator: '%c' in rule: '%s'\n", op,rule);
      }//else
    }//else
  }//else if
  else{ //unrecognized field
    printf("Error:  unrecognized field: '%s' in rule: '%s'\n", field.c_str(),rule);
  }//else
  if(!more){
    for(int i=0;i<number_of_functions;i++){
      if((*countExcluded[i])==numstmts){
	addFunctionName((*elemArray[i]).getName());
	if(verbose)
	  printf("\t%s\n",(*elemArray[i]).getName().c_str());
      }//if
    }//for
  }//if
}//processCommand()

//if the rule has a group prefix, we need to process that prefix, and then 
//remove the group prefix from the line.
void processGroupPrefix(){
  //first, remove the group prefix from the line
  string rest = line;
  string sline=line;
  string s=rest.substr(0,rest.find(":"));
  rest=rest.substr(s.length()+1);
  strncpy(line, rest.c_str(), rest.length()+1);
  //now check to see if we even have any group data -- double check?
  if(!groupnames_used){
    printf("ERROR: There is no groupnames data for this file. Rule that caused the error: %s\n",sline.c_str());
    return;
  }//if
  //count this as a statement, so increment the count
  numstmts++;
  //now, increment the excluded array for all element positions that
  //belong to group s.  NOTE: elements may belong to multiple groups,
  //so we will use the find() routine to check.
  for(int i=0;i<number_of_functions;i++){
    if((*elemArray[i]).getGroupNames().find(s)!=string::npos){
      //we think the group name is included, however we have to make sure that
      //we actually have the groupname and that the string we are looking for 
      //is not a substring of a larger groupname, ie if we are looking for a 
      //a group called TAU_USER we want to make sure that any matching groups 
      //not substrings of larger groups, such as TAU_USER_DEFAULT.  First, we 
      //check that the first character of the found string is either at index 0, 
      //or there is a space preceding it.  If this is true, we chop off anything
      //that might be preceding it.  At this point, we can use sscanf() to 
      //copy the first string up until the first whitespace into a new variable.
      //Then, we just have to check that the two lengths are equal.
      string gn=(*elemArray[i]).getGroupNames();
      char gnarray[SIZE_OF_LINE];
      strncpy(gnarray, gn.c_str(), gn.length()+1);
      int index = gn.find(s);
      if(index==0 || gnarray[index-1]==' '){
	//now, check the end of the string.  
	gn=gn.substr(index);
	strncpy(gnarray, gn.c_str(),gn.length()+1);
	char temp[SIZE_OF_LINE];
	sscanf(gnarray, "%s",temp);
	gn = temp;
	//gn now equals the first string in gnarray--now just check to see if lengths match!
	if(gn.length()==s.length()){
	  (*countExcluded[i])++;
	}//if
      }//if
    }//if
  }//for
}//processGroupPrefix()

//parseRules() will determine if the rule is a simple rule or a compound
//rule.  If it is a simple rule, then pass it to the processCommand() with
//an argument of 0, telling processCommand() that there is no more.  Else,
//pass a 1, telling processCommand() there is more to come.
void parseRules(){
  //first, we need to see if this is a group specific command
  //if it is, it will be the format group:rule1 [& rule2 & ...]  
  //so, look for the ':'  if there is one, send it to procesGroupPrefix()
  if(strstr(line, ":")!=NULL)
    processGroupPrefix();
  //now if there's the " & " string, then we know there is atleast two rules.
  //so first, increment the numstmts variable and call the processCommand()
  //function with the value of 1 passed to it, signalling that there are
  //more rules to follow.  Then, cut off the first rule in the line and
  //recall this function.
  numstmts++;
  if(strstr(line," & ")!=NULL){
    processCommand(1);
    int found=0;
    int index=0;
    for(int i=0; i<strlen(line); i++){
      if(!found){
	if(line[i]=='&'){
	  found++;
	  i++;
	}//if
      }//if
      else{
	line[index]=line[i];
	index++;
      }//else
    }//for
    line[index]='\0';
    parseRules();
  }//if
  //else, we know there is just a single rule, so simply call processCommand()
  //with a 0 passed in to signify that there are no rules to follow.
  else{
    processCommand(0);
  }//else
}//parseRules()

//acceptRules() begins by initializing the array of function names to be
//excluded.  It then checks to see if we are using a file of rules or
//the default rule.
void acceptRules(){
  number_excluded=0;
  //initialize array of functions
  excludedFunctions = new string*[number_of_functions];
  countExcluded=new int*[number_of_functions];
  for(int i=0; i<number_of_functions; i++){
    excludedFunctions[i] = new string();
    countExcluded[i] = new int(0);
  }//for
  
  //now, are we applying the default rule?  
  if(apply_default_rules){
    char defaultrule[]= "numcalls > 1000000 & usecs/call < 2";
    strncpy(line, defaultrule, strlen(defaultrule)+1);
    strncpy(rule,line,strlen(line)+1);
    if(verbose)
     printf("___________________________________________________________\n%s\n",line);
    numstmts=0;
    //first we need to put a copy of the whole rule in the rule variable
    //this is used for error messages and debugging
    strncpy(rule,line,strlen(line)+1);
    parseRules();
  }//if

  else{  //don't use default rule
    //now open the file up and start reading
    //all lines that begin with a # are a comment
    if((fp=fopen(rulefilename.c_str(), "r"))==NULL){
      string errormessage="Error: Could not open rulefile: ";
      errormessage+=rulefilename+"\n";
      perror(errormessage.c_str());
      exit(0);
    }//if    
    
    while(fgets(line,sizeof(line),fp)!=NULL){
      //check to see if line is a comment, newline, or space--if so, ignore it
      if(line[0]!='#'&&line[0]!='\n'&&line[0]!=' '){
	if(verbose)
	  printf("___________________________________________________________\n%s",line);
	numstmts=0;
	strncpy(rule,line,strlen(line)+1);
	if(rule[strlen(rule)-1]=='\n'){
	  rule[strlen(rule)-1]='\0';
	}//if
	parseRules();
	//zero out array again
	for(int i=0; i<number_of_functions;i++){
	  countExcluded[i]= new int(0);
	}//for
      }//if
    }//while 
    fclose(fp);
  }//else 
}//acceptRules()


//main()
int main (int argc, char *argv[]){
  int errflag;                 //determine if an error has occurred
  outputfilename="select";     //default
  int df = 0;                  //have we entered a dump file?
  int p=0;                     //is the print flag set?
  int i,j;
#ifdef TAU_WINDOWS
  int optind = 0;
#else
  extern char *optarg;
  extern int optind;
#endif //TAU_WINDOWS

#ifdef TAU_WINDOWS
  /* -- parse command line arguments ---------------------------------------- */  
  errflag = FALSE;
  for( int j = 1; j < argc; j++){  
    char *argchar = argv[j];
    switch(argchar[0]){
    case '-':{
      switch(argchar[1]){
      case 'f': /* -- name of pprof dump *F*ile -- */
	//first check to see if there is something after the -d option
	if(argv[j+1]==NULL){
	  errflag=true;
	}//if
	else{
	  //if the first character is a '-', maybe they forgot the file name
	  if(argv[j+1][0]=='-'){
	    cout<<"It is likely that you have forgotten to give the filename after the -d option!"<<endl;
	    dumpfilename=arv[j+1];
	    j=j+1;
	  }//if
	  else{
	    dumpfilename=argv[j+1];
	    j=j+1;
	  }//else
	}//else
	df++;
	break;
      case 'r': /* -- name of *R*ule file -- */
	//first check to see if there is something after the -r option
	if(argv[j+1]==NULL){
	  errflag=true;
	}//if
	else{
	  //if the first character is a '-', maybe they forgot the file name
	  if(argv[j+1][0]=='-'){
	    cout<<"It is likely that you have forgotten to give the filename after the -f option!"<<endl;
	    rulefilename=arv[j+1];
	    j=j+1;
	  }//if
	  else{
	    rulefilename=argv[j+1];
	    j=j+1;
	  }//else
	}//else
	appy_default_rules=false;
	break;
      case 'p': /* -- *P*rint function data -- */
	p++;
	break;
      case 'o': /* -- select *O*utput file name -- */
	//first check to see if there is something after the -s option
	if(argv[j+1]==NULL){
	  errflag=true;
	}//if
	else{
	  //if the first character is a '-', maybe they forgot the file name
	  if(argv[j+1][0]=='-'){
	    cout<<"It is likely that you have forgotten to give the filename after the -o option!"<<endl;
	    outputfilename=arv[j+1];
	    j=j+1;
	  }//if
	  else{
	    outputfilename=argv[j+1];
	    j=j+1;
	  }//else
	}//else
	output=1;
	break;
      case 'v':  /* -- *V*erbose mode -- */
	verbose=true;
	break;
      default:
	errflag = true;
	break;
      }
    }//switch
    break;	  
    default:
      errflag = true;
      break;
    }//switch
  }//for
#else  
  /* -- parse command line arguments ---------------------------------------- */
  int ch;       //to hold option character from command line
  errflag = false;
  while ( (ch = getopt (argc, argv, "f:npr:o:v")) != EOF ) {
    switch ( ch ) {
    case 'f': /* -- name of pprof dump *F*ile -- */
      dumpfilename = optarg;
      df++;
      break;
    case 'p': /* -- *P*rint function data -- */
      p++;
      break;
    case 'r': /* -- name of *R*ule file -- */
      rulefilename = optarg;
      apply_default_rules=false;
      break;
    case 'o': /* -- name of *S*elect output file -- */
      output=1;
      outputfilename = optarg;
      break;
    case 'v': /* -- *V*erbose mode -- */
      verbose=true;
      break;
    default:
      errflag = true;
      break;
    }//while
  }//switch
#endif //TAU_WINDOWS

  if(!df){
    fprintf(stderr,"You must enter a filename that specifies a pprof dump file\n\n");
    errflag=true;
  }//if

  //if there was an error, print out the usage and exit
  if(errflag){
    fprintf(stderr, "usage: %s -f filename [-n] [-r filename] [-o filename] [-v]\n", argv[0]);
    fprintf(stderr," -f filename : specify filename of pprof dump file\n");
    fprintf(stderr," -p : print out all functions with their attributes\n");
    fprintf(stderr," -o filename : specify filename for select file output (default: print to screen\n");
    fprintf(stderr," -r filename : specify filename for rule file\n");
    fprintf(stderr," -v : verbose mode (for each rule, print out rule and all functions that it excludes)\n");
    exit (1);
  }//if(errflag)

  //first open up the dumpfile to read in.  fp is now associated with
  //the dump file.
  if((fp=fopen(dumpfilename.c_str(), "r"))==NULL){
    printf("Error: Could not open %s",dumpfilename.c_str());
    exit(0);
  }//if

  initialize();

  //now, we want to store just the total information in the table, so we
  //can skip over all the individual data at the beginning. 
  while(line[0]!='t'){
    getLine();
  }//while

  //now fill our array of elements
  fillTable();
  if(p)
    printTable();
  
  //now we are done with the dump file, so close it up.  fp is now clear.
  fclose(fp); 

  //accept rules, either the default rule or a set of user-defined rules,
  //and apply them on our array of elements
  acceptRules();

  //now create the select output
  select();

  //cleanup
  for(i=0;i<number_of_functions;i++){
    delete elemArray[i];
  }//for
  delete [] elemArray;
  for(i=0;i<number_of_functions;i++){
    delete countExcluded[i];
  }//for
  delete [] countExcluded;
  for(j=0;j<number_excluded;j++){
    delete excludedFunctions[j];
  }//for
  delete [] excludedFunctions;

  exit(0);
}//main

/***************************************************************************
 * $RCSfile: tau_reduce.cpp,v $   $Author: sameer $
 * $Revision: 1.11 $   $Date: 2002/12/20 20:09:42 $
 * TAU_VERSION_ID: $Id: tau_reduce.cpp,v 1.11 2002/12/20 20:09:42 sameer Exp $
 ***************************************************************************/

