#pragma once
#include<map>
#include<list>
#include<string>
#include<iostream>
using namespace std;

//Ruleset is maintained in this class
class RuleSet
{
private:
		//image pattern, and rtn patterns and profiling flag
		string ImagePat;
		string RtnPat;
		int ProfileFlag;
public:
		RuleSet(string ImagePat, string RtnPat, int flag);
		RuleSet(string ImagePat, string RtnPat);
		//functions to retrieve Ruleset parameters  
		string GetImagePat();
		string GetRtnPat();
		int getProfileFlag(); 
		~RuleSet();
};

//ARguments are parsed in this function
class Arguments
{

private:
    //maintains the list of files to be parsed
	list<string> files;
	//maintain the list of rule chain given as argument
	list<RuleSet*> rulechain;
	//this is currently not used
	//this can be used as input parsing
	string arguments;
	//the input arguments as passed 
	char ** args;
	//total arguments
	int argcnt;
	//helper functions for getting the rules and files
	void PopulateRule(string rule);
	void PopulateFile(string file);
	//helper function for parsing rules
	void ParseRules();
	void ParseFiles();
	//just simply retrieve the rule 
	void GetRules();
	void GetFiles();
	//parseout all file arguments 
	void ParseRuleFile(char* filename);
	//parse out all the Flag information
	int ParseFlag(string flag);
  
public:

	Arguments(string);
	Arguments(char ** args, int argcnt);
	~Arguments(void);
	//returns the ruleset list
	list<RuleSet*> GetRuleIt();
	list<string> GetFileIt();
	//just for printing
	void PrintRules();
	int GetPinArgC();
	char ** GetPinArgV();
	void PrintArgs();

};
