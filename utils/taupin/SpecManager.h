#pragma once
#include <list>
#include <map>
#include<string>
#include "Arguments.h"
using namespace std;

//Rtn Information copied here because of symbol conflict
//nothing much here
class RtnInfo{
private:
	string ImgName; 
	string RtnName;
	long RtnAddr;
	int ProfileFlag;
public:
	RtnInfo(string ImgName, string RtnName, int ProfileFlag );
	~RtnInfo();
	int GetFlag();
	string GetImg();
	string GetRtn();
};

//Image information held here 
class ImageInfo{
private:
	string ImgeName;
	//address of the image loaded 
	//currently not used 
	//later can be useful to find routines quickly
	long ImgStartAddr;
	long ImgEndAddr;
	int ProfileFlag;
	//this is a map of Rtnname and Rtninfo object 
	//for each routine object
	map<string,RtnInfo*> rtns;
public:
	ImageInfo(string ImageName, int ProfileFlag);
	~ImageInfo();
	//adds routine to the list
	void AddRtn(string rtn);
	//returns image name
	string GetImage();
	//Finds out the Rtn with the name
	RtnInfo * FindRtn(string rtn);
	//just prointing
	void PrintRtns();
};

class SpecManager
{
private:
	list<RuleSet*> rules;
	//the following are only for default type rules ..no regular expression
	//Create here two different indexex based on routine and image for fast search
	map<string,RuleSet*> rtnIndx;
	map<string,RuleSet*> imgIndx;
	//this is for image search
	map<string,ImageInfo*> imagemap;
	//maintains the current specfile
	string SpecFile;
	//current argument
	Arguments * CurArgs;
	//current image being dealt with
	ImageInfo *CurImage;
	//XmlDocument spec;
	//this is an initializing of the MPI instrumengtation rules 
	void InitDefaultRules();
	//helper funcs for adding image and routines information  
	void AddImage(string img,int flag);
	void AddRtn(string rtn,string img);
	int ParseFlag(string flag);
	bool MatchRecurse(char *txt,int txt_len ,int txt_indx, char * expr, int expr_len , int expr_indx);
public:
	//constructors here
	SpecManager(void);
	SpecManager(Arguments *args);
	SpecManager(string specfile);	
	~SpecManager(void);
	//check if image or routine is instrumentable   
	bool InstImage(string image);
	bool InstRtn(string rtn,string image);
	//prints out Instrumentable  
	void PrintInsts();
	//get the profile flag
	int GetProfileFlag(string rtn, string image);
	bool IsMpiRtn(string rtn_name);
	bool SpecManager::Match(string my_txt, string my_expr);
};


