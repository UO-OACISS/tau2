/***************************************************************************************************
This module parse arguments and files to populate instrumentation rules.    
****************************************************************************************************/

#include "Arguments.h"
#include<fstream>
#include "TraceManager.h"

//some of the Ruleset access utilities
RuleSet::RuleSet(string imagePat, string rtnPat, int flag)
{
	ImagePat=imagePat;
	RtnPat = rtnPat;
	ProfileFlag=flag;
}

RuleSet::RuleSet(string imagePat, string rtnPat)
{
	ImagePat=imagePat;
	RtnPat = rtnPat;
	ProfileFlag=0;

}
RuleSet::~RuleSet()
{
}
string RuleSet::GetImagePat()
{
	return ImagePat;
}
string RuleSet::GetRtnPat()
{
	return RtnPat;
}
int RuleSet::getProfileFlag()
{
	return ProfileFlag;
}


//******Start of Argument parsing ****


Arguments::Arguments(string args)
{
	arguments=args;
	//parse rules from argument and file 
	ParseRules();
	ParseFiles();
}

Arguments::Arguments(char ** args, int argcnt)
{
	this->args=args;
	this->argcnt=argcnt;
	//parse rules from argument and file
	this->GetRules();
	this->GetFiles();
}

Arguments::~Arguments(void)
{
}

void Arguments::PopulateRule(string rule)
{
	string image;
	string rtn;
	string flag;
	//start parsing the rules based on ! delemitor
	int prevpos=rule.find('!');
	if(prevpos<0) return;
	//grab the first part
	image=rule.substr(0,prevpos);
	prevpos++;
	int curpos=rule.find('!',prevpos);
    if(curpos<0) return;
	//grab the second part
	rtn=rule.substr(prevpos,curpos-prevpos);
	curpos++;
	prevpos=rule.length();
	//grab the last part
	flag=rule.substr(curpos,prevpos-curpos);
    
	if(image.length()<=0 || rtn.length()<=0 || flag.length()<=0) return;
	//cerr<<"Populating "<<image<<"::"<<rtn<<endl;	
	RuleSet* robj=new RuleSet(image,rtn ,this->ParseFlag(flag));
	//insert the parsed rule object in the list
	this->rulechain.insert(rulechain.begin(), robj);

}

int Arguments::ParseFlag(string flag)
{
	int flagint=0;
	int prevpos=0,curpos=0;
	int flaglen=flag.length();
	while(prevpos<flaglen)
	{
		curpos=flag.find(',',prevpos);
		if(curpos<flaglen)
		{
			flagint&=atoi(flag.substr(prevpos,flaglen-prevpos).c_str());
			break;
		}
		flagint=flagint&atoi(flag.substr(prevpos,curpos).c_str());
		prevpos=curpos+2;
	}
	return flagint;
}


void Arguments::PopulateFile(string file)
{
	this->files.insert(files.begin(),file);	
}

int Arguments::GetPinArgC()
{
    int i=3,cnt=0;
	//just scan through args excluding the previous part
	while((i<this->argcnt)&& (strcmp(this->args[i],"--")!=0))
	{
		i++;
		cnt++;
	}
	//cerr<<"ARGCNT:"<<this->argcnt<<"::CNT:"<<cnt<<endl;
	return (this->argcnt-cnt-1);
}

char ** Arguments::GetPinArgV()
{
    int i=3,cnt=0;
	//scan through arguments untill --
	while((i<this->argcnt)&& (strcmp(this->args[i],"--")!=0))
	{
		i++;
		cnt++;
	}    
	char ** argv=(char**)malloc(sizeof(char**)*(this->argcnt-cnt+2));
	argv[0]=args[0];
	argv[1]=args[1];
	argv[2]=args[2];

	//Compose new argument parameters to be passed to PIN
	for(int j=3+cnt,i=3;j<this->argcnt;j++)
	{
		//cerr<<"COMPOSE:"<<args[j]<<endl;
		argv[i]=args[j];
		i++;
	}

	return argv;
}

void Arguments::ParseRules()
{
	int curPos=0;
	int prevPos=0;
	//move two indexex trapping the argument string 
	//this is for string parsing 
	while(curPos>=0 && prevPos>=0)
	{
			prevPos=this->arguments.find("-r",curPos+1);
			if(prevPos>=0)
			{
				prevPos+=2;
				curPos=this->arguments.find('-',prevPos);
				if(curPos<0)
					break;
				PopulateRule(arguments.substr(prevPos,curPos-prevPos));
					
			}					
	}
}

void Arguments::GetRules()
{	
	//loop through all the arguments
	for(int i=3;i<this->argcnt-1;i++)
	{
		if(strcmp(this->args[i],"-r")==0)
		{
			i++;
			//cerr<<"Populate rule::"<<this->args[i]<<endl;
			//if -r option is found grab the next string
			//and parse the rule
			PopulateRule(this->args[i]);  
		}
	}
}

void Arguments::ParseFiles()
{
	int curPos=0;
	int prevPos=0;
	//parses from a string the file options 
	while(curPos>=0 && prevPos>=0)
	{
			prevPos=this->arguments.find("-f",curPos+1);
			if(prevPos>=0)
			{
				prevPos+=2;
				curPos=this->arguments.find('-',prevPos);
				if(curPos<0)
					break;
				PopulateFile(arguments.substr(prevPos,curPos-prevPos));
					
			}					
	}
}

void Arguments::GetFiles()
{	
	//iterate through all the arguments
	for(int i=3;i<this->argcnt-1;i++)
	{
		if(strcmp(this->args[i],"-f")==0)
		{
			i++;
			//if -f option is found 
			//get the file names as next string 
			//then parse it
			this->PopulateFile(this->args[i]);
			this->ParseRuleFile(this->args[i]);
		}
	}
}

void Arguments::ParseRuleFile(char* filename)
{
	ifstream myinput(filename);
	if(myinput.is_open())
	{
		while(!myinput.eof())
		{
			string ruleset;
			//get rules from the file 
			getline(myinput,ruleset);
			PopulateRule(ruleset);
			DBG_TRACE(ruleset);
		}
		myinput.close();
	}else{
		cerr<<"unable to open file"<<endl;
	}
}

list<RuleSet*> Arguments::GetRuleIt()
{	
	return this->rulechain;
}

void Arguments::PrintRules()
{
	list<RuleSet*>::iterator rit=this->rulechain.begin();

	while(rit!=this->rulechain.end())
	{		
		RuleSet * robj=*rit;
		cerr<<"ImagePat::"<<robj->GetImagePat()<<" RtnPat::"<<robj->GetRtnPat()<<endl;
		rit++;
	}
}

list<string> Arguments::GetFileIt()
{
	return this->files;
}

void Arguments::PrintArgs()
{
	list<RuleSet*>::iterator rit=this->rulechain.begin();

	cerr<<"Rulechain>>"<<endl;
	//iterate through rules and print
	while(rit!=this->rulechain.end())
	{
		RuleSet* rst=(RuleSet*)*rit;
		cerr<<rst->GetImagePat()<<"#"<<rst->GetRtnPat()<<"#"<<rst->getProfileFlag()<<endl;
		rit++;
	}

	list<string>::iterator fit=this->files.begin();
	cerr<<"Files>>" <<endl;
    //iterate throug files and print
	while(fit!=this->files.end())
	{
		cerr<<*fit<<endl;
		fit++;
	}
}