//tauprofile.cpp : Defines the entry point for the console application.
//this is just a wrapper to call the relevant Taupin
//

#include "stdafx.h"
#include <iostream>
#include <windows.h>
#include <tchar.h>
#include <string>
#include <stdlib.h>
using namespace std;

//displays the help
void Usage()
{
	cout<<"Usage: tau_pin [-n num_proc] [-r rule_1  -r rule_2 ...] -- app.exe ..."<<endl;
	cout<<"Example tau_pin  myapp.exe will instrument all"<<endl;  
	cout<<"Example: tau_pin -n 4 -r func* -- mpi.exe"<<endl;
	cout<<"Example: tau_pin -r func* -- hello.exe"<<endl;
	cout<<"If -n is there mpiexec get invoked but you need to make sure mpiexec is there"<<endl;
	cout<<"IMPORTANT: Please make sure that pdb(symbol) file of the target application is present"<<endl;
}

// retrieves environment variable info
//used for the registration process while installing
string ReadEnv(LPCSTR field)
{
	HKEY hKey_net;
	
	// Net information
    if (RegOpenKeyExA(HKEY_CURRENT_USER,
		"Environment",
        0, KEY_READ, &hKey_net) != ERROR_SUCCESS)
    {
        cerr<<"Registry net not found"<<endl;
        return "";
    }
	
	DWORD dwLength, dwType;
	string myval="";
 	if(RegQueryValueExA(hKey_net, field, NULL, NULL, NULL, &dwLength) == ERROR_SUCCESS){
			char *value = new char[dwLength];
            if(RegQueryValueExA(hKey_net, field, NULL, &dwType, (LPBYTE)value, &dwLength) == ERROR_SUCCESS){
               myval.append(value);
			}else{
				cerr<<"Registry Query failed"<<endl;
			}
            delete value;
        }
	return myval;
}

// this will se the environment variable 
void SetEnv(LPCSTR field, string value)
{
	HKEY hKey_sh;
	
    if (RegOpenKeyExA(HKEY_CURRENT_USER,
		"Environment",
        0, KEY_SET_VALUE, &hKey_sh) != ERROR_SUCCESS)
    {
        cerr<<"Registry shell not found"<<endl;
        return;
    }

	DWORD dwLen=value.length();
	DWORD hRes = RegSetValueExA(
               hKey_sh,
               field,
               0,                  
               REG_SZ,
               (const BYTE*)value.c_str(),
               dwLen);
	if( hRes !=ERROR_SUCCESS)
		cerr<<"Failed to set registry"<<endl;

}

//retrieves the path of the installation of taupin
string GetTauPinPath()
{
	char    szAppPath[MAX_PATH] = "";
	string my_path;
	::GetModuleFileNameA(NULL, szAppPath, sizeof(szAppPath) - 1);
	my_path.append(szAppPath);
	int indx=my_path.find_last_of('\\');
	if(indx==string::npos)
		return my_path;
	my_path=my_path.substr(0,indx);
	return my_path;
}

//this was suppose to spwan the process 
// but currently not used as system works pretty well and this has some path issues
//in future we might replace with this
void SpawnProfiler(char * cmd, string params )
{
	STARTUPINFO si;
    PROCESS_INFORMATION pi;
	cout<<params<<endl;
    ZeroMemory( &si, sizeof(si) );
    si.cb = sizeof(si);
    ZeroMemory( &pi, sizeof(pi) );
	WCHAR *my_args;
	//allocate a unicode string and copy the arguments 
	my_args= new WCHAR[params.length()+1];
	MultiByteToWideChar(CP_ACP, 0, (LPCSTR) params.c_str(), -1, my_args,params.length()+1);

    // Start the child process.
	cout<<params<<endl;
	if( !CreateProcess((LPCTSTR)"pin.exe",   // No module name (use command line)
        //(LPWSTR)my_args,        // Command line
		L"pin.exe -t TauPin.dll -- testapp.exe",
        NULL,           // Process handle not inheritable
        NULL,           // Thread handle not inheritable
        FALSE,          // Set handle inheritance to FALSE
        0,              // No creation flags
        NULL,           // Use parent's environment block
        NULL,           // Use parent's starting directory 
        &si,            // Pointer to STARTUPINFO structure
        &pi )           // Pointer to PROCESS_INFORMATION structure
    ) 
    {
		cout<<"CreateProcess failed:"<<GetLastError()<<endl;
        return;
    }

    // Wait until child process exits.
    WaitForSingleObject( pi.hProcess, INFINITE );

    // Close process and thread handles. 
    CloseHandle( pi.hProcess );
    CloseHandle( pi.hThread );
}

//get the pintool path  
string GetTauPin()
{
	//for now just getting the dll
	//string taupin="\""+GetTauPinPath();
	//taupin.append("\\TauPin.dll\"");
	//return taupin;
	return "TauPin.dll";
}
//get the PIN tool
string GetMyPin()
{
	//string mypin="\""+GetTauPinPath();
	//mypin.append("\\pin.exe\"");
	//return mypin;
	return "pin.exe";
}

//get all the MPI related parameters
string GetMpiArgs(int argc,char** argv )
{
	string args("");
	for(int i=0;i<argc;i++)
	{
		if(strcmp((char*)argv[i],"-n")==0)
		{
			args.append("mpiexec -np ");
			i++;
			if(argc>i)
			{
				args.append((char*)argv[i]);
				args.append(" ");
				return args;
			}
			break;
		}
	}
	return args;
}

// Get the target application parameters
string GetTargetApp(int argc, char ** argv)
{
	string myapp;
	for(int i=0;i<argc;i++)
	{
		if(strcmp((char*)argv[i],"--")==0 && i<argc-1)
		{
			myapp=argv[i+1];
			return myapp;
		}
	}
	return myapp;
}

//extragcts EXE name from the argiument parameters
string ExtractExeName(string my_app)
{	
	int indx_start= my_app.find_last_of('\\');
	if(indx_start==string::npos)
		indx_start=0;
	int indx_end = my_app.find_first_of('.',indx_start);
	if(indx_end==string::npos)
	{
		cerr<<"Give the complete EXE name like myapp.exe"<<endl;
		return "";
	}

	string app_name;
	if(indx_start==0)
		app_name= my_app.substr(0,indx_end+4);
	else
		app_name= my_app.substr(indx_start+1, (indx_end-indx_start)+5);
	//app_name.append(".exe");
	return app_name;
}

// this compose the actual rule taken by the pintool
string GetFullRule(string target_app, string cur_rule)
{
	int indx=cur_rule.find_first_of("!");
	if(indx==string::npos)
	{
		string my_rule=target_app;
		my_rule.append("!");
		my_rule.append(cur_rule);
		my_rule.append("!1");
		return my_rule;
	}
	string next_part=cur_rule.substr(indx);
	indx=next_part.find_first_of("!");
	if(indx==string::npos)
	{
		string my_rule=cur_rule;
		my_rule.append("!1");
		return my_rule;
	}
	return cur_rule;
}

//here parsing of the tool arguments done
string GetToolArgs(int argc, char** argv)
{
	//string args("pin.exe -t TauPin.dll ");
	string args=GetMyPin();
	args.append(" -t ");
	args.append(GetTauPin());
	args.append(" ");
	string target_app=GetTargetApp(argc,argv);
	target_app=ExtractExeName(target_app);
	int rule_cnt=0;
	bool found_app=false;
	if(target_app.length()==0)
	{
		target_app=ExtractExeName(argv[1]);
		if(target_app.length()==0)
			return "";
		args.append(" -r ");
		args.append(GetFullRule(target_app,"*"));
		int i=1;
		args.append(" -- ");
		while(i<argc)
			{	
				args.append((char*)argv[i]);
				args.append(" ");
				i++;
			}	
		return args;
	}

	for(int i=0;i<argc;i++)
	{
		if(strcmp((char*)argv[i],"-r")==0)
		{
			args.append(" -r ");
			i++;
			args.append(GetFullRule(target_app,argv[i]));
			args.append(" ");
			rule_cnt++;
		}
		if(strcmp((char*)argv[i],"--")==0)
		{
			found_app=true;
			if(rule_cnt==0)
			{
				args.append(" -r ");
				args.append(GetFullRule(target_app,"*"));
			}
			args.append(" --  ");
			i++;
			while(i<argc)
			{			
				args.append((char*)argv[i]);
				args.append(" ");
				i++;
			}
		}
	}

	return args;
}

//does take care of installation with -i switch
void HandleInstall(string target_path)
{
	string path_info=ReadEnv("PATH");
	path_info.append(";");
	path_info.append(target_path);
	SetEnv("PATH",path_info);
}
//takes care ofn uninstall with -u switch
void HandleUnInstall(string target_path)
{
	string path_info=ReadEnv("PATH");
	int indx=path_info.find(target_path,0);
	if(indx==string::npos)
		return;
	path_info.erase(indx,target_path.length());
	//cout<<path_info<<endl;
	SetEnv("PATH",path_info);
}
//simply for debugging
void PrintArgs(int argc, char ** argv)
{
	for(int i=0;i<argc;i++)
	{
		cout<<argv[i]<<endl;
	}
	Sleep(10000);
}

// the main entry point of the application
int main(int argc, char** argv)
{

	if(argc==1 || (argc==2 && strcmp(argv[1],"-help")==0))
	{
			Usage();
		    return -1;
	}

	if(argc==3 && strcmp(argv[1],"-i")==0)
	{
		HandleInstall(argv[2]);
		return 0;
	}

	if(argc==3 && strcmp(argv[1],"-u")==0)
	{
		HandleUnInstall(argv[2]);
		return 0;
	}
	
	string mpiarg=GetMpiArgs(argc,argv);
	string toolarg=GetToolArgs(argc,argv);

	if(toolarg.length()==0)
	{
		Usage();
		return 1;
	}

	if(mpiarg.length()>0)
	{
		mpiarg.append(" ");
	}
	mpiarg+=toolarg;
	//mpiarg="\""+mpiarg+"\"";
	cout<<mpiarg<<endl;
	system(mpiarg.c_str());
	return 0;
}

