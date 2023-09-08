//===- Hello.cpp - Example code from "Writing an LLVM Pass" ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements two versions of the LLVM "Hello World" pass described
// in docs/WritingAnLLVMPass.html
//
//===----------------------------------------------------------------------===//
//
// Information on the new pass mananger:
// https://clang.llvm.org/docs/ClangPlugins.html
// https://llvm.org/devmtg/2020-09/slides/Finkel-Changing_Everything_With_Clang_Plugins.pdf
//
//===----------------------------------------------------------------------===//
//
// Updated to work with the legacy PM (which once was called the new PM)
// working with LLVM 7 -> 13 (with -flegacy-pm) and LLVM 13 and beyond
//
//===----------------------------------------------------------------------===//

#define GLIBCXX_USE_CXX11_ABI 0

#include <fstream>
#include <regex>
#include <ctype.h>

#include "llvm/ADT/Statistic.h" //
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/Pass.h"

//#include "llvm/ADT/Triple.h"

#include <llvm/IR/DebugLoc.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <clang/Basic/SourceManager.h>
#include <llvm/Analysis/LoopInfo.h>

#if LEGACY_PM
// Need these to use Sampson's registration technique
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/IR/LegacyPassManager.h"

#else
// new pass manager
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Passes/PassBuilder.h"

#endif

#ifdef TAU_PROF_CXX
#include <cxxabi.h>
#endif

// Other passes do this, so I assume the macro is useful somewhere
#define DEBUG_TYPE "tau-profile"

#define TAU_BEGIN_INCLUDE_LIST_NAME      "BEGIN_INCLUDE_LIST"
#define TAU_END_INCLUDE_LIST_NAME        "END_INCLUDE_LIST"
#define TAU_BEGIN_EXCLUDE_LIST_NAME      "BEGIN_EXCLUDE_LIST"
#define TAU_END_EXCLUDE_LIST_NAME        "END_EXCLUDE_LIST"
#define TAU_BEGIN_FILE_INCLUDE_LIST_NAME "BEGIN_FILE_INCLUDE_LIST"
#define TAU_END_FILE_INCLUDE_LIST_NAME   "END_FILE_INCLUDE_LIST"
#define TAU_BEGIN_FILE_EXCLUDE_LIST_NAME "BEGIN_FILE_EXCLUDE_LIST"
#define TAU_END_FILE_EXCLUDE_LIST_NAME   "END_FILE_EXCLUDE_LIST"
#define TAU_BEGIN_LOOP_LIST_NAME         "BEGIN_LOOPS_LIST"
#define TAU_END_LOOP_LIST_NAME           "END_LOOPS_LIST"

#define TAU_REGEX_STAR              '#'
#define TAU_REGEX_FILE_STAR         '*'
#define TAU_REGEX_FILE_QUES         '?'

using namespace llvm;

namespace {

    // Command line options for this plugin.  These permit the user to specify what
    // functions should be instrumented and what profiling functions to call.  The
    // only real caveat is that the profiling function symbols must be present in
    // some source/object/library or compilation will fail at link-time.
    cl::opt<std::string>
        TauInputFile("tau-input-file",
                cl::desc("Specify file containing the names of functions to instrument"),
                cl::value_desc("filename"));

    cl::opt<std::string>
        TauStartFunc("tau-start-func",
                cl::desc("Specify the profiling function to call before functions of interest"),
                cl::value_desc("Function name"),
                cl::init("Tau_start"));

    cl::opt<std::string>
        TauStopFunc("tau-stop-func",
                cl::desc("Specify the profiling function to call after functions of interest"),
                cl::value_desc("Function name"),
                cl::init("Tau_stop"));

    cl::opt<std::string>
        TauRegex("tau-regex",
                cl::desc("Specify a regex to identify functions interest (case-sensitive)"),
                cl::value_desc("Regular Expression"),
                cl::init(""));

    cl::opt<std::string>
        TauIRegex("tau-iregex",
                cl::desc("Specify a regex to identify functions interest (case-insensitive)"),
                cl::value_desc("Regular Expression"),
                cl::init(""));

    cl::opt<bool>
        TauDryRun("tau-dry-run",
                cl::desc("Don't actually instrument the code, just print what would be instrumented"));

    auto TauInitFunc = "Tau_init"; // arguments to pass: argc, argv
    auto TauDestFunc = "Tau_destructor_trigger"; // arguments to pass:
    auto TauFiniFunc = "Tau_shutdown"; // arguments to pass:
    auto TauSetNodeFunc = "Tau_set_node"; // argument to pass: 0

    // Demangling technique borrowed/modified from
    // https://github.com/eklitzke/demangle/blob/master/src/demangle.cc
    static StringRef normalize_name(StringRef mangled_name) {
#ifdef TAU_PROF_CXX
        int status = 0;

        const char *str = abi::__cxa_demangle(mangled_name.begin(), 0, 0, &status);
        StringRef realname{str};

        switch (status) {
            case 0:
                break;
            case -1:
                // errs() << "FAIL: failed to allocate memory while demangling "
                //        << mangled_name << '\n';
                break;
            case -2:
                /* This case can happen in two cases:
                   - the name is not a valid name
                   - the name is main.
                   However, names from C libraries or with extern "C" are not mangled.
                   So, return anyway.
                   */
                realname = mangled_name;
                break;
            default:
                // errs() << "FAIL: couldn't demangle " << mangled_name
                //        << " for some unknown reason: " << status << '\n';
                break;
        }

        return realname;
#else
        return mangled_name;
#endif
    }

    /*!
     *  Find/declare a function taking a single `i8*` argument with a void return
     *  type suitable for making a call to in IR. This is used to get references
     *  to the TAU profiling function symbols.
     *
     * \param funcname The name of the function
     * \param ctx The LLVMContext
     * \param mdl The Module in which the function will be used
     */
#if( LLVM_VERSION_MAJOR <= 8 )
    static Constant *getVoidFunc(StringRef funcname, LLVMContext &context, Module *module)
#else
    static FunctionCallee getVoidFunc(StringRef funcname, LLVMContext &context, Module *module)
#endif // LLVM_VERSION_MAJOR <= 8
    {
        // Void return type
        Type *retTy = Type::getVoidTy(context);

        // single i8* argument type (char *)
        Type *argTy = Type::getInt8PtrTy(context);
        SmallVector<Type *, 1> paramTys{argTy};

        // Third param to `get` is `isVarArg`.  It's not documented, but might have
        // to do with variadic functions?
        FunctionType *funcTy = FunctionType::get(retTy, paramTys, false);
        return module->getOrInsertFunction(funcname, funcTy);
    }

    #if( LLVM_VERSION_MAJOR <= 8 )
    static Constant *getVoidFunc_init(StringRef funcname, LLVMContext &context, Module *module)
#else
    static FunctionCallee getVoidFunc_init(StringRef funcname, LLVMContext &context, Module *module)
#endif // LLVM_VERSION_MAJOR <= 8
    {
        // Void return type
        Type *retTy = Type::getVoidTy(context);

        // First argument is an int or i32 in LLVM (argc)
        Type *argTy0 = Type::getInt32Ty(context);

        // Second argument is an i8* pointer or i8** in LLVM (argv)
        Type* argTy1 = PointerType::getUnqual(Type::getInt8PtrTy(context));
        
        SmallVector<Type *, 2> paramTys{argTy0,argTy1};

        // Third param to `get` is `isVarArg`.  It's not documented, but might have
        // to do with variadic functions?
        FunctionType *funcTy = FunctionType::get(retTy, paramTys, false);
        return module->getOrInsertFunction(funcname, funcTy);
    }

#if( LLVM_VERSION_MAJOR <= 8 )
    static Constant *getVoidFunc_set(StringRef funcname, LLVMContext &context, Module *module)
#else
    static FunctionCallee getVoidFunc_set(StringRef funcname, LLVMContext &context, Module *module)
#endif // LLVM_VERSION_MAJOR <= 8
    {
        // Void return type
        Type *retTy = Type::getVoidTy(context);

        // Takes only one argument of type int
        Type *argTy0 = Type::getInt32Ty(context);

        SmallVector<Type *, 1> paramTys{argTy0};

        // Third param to `get` is `isVarArg`.  It's not documented, but might have
        // to do with variadic functions?
        FunctionType *funcTy = FunctionType::get(retTy, paramTys, false);
        return module->getOrInsertFunction(funcname, funcTy);
    }


#if( LLVM_VERSION_MAJOR <= 8 )
    static Constant *getVoidFunc_noargs(StringRef funcname, LLVMContext &context, Module *module)
#else
    static FunctionCallee getVoidFunc_noargs(StringRef funcname, LLVMContext &context, Module *module)
#endif // LLVM_VERSION_MAJOR <= 8
    {
        // Void return type
        Type *retTy = Type::getVoidTy(context);

        // Does not take any argument

        FunctionType *funcTy = FunctionType::get(retTy, false);
        return module->getOrInsertFunction(funcname, funcTy);
    }



/* All the common methods and attributes go in there */

    class Tools {

    public:
        using CallAndName = std::pair<CallBase *, StringRef>;

        StringSet<> funcsOfInterest;
        StringSet<> funcsExcl;
        //StringSet<> funcsOfInterestRegex;
        //StringSet<> funcsExclRegex;
        std::vector<std::regex> funcsOfInterestRegex;
        std::vector<std::regex> funcsExclRegex;
        bool verbose;
        bool callsites;
        bool definitions;
        unsigned minInstructionCount;

        StringSet<> filesIncl;
        StringSet<> filesExcl;
        // StringSet<> filesInclRegex;
        //  StringSet<> filesExclRegex;
        std::vector<std::regex> filesInclRegex;
        std::vector<std::regex> filesExclRegex;
        StringSet<> loopsIncl;
        std::vector<std::regex> loopsInclRegex;

        // basic ==> POSIX regular expression
        std::regex rex{TauRegex,
                    std::regex_constants::ECMAScript};
        std::regex irex{TauIRegex,
                    std::regex_constants::ECMAScript | std::regex_constants::icase};


        /*!
         * Given an open file, a token and two vectors, read what is coming next and
         * put it in the vector or its regex counterpart until the token has been
         * reached.
         */
        void readUntilToken( std::ifstream& file, StringSet<>& vec, std::vector<std::regex>& vecReg, const char* token ){
            std::string funcName;
            std::string s_token( token ); // used by an errs()
            bool rc = true;

            while( std::getline( file, funcName ) ){
                if( funcName.find_first_not_of(' ') != std::string::npos ) {
                    /* Exclude whitespace-only lines */

                    if( 0 == funcName.compare( token ) ){
                        return;
                    }

                    if( s_token.end() == std::find( s_token.begin(), s_token.end(), 'X' ) ){
                        if (verbose) errs() << "Include";
                    } else {
                        if (verbose) errs() << "Exclude";
                    }
                    if( s_token.end() == std::find( s_token.begin(), s_token.end(), 'F' )
                        &&  s_token.end() == std::find( s_token.begin(), s_token.end(), 'O' ) ){
                        std::regex par_o( std::string( "\\([\\s]" ) );
                        std::regex par_c( std::string( "[\\s]\\)" ) );
                        std::string s_o( std::string(  "(" ) );
                        std::string s_c( std::string(  ")" ) );
                        std::string regex_1, regex_0;
                        std::regex_replace( std::back_inserter( regex_0 ), funcName.begin(), funcName.end(), par_o, s_o );
                        std::regex_replace( std::back_inserter( regex_1 ), regex_0.begin(), regex_0.end(), par_c, s_c );
                        funcName =  std::string( regex_1 );
                        if (verbose) errs() << " function: " << funcName;
                        /* TODO: trim whitespaces */
                    } else {
                        if( s_token.end() == std::find( s_token.begin(), s_token.end(), 'O' ) ){
                            if (verbose) errs() << " file " << funcName;
                        } else {
                            // TODO regex for loops
                             if (verbose) errs() << " loop " << funcName;
                       }
                    }

                    /* The regex wildcards are not the same for filenames and function names. */

                    if( s_token.end() != std::find( s_token.begin(), s_token.end(), 'F' ) ){
                        /* This is a filename */
                        if( funcName.end() != std::find( funcName.begin(), funcName.end(), TAU_REGEX_FILE_STAR )
                                || funcName.end() != std::find( funcName.begin(), funcName.end(), TAU_REGEX_FILE_QUES )){

                            std::regex q( std::string( "[*]" ) );
                            std::string q_reg( std::string(  "(.*)" ) );
                            std::string regex_1;

                            std::regex_replace( std::back_inserter( regex_1 ), funcName.begin(), funcName.end(), q, q_reg );

                            std::regex q2( std::string( "[?]" ) );
                            std::string q2_reg( std::string(  "(.?)" ) );
                            std::string regex_2;

                            std::regex_replace( std::back_inserter( regex_2 ), regex_1.begin(), regex_1.end(), q2, q2_reg );

                            //errs()<< "regex filename: " << regex_2 << "\n";

                            vecReg.push_back( std::regex( regex_2 ) );
                            if (verbose) errs() << " (regex)";

                        } else {
                            vec.insert( funcName );
                        }
                    } else {
                        /* This is a function name */
                        if( funcName.end() != std::find( funcName.begin(), funcName.end(), TAU_REGEX_STAR ) ) {
                            /* We need to pre-process this regex: escape the parenthesis */
                            std::regex par_o( std::string( "\\(" ) );
                            std::regex par_c( std::string( "\\)" ) );
                            std::string s_o( std::string(  "\\(" ) );
                            std::string s_c( std::string(  "\\)" ) );
                            std::string regex_1, regex_0;
                            std::regex_replace( std::back_inserter( regex_0 ), funcName.begin(), funcName.end(), par_o, s_o );
                            std::regex_replace( std::back_inserter( regex_1 ), regex_0.begin(), regex_0.end(), par_c, s_c );

                            /* Escape the stars (pointers) */
                            std::regex r_s( std::string( "[\\*]" ) );
                            std::string star( std::string(  "\\*" ) );
                            std::string regex_2;
                            std::regex_replace( std::back_inserter( regex_2 ), regex_1.begin(), regex_1.end(), r_s, star );

                            /* Wildcard: replace the # by stars */
                            std::regex cross( std::string( "[#]" ) );
                            std::string wildcard( std::string(  ".*" ) );
                            std::string regex_3;
                            std::regex_replace( std::back_inserter( regex_3 ), regex_2.begin(), regex_2.end(), cross, wildcard );

                            vecReg.push_back( std::regex( regex_3 ) );
                            if (verbose) errs() << " (regex)";
                        } else {
                            vec.insert( funcName );
                        }
                    }
                    if (verbose) errs() << "\n";
                }
            }

            if( rc ){
                errs() << "Error while reading the instrumentation list in the input file. Did you close it with " << token << "?\n";
            }

        }


        /*!
         *  Given an open file, read each line as the name of a function that should
         *  be instrumented.  This modifies the class member funcsOfInterest to hold
         *  strings from the file.
         */
        void loadFunctionsFromFile(std::ifstream & file) {
            std::string funcName;

            /* This will be necessary as long as we don't have pattern matching in C++ */
            enum TokenValues { wrong_token, begin_func_include, begin_func_exclude,
                               begin_file_include, begin_file_exclude, begin_loop };

            static std::map<std::string, TokenValues> s_mapTokenValues;

            s_mapTokenValues[ TAU_BEGIN_INCLUDE_LIST_NAME ] = begin_func_include;
            s_mapTokenValues[ TAU_BEGIN_EXCLUDE_LIST_NAME ] = begin_func_exclude;
            s_mapTokenValues[ TAU_BEGIN_FILE_INCLUDE_LIST_NAME ] = begin_file_include;
            s_mapTokenValues[ TAU_BEGIN_FILE_EXCLUDE_LIST_NAME ] = begin_file_exclude;
            s_mapTokenValues[ TAU_BEGIN_LOOP_LIST_NAME ] = begin_loop;

            while(std::getline(file, funcName)) {
                if( funcName.find_first_not_of(' ') != std::string::npos ) {
                    /* Exclude whitespace-only lines */

                    switch( s_mapTokenValues[ funcName ]){
                        case begin_func_include:
                            if (verbose) errs() << "Included functions: \n";
                            readUntilToken( file, funcsOfInterest, funcsOfInterestRegex, TAU_END_INCLUDE_LIST_NAME );
                            break;

                        case begin_func_exclude:
                            // errs() << "Excluded functions: \n"<< s_mapTokenValues[ funcName ] << "\n";
                            readUntilToken( file, funcsExcl, funcsExclRegex, TAU_END_EXCLUDE_LIST_NAME );
                            break;

                        case begin_file_include:
                            if (verbose) errs() << "Included files: \n";
                            readUntilToken( file, filesIncl, filesInclRegex, TAU_END_FILE_INCLUDE_LIST_NAME );
                            break;

                        case begin_file_exclude:
                            if (verbose) errs() << "Excluded files: \n";
                            readUntilToken( file, filesExcl, filesExclRegex, TAU_END_FILE_EXCLUDE_LIST_NAME );
                            break;

                        case begin_loop:
                            if (verbose) errs() << "Included loops: \n";
                            readUntilToken( file, loopsIncl, loopsInclRegex, TAU_END_LOOP_LIST_NAME );
                            break;

                        default:
                            errs() << "Wrong syntax: the lists must be between ";
                            errs() << TAU_BEGIN_INCLUDE_LIST_NAME << " and " << TAU_END_INCLUDE_LIST_NAME;
                            errs() << " for the list of functions to instrument and ";
                            errs() << TAU_BEGIN_EXCLUDE_LIST_NAME << " and " << TAU_END_EXCLUDE_LIST_NAME;
                            errs() << " for the list of functions to exclude.\n";
                            break;
                    }
                }
            }
        }

        /* Get the call's location.
           NB: this is obtained from debugging information, and therefore needs
           -g to be acessible.
        */
        std::string getFilename( Function& call ){
            std::string filename;

            DISubprogram* s = call.getSubprogram();
            if( s != nullptr ){
                StringRef theFile = s->getFilename();
                //           StringRef theDir = s->getDirectory();
                //filename = theDir.str() + "/" + theFile.str();
                filename = theFile.str();
            } else {
                if( 0 ==  call.getInstructionCount() ){
                    /* We do not have the list of instruction of some functions,
                       such as functions defined in external libraries. In this case,
                       just return an empty string. The used might or might not want
                       to instrument them, so it is up to them to decide, but not based
                       on information on the file the function is defined in. */
                    return "";
                }
                auto pi = inst_begin( &call );
                Instruction* instruction = &*pi;
                if( nullptr == instruction ){
                    /* This should already be handled by the test above, but this is safer */
                    return "";
                }
               const llvm::DebugLoc &debugInfo = instruction->getDebugLoc();

                if( NULL != debugInfo ){ /* if compiled with -g */
                    //  filename = debugInfo->getDirectory().str() + "/" + debugInfo->getFilename().str();
                    filename = debugInfo->getFilename().str();
                } else {
                    filename = call.getParent()->getSourceFileName();
                }
            }
            return filename;
        }

         std::string getFilename( const Loop *loop ){
            std::string filename;

            const llvm::DebugLoc &debugInfo = loop->getStartLoc();
            if( NULL != debugInfo ){ /* if compiled with -g */
                //  filename = debugInfo->getDirectory().str() + "/" + debugInfo->getFilename().str();
                filename = debugInfo->getFilename().str();
            } else {
                ///filename = loop.getParent()->getSourceFileName();
            }
            
            return filename;
        }

       std::string getLineAndCol( Function& call ){
            std::string loc;
            /* Not as precise: gives the line and column numbers of the first instruction */
            /*
            auto pi = inst_begin( &call );
            Instruction* instruction = &*pi;
            const llvm::DebugLoc &debugInfo = instruction->getDebugLoc();

            if( NULL != debugInfo ) {
                loc = std::to_string( debugInfo.getLine() ) + "," + std::to_string( debugInfo.getCol() );
            }
             */
            DISubprogram* s = call.getSubprogram();
            if( NULL != s ){
                int line = s->getLine();
                loc = std::to_string( line );
            } else {
                loc = "0,0";
            }

            return loc;
        }

        std::string getLineAndCol( CallBase* call ){
            std::string loc;

            const DebugLoc &location = call->getDebugLoc();
            if( location ) {
                int line = location.getLine();
                loc = std::to_string( line );
            } else {
                // No location metadata available
                loc = "0,0";
            }

            return loc;
        }
        
        std::string getLineAndCol( const Loop *loop ){
            std::string loc;

            const llvm::DebugLoc &location = loop->getStartLoc();
            if( location ) {
                int line = location.getLine();
                loc = std::to_string( line );
            } else {
                // No location metadata available
                loc = "0,0";
            }

            return loc;
        }

        bool includeFunction(StringRef prettycallName, std::string& filename) {
            if (verbose) errs() << "Func: '" << prettycallName.str() << "' : ";
            /* This big test was explanded for readability */
            bool instrumentHere = false;
            /* Are we including or excluding some files? */
            if( (filesIncl.size() + filesInclRegex.size() + filesExcl.size() + filesExclRegex.size() == 0 ) ){
                instrumentHere = true;
            } else {
                /* Yes: are we in a file where we are instrumenting? */
                // did not specify a list of files to instrument -> instrument them all, except the excluded ones
                if( ( ( filesIncl.size() + filesInclRegex.size() == 0)
                            || ( filesIncl.count( filename ) > 0
                                || regexFits( filename, filesInclRegex ) ) )
                        && !( filesExcl.count( filename ) > 0
                            || regexFits( filename, filesExclRegex ) ) ){
                    instrumentHere = true;
                }
            }
            // Not doing this file?  Skip it.
            if (!instrumentHere) {
                return false;
            }
            // Nothing included/excluded?  Instrument it.
            if (funcsOfInterest.size() == 0 &&
                    funcsExcl.size() == 0 &&
                    funcsOfInterestRegex.size() == 0 &&
                    funcsExclRegex.size() == 0) {
                if (verbose) errs() << "include everyone, therefore include\n";
                return true;
            }

            // simple cases...
            if (verbose) errs() << "Looking up...\n";

            // Is this function explicitly included?  Instrument it.
            if (funcsOfInterest.size() > 0 && funcsOfInterest.count( prettycallName ) > 0) {
                if (verbose) errs() << "include Match\n";
                return true;
            }
            // Is this function explicitly excluded?  Skip it.
            if (funcsExcl.size() > 0 && funcsExcl.count( prettycallName ) > 0) {
                if (verbose) errs() << "exclude Match\n";
                return false;
            }

            // Ok, now it gets complicated...

            if (funcsExclRegex.size() > 0 && regexFits( prettycallName, funcsExclRegex, true )) {
                if (verbose) errs() << "excluded as regex!\n";
                return false;
            }

            if (funcsOfInterestRegex.size() > 0 && regexFits ( prettycallName, funcsOfInterestRegex, true )) {
                if (verbose) errs() << "included as regex!\n";
                return true;
            }

            if (funcsOfInterest.size() == 0 && funcsOfInterestRegex.size() == 0) {
                // finally, if we didn't match to anything, and there weren't any inclusion
                // conditions, return true by default.
                if (verbose) errs() << "Default - Returning true!\n";
                return true;
            }
            return false;
        }

        /*!
         *  Inspect the given CallInst and, if it should be profiled, add it
         *  and its recognized name the given vector.
         *
         * \param call The CallInst to inspect
         * \param calls Vector to add to, if the CallInst should be profiled
         */

        bool maybeSaveForProfiling( Function& fn ) {
            StringRef callName = fn.getName();
            errs() << callName << "\n";
            std::string filename = getFilename( fn );
            StringRef prettycallName = normalize_name(callName);
            unsigned instructionCount = fn.getInstructionCount();
            auto *module = fn.getParent();
            const std::string triple = module->getTargetTriple();
            bool is_host_func = triple.compare(std::string("amdgcn-amd-amdhsa")); // returns 0 if it matches
            // Compare similarly for other GPUs. If it matches, do not instrument it.

            // if not running on the CPU, skip it
            if (is_host_func == false){
		    if (verbose) errs() << " Not host function, skip\n";
		    return false;
	    }
            // if the function name is empty, skip it
            if( prettycallName == "" ){
		    if (verbose) errs() << " Function name is empty, skip\n";
		    return false;
	    }
            // if the instruction count is small, skip it
            if( instructionCount == 0){
		    if (verbose) errs() << " Empty or function from a library, skip\n";
		    return false;
	    }

            if( instructionCount < minInstructionCount && prettycallName.compare("main") != 0) {
                if (verbose) errs() << "Skipping, only " << instructionCount << " instructions, minimum set to " << minInstructionCount << ".\n";
                return false;
            }

            if (verbose) errs() << "Func: '" << prettycallName.str() << "' : ";
            if (includeFunction(prettycallName, filename)) {
                return true;
            }
            return false;
        }

        bool maybeSaveForProfiling(CallBase *call, std::vector<CallAndName> &calls) {
            /* In a previous implementation, we entered the function from its call somewhere in the program.
               Now we are wrapping the call with the instrumentaiton instructions.  */
            auto fn = call->getCalledFunction();
            if( fn && maybeSaveForProfiling(*fn)) {
                StringRef callName = fn->getName();
                StringRef prettycallName = normalize_name(callName);
                calls.push_back( { call, prettycallName } );
                return true;
            }
            // all other cases, return false
            //if (verbose) errs() << "Default - Returning false!\n";
            return false;
        }

        /* Do we want to instrument this loop?
         */

        bool maybeInstrument( const Loop* loop ){
            std::string loc = getFilename( loop ) +  ":" + getLineAndCol( loop );
            if( verbose ) errs() << "Loop: " << loc << "\n";
            if( loopsIncl.size() + loopsInclRegex.size() == 0 )
                return false;
            if( loopsIncl.count( loc ) > 0 )
                return true;
            if( regexFits( loc, loopsInclRegex ) )
                return true;
            return false;
        }
        
        /*!
         * This function determines if the current function name (parameter name)
         * matches a regular expression. Regular expressions can be passed either
         * on the command-line (historical behavior) or in the input file. The latter
         * use a specific wildcard.
         */
        bool regexFits( const StringRef & name, std::vector<std::regex>& regexList, bool cli = false ) {
            /* Regex coming from the command-line */
            bool match = false, imatch = false;
            if( cli ){
                if(!TauRegex.empty()) match = std::regex_search(name.str(), rex);
                if(!TauIRegex.empty()) imatch = std::regex_search(name.str(), irex);
            }

            if( match || imatch ) return true;

            for( auto& r : regexList ){
                match  = std::regex_match( name.str(), r );
                if( match ) return true;
            }

            return false;
        }

        /*!
         *  Add instrumentation to the CallInsts in the given vector, using the
         *  given function for context.
         *
         * \param calls vector of calls to add profiling to
         * \param func Function in which the calls were found
         * \return False if no new instructions were added (only when calls is empty),
         *  True otherwise
         */
        bool addInstrumentation( std::vector<CallAndName> &calls, Function &func, bool instrumentSelf ) {

            // Declare and get handles to the runtime profiling functions
            auto &context = func.getContext();
            auto *module = func.getParent();
            StringRef prettyname = normalize_name(func.getName());
            std::string filename = getFilename( func ); // the file where the calls we are instrumenting are happening
#if( LLVM_VERSION_MAJOR <= 8 )
            Constant *onCallFunc = getVoidFunc(TauStartFunc, context, module);
            Constant *onRetFunc = getVoidFunc(TauStopFunc, context, module);
            Constant *onFiniFunc = getVoidFunc_noargs(TauFiniFunc, context, module);
            Constant *onDestFunc = getVoidFunc_noargs(TauDestFunc, context, module);
#else
            FunctionCallee onCallFunc = getVoidFunc(TauStartFunc, context, module);
            FunctionCallee onRetFunc = getVoidFunc(TauStopFunc, context, module);
            FunctionCallee onFiniFunc = getVoidFunc_noargs(TauFiniFunc, context, module);
            FunctionCallee onDestFunc = getVoidFunc_noargs(TauDestFunc, context, module);
#endif // LLVM_VERSION_MAJOR <= 8

            std::string shorter(prettyname);
            // to keep formatting pretty, trim any long names
            if (shorter.size() > 80) {
                shorter.resize(77);
                shorter.resize(80, '.');
            }
            if (verbose) errs() << "Instrumenting " << shorter << " and " << calls.size() << " callsites.\n";
            bool mutated = false; // TODO

            /* Add TAU init in main */

            if( 0 == prettyname.compare( "main" ) /*||  0 == prettyname.compare( "_QQmain" ) TODO fix the Fortran call */){
                if (verbose) errs() << "\tmain function: adding init\n";
                auto initfun = getVoidFunc_init( TauInitFunc, context, module );
                auto setnodefun = getVoidFunc_set( TauSetNodeFunc, context, module );

                auto beg = inst_begin( &func );
                Instruction* b = &*beg;
                IRBuilder<> b4( b );

                /* TauInitFunc takes two arguments: argc and argv */
                SmallVector<Value *, 2> mainArgsVect;

                if( func.args().empty() ){
                    /* Fortran uses command_argument_count() and get_command_argument() to access the command-line parameters */
                    /*    mainArgsVect.push_back( 0 );
                        mainArgsVect.push_back( nullptr );
                    */

                } else {
                    for( Argument &arg : func.args() ){
                        mainArgsVect.push_back( &arg );
                    }
                }
                if( !func.args().empty() ){
                    b4.CreateCall( initfun, mainArgsVect );
                }
                
                /* TauSetNodeFunc takes one argument: 0 */

                Value* z = ConstantInt::get( context, llvm::APInt( 32, 0, false ) );
                SmallVector<Value *, 1> zero{ z };
                b4.CreateCall( setnodefun, zero );

                mutated = true;
            }

            /* Add regular TAU calls for callsites, if requested */
            for (auto &pair : calls) {
                auto *op = pair.first;
                auto calleeName = pair.second;
                std::string location( "[{" + filename + "} {" +  getLineAndCol( op ) + "}]" );

                if (verbose) errs() << "wrapping: " << calleeName << '\n';

                IRBuilder<> builder(op);

                // This is the recommended way of creating a string constant (to be used
                // as an argument to runtime functions)
                Value *strArg = builder.CreateGlobalStringPtr( ( calleeName + " " + location ).str() );
                SmallVector<Value *, 1> args{strArg};
                builder.CreateCall( onCallFunc, args );

                // Set insert point to just after the CallInst
                /* next non-debug instruction */
                /* We can't just use  ++builder.GetInsertPoint() because of InvokeIns */
                auto next = op->getNextNonDebugInstruction( /* true*/ );
                if( next == nullptr ){ // we have reached the end of the bb
                    BasicBlock::iterator it( op );
                    next = &(*(it++));
                }
                IRBuilder<> builder2( next );
                builder2.CreateCall(onRetFunc, args);

                mutated = true;
            }

            /* Add instrumentation for the function definition, if requested */
            if (instrumentSelf) {
                // Insert instrumentation before the first instruction
                auto pi = inst_begin( &func );
                Instruction* i = &*pi;
                /* Here we can have a null pointer if the function is empty.
                   By default, this should not happen because the default threshold for the instruction
                   count is > 0. But if we set it to 0, pi is nullptr.
                   In this case, do not do what is next because it does not make any sense
                   to instrument here. */
                if( nullptr != i ){
                    IRBuilder<> before( i );
                    std::string location( "[{" + filename + "} {" +  getLineAndCol( func ) + "}]" );

                    // This is the recommended way of creating a string constant (to be used
                    // as an argument to runtime functions)
                    Value *strArg = before.CreateGlobalStringPtr( ( prettyname + " " + location ).str() );
                    SmallVector<Value *, 1> args{strArg};
                    before.CreateCall( onCallFunc, args );
                    mutated = true;

                    // We need to find all the exit points for this function

                    for( inst_iterator I = inst_begin( func ), E = inst_end( func ); I != E; ++I){
                        Instruction* e = &*I;
                        if( isa<ReturnInst>( e ) ) {
                            IRBuilder<> fina( e );
                            if( 0 == prettyname.compare( "main" ) ||  0 == prettyname.compare( "_QQmain" ) ){
                                fina.CreateCall( onDestFunc );
                                fina.CreateCall( onFiniFunc );
                           } else {
                                fina.CreateCall( onRetFunc, args );
                            }
                        }
                    }
                }
            }
            return mutated;
        }

        /* Instrument a loop */
        
#if( LLVM_VERSION_MAJOR > 8 )
        bool addInstrumentation( const Loop* loop ){
            bool mutated = false;

            std::string loopname = "LOOP " + getFilename( loop ) +  ":" + getLineAndCol( loop );
            if( verbose ) errs() << "Instrumenting " << loopname << "\n";

            /* Find the entrance of this loop */

            BasicBlock* const* enterit = loop->block_begin();
            BasicBlock* enter = *enterit;
            
            auto function = enter->getParent();
            auto &context = function->getContext();
            auto *module = enter->getModule();

            FunctionCallee onLoopEnter = getVoidFunc(TauStartFunc, context, module);
            FunctionCallee onLoopExit = getVoidFunc(TauStopFunc, context, module);

            /* We don't want to insert a timer ar the beginning of the first block, but before that */

            for( BasicBlock *pred : predecessors( enter ) ) {
                if( !loop->contains( &*pred ) ){
                    Instruction* last = pred->getTerminator();

                    Instruction* i = &*last;                
                    IRBuilder<> before( i );

                    Value *strArg = before.CreateGlobalStringPtr( loopname );
                    SmallVector<Value *, 1> args{strArg};
                    before.CreateCall( onLoopEnter, args );

                    mutated = true;
                }
            }
            
            /* Find all the exits */

            SmallVector<BasicBlock *> exitblocks;
            loop->getExitBlocks( exitblocks );
            
            for( auto ex : exitblocks ){

                /* Find the last instruction of the block */

                Instruction* last = ex->getTerminator();
                
                /* We want to instrument *before* this instruction (the bb ends with an explicit br) */

                Instruction* i = &*last;                
                IRBuilder<> before2( i );

                Value *strArg = before2.CreateGlobalStringPtr( loopname );
                SmallVector<Value *, 1> args{strArg};
                before2.CreateCall( onLoopExit, args );

                mutated = true;
            }

            return mutated;
        }
#endif
        
    };

    /*!
     * The instrumentation pass.
     */
#if LEGACY_PM
    struct Instrument : public FunctionPass, public Tools
#else
    class Instrument :  public PassInfoMixin<Instrument>, public Tools
#endif
    {
        static char ID; // Pass identification, replacement for typeid
        public:
#if LEGACY_PM
        Instrument() : FunctionPass(ID)
#else
        Instrument()
#endif
        {
            // by default, no verbose output
            char *do_verbose = getenv("TAU_COMPILER_VERBOSE");
            verbose = parse_bool(do_verbose, false);
            // by default, do not instrument callsites
            char *do_callsites = getenv("TAU_COMPILER_CALLSITES");
            callsites = parse_bool(do_callsites, false);
            if (verbose) errs() << "TAU_COMPILER_CALLSITES set to "
                << (callsites ? "true" : "false") << "\n";
            // by default, DO instrument definitions
            char *do_definitions = getenv("TAU_COMPILER_DEFINITIONS");
            definitions = parse_bool(do_definitions, true);
            if (verbose) errs() << "TAU_COMPILER_DEFINITIONS set to "
                << (definitions ? "true" : "false") << "\n";
            char *mic = getenv("TAU_COMPILER_MIN_INSTRUCTION_COUNT");
            minInstructionCount = parse_unsigned(mic, 50);
            if (verbose) errs() << "TAU_COMPILER_MIN_INSTRUCTION_COUNT set to "
                << minInstructionCount << "\n";

            /* Add some important regular expressions to the exclusion set of regular expressions */
            // Anything from the standard library
            funcsExclRegex.push_back( std::regex( "[^(]+ std::.*" ) );
            funcsExclRegex.push_back( std::regex( "decltype(.*) std::.*" ) );
            funcsExclRegex.push_back( std::regex( "decltype (.*) std::.*" ) );
            funcsExclRegex.push_back( std::regex( "std::.*" ) );
            // Anything from the C++ internals
            funcsExclRegex.push_back( std::regex( "[^(]+ __cxx.*" ) );
            funcsExclRegex.push_back( std::regex( "__cxx.*" ) );
            // Anything from the GNU C++ internals
            funcsExclRegex.push_back( std::regex( "[^(]+ __gnu_cxx.*" ) );
            funcsExclRegex.push_back( std::regex( "__gnu_cxx.*" ) );
            // Anything from the OpenMP expansion/outlining
            funcsExclRegex.push_back( std::regex( "[^(]+omp_outlined.*" ) );
            funcsExclRegex.push_back( std::regex( "\\.omp_outlined.*" ) );
            funcsExclRegex.push_back( std::regex( "[^(]+omp_outlined\\._debug__.*" ) );
            funcsExclRegex.push_back( std::regex( "\\.omp_outlined\\._debug__.*" ) );
            // Anything from the LLVM internals
            funcsExclRegex.push_back( std::regex( "[^(]+ llvm::.*" ) );
            funcsExclRegex.push_back( std::regex( "decltype(.*) llvm::.*" ) );
            funcsExclRegex.push_back( std::regex( "decltype (.*) llvm::.*" ) );
            funcsExclRegex.push_back( std::regex( "llvm::.*" ) );
            // Anything that couldn't be demangled
            funcsExclRegex.push_back( std::regex( "_Z.*" ) );
            // Other things we've encountered that aren't interesting
            funcsExclRegex.push_back( std::regex( "__gthread_active_p.*" ) );
            funcsExclRegex.push_back( std::regex( "_GLOBAL__sub_.*" ) );
	    //rocm functions that should not be measured
            funcsExclRegex.push_back( std::regex( "__hip_module.*" ) );
            funcsExclRegex.push_back( std::regex( "__hip_register_globals.*") );
            funcsExclRegex.push_back( std::regex( "__device_stub__.*" ) );


            // Let's not instrument TAU stuff. They should normally be eliminated because of their instruction count anyway
            std::string taufuncname = TauStartFunc + '*';
            funcsExclRegex.push_back( std::regex( taufuncname ) );
            taufuncname = TauStopFunc + '*';
            funcsExclRegex.push_back( std::regex( taufuncname ) );
            taufuncname = "Tau_init*";
            funcsExclRegex.push_back( std::regex( taufuncname ) );
            taufuncname = "Tau_shutdown*";
            funcsExclRegex.push_back( std::regex( taufuncname ) );
            taufuncname = "Tau_set_node*";
            funcsExclRegex.push_back( std::regex( taufuncname ) );

            if(!TauInputFile.empty()) {
                std::ifstream ifile{TauInputFile};
                if( !ifile ){
                    errs() << "Could not open input file\n";
                    return;
                }
                loadFunctionsFromFile(ifile);
            } else {
                if (verbose) errs() << "Checking selective instrumentation file specified in env. variable TAU_COMPILER_SELECT_FILE\n";
                char *fname = getenv("TAU_COMPILER_SELECT_FILE");
                if (fname) {
                    if (verbose) errs() << "TAU_COMPILER_SELECT_FILE = "<<fname<<"\n";
                    std::ifstream ifile{fname};
                    if( !ifile ){
                        errs() << "Could not open input file: " << fname<<"\n";
                        return;
                    }
                    loadFunctionsFromFile(ifile);
                }
		else{
		   if (verbose) errs() << "No selective instrumentation file specified\n";
		}
            }
        }
        ~Instrument() {
            if (verbose) errs() << "Done instrumentation pass.\n";
        }
        /*********************************************************************
         * Parse a boolean value
         ********************************************************************/
        bool parse_bool(const char *str, bool default_value = false) {
            if (str == NULL) {
                return default_value;
            }
            static char strbuf[129];
            char *ptr = strbuf;
            strncpy(strbuf, str, 128);
            while (*ptr) {
                *ptr = tolower(*ptr);
                ptr++;
            }
            if (strcmp(strbuf, "yes") == 0  ||
                    strcmp(strbuf, "true") == 0 ||
                    strcmp(strbuf, "on") == 0 ||
                    strcmp(strbuf, "1") == 0) {
                return true;
            }
            return false;
        }
        /*********************************************************************
         * Parse an integer value
         ********************************************************************/
        unsigned parse_unsigned(const char *str, unsigned default_value = 0UL) {
            if (str == NULL) {
                return default_value;
            }
            int tmp = std::stoul(std::string(str));
            return tmp;
        }

        /*!
         *  The FunctionPass interface method, called on each function produced from
         *  the original source.
         */
#if LEGACY_PM
        bool runOnFunction(Function &func) override {
            std::vector<CallAndName> calls; // too big for a SmallVector
            bool instru = false;

            if (callsites) {
                if (verbose) {
                    StringRef prettyname = normalize_name(func.getName());
                    if(prettyname.empty()) prettyname = func.getName();
                    std::string shorter(prettyname);
                    // to keep formatting pretty, trim any long names
                    if (shorter.size() > 80) {
                        shorter.resize(77);
                        shorter.resize(80, '.');
                    }
                    errs() << "Checking calls from: " << shorter << '\n';
                }
                for(BasicBlock &bb: func){
                    for(Instruction &i: bb){

                        if (auto *op = dyn_cast<CallInst>(&i)) {
                            instru |= maybeSaveForProfiling( op, calls );
                        }
                        if (auto *op = dyn_cast<InvokeInst>(&i)) {
                            instru |= maybeSaveForProfiling( op, calls );
                        }
                    }
                }
            }

            if( TauDryRun ) {
                auto pretty_name = normalize_name(func.getName());
                if(pretty_name.empty()) pretty_name = func.getName();

                errs() << "In function " << pretty_name
                    << "\nThe following would be instrumented:\n";
                for (auto &pair : calls) {
                    errs() <<  pair.second << '\n';
                }
                errs() << '\n';
                return false; // Dry run does not modify anything
            }
            bool instrumentSelf = definitions & maybeSaveForProfiling(func);
            if( instru || instrumentSelf){
                return addInstrumentation( calls, func, instrumentSelf );
            } else {
                return false;
            }
        }
#else
        bool runOnFunction( Function& func ){
            std::vector<CallAndName> calls; // too big for a SmallVector
            bool instru = false;
            if (callsites) {
                if (verbose) {
                    StringRef prettyname = normalize_name(func.getName());
                    std::string shorter(prettyname);
                    // to keep formatting pretty, trim any long names
                    if (shorter.size() > 80) {
                        shorter.resize(77);
                        shorter.resize(80, '.');
                    }
                    errs() << "Checking calls from: " << shorter << '\n';
                }
                for(BasicBlock &bb: func){
                    for(Instruction &i: bb){

                        if( isa<CallInst>( &i ) ) {
                            if( auto* op = dyn_cast<CallInst>( &i ) ){
                                instru |= maybeSaveForProfiling( op, calls );
                            }
                        }

                        if( isa<InvokeInst>( &i )) {
                            if( auto* op = dyn_cast<InvokeInst>( &i ) ){
                                instru |= maybeSaveForProfiling( op, calls );
                            }
                        }
                    }
                }
            }
  
            if( TauDryRun && instru ){
                auto pretty_name = normalize_name(func.getName());
                if(pretty_name.empty()) pretty_name = func.getName();

                errs() << "In function " << pretty_name
                    << "\nThe following would be instrumented:\n";
                for (auto &pair : calls) {
                    errs() <<  pair.second << '\n';
                }
                errs() << '\n';
                return false; // Dry run does not modify anything
            }
            bool instrumentSelf = definitions & maybeSaveForProfiling(func);
            if( instru || instrumentSelf){
                return addInstrumentation( calls, func, instrumentSelf );
            } else {
                return false;
            }
        }

#if( LLVM_VERSION_MAJOR > 8 )
        bool runOnloop( const Loop* loop ){
            bool instru;
            instru = maybeInstrument( loop );
            if( verbose ) errs() << "Instrument " << getFilename( loop ) <<  ":" << getLineAndCol( loop ) << "? " << instru << "\n";
            if( TauDryRun && instru ){
                std::string loc = getFilename( loop ) +  ":" + getLineAndCol( loop );
                errs() << "Loop " << loc << " would be instrumented\n";
                return false;
            }

            if( instru ){
                addInstrumentation( loop );
                return true;
            } else {
                return false;
            }            
        }
#endif

        PreservedAnalyses run(Module &module, ModuleAnalysisManager &mam ) {
            bool modified = false;
            
            auto &fam = mam.getResult<FunctionAnalysisManagerModuleProxy>( module ).getManager();

            for (auto &func : module.getFunctionList()) {

                /* Functions */
                modified |= runOnFunction( func );

#if( LLVM_VERSION_MAJOR > 8 )
                /* Loops */
                if( 0 !=  func.getInstructionCount() ){
                    LoopInfo &li = fam.getResult<LoopAnalysis>( func );
                    SmallVector<Loop *, 16> PreorderLoops = li.getLoopsInPreorder();
                    for( auto& loop : PreorderLoops ){
                        modified |= runOnloop( loop );
                   }
                }
#endif
            }

            if( modified ){
                return PreservedAnalyses::none();
            } else {
                return PreservedAnalyses::all();
            }
        }

        static bool isRequired() { return true; }
#endif

    }; // end class Instrument
}

#if LEGACY_PM      /* Legacy pass manager */
char Instrument::ID = 0;

static RegisterPass<Instrument> X("TAU", "TAU Profiling", false, false);

// Automatically enable the pass.
// http://adriansampson.net/blog/clangpass.html
static void registerInstrumentPass(const PassManagerBuilder &,
        legacy::PassManagerBase &PM) {
    PM.add(new Instrument());
}
static RegisterStandardPasses
RegisterMyPass(PassManagerBuilder::EP_EarlyAsPossible, registerInstrumentPass);

#else  /* New pass manager */

llvm::PassPluginLibraryInfo getInstrumentPluginInfo() {
    return {LLVM_PLUGIN_API_VERSION, "TAU", LLVM_VERSION_STRING, []( PassBuilder &PB ) {
        PB.registerPipelineStartEPCallback( [&](ModulePassManager &MPM, OptimizationLevel Level ) {
                MPM.addPass( Instrument() );
                });
    }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
    return getInstrumentPluginInfo();
}

#endif
