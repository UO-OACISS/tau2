#ifndef FD_frmMain_h_
#define FD_frmMain_h_
/* Header file generated with fdesign. */



#include "forms.h"
#include <stdlib.h>

#define MAXSTRINGLENGTH 255
#define TRUE  1
#define FALSE 0
#define MSECS_TOCHECKFORDATA 5000

#include <string.h>
#include <stdio.h>
#include <iostream.h>
#include <sys/types.h>
#include <sys/dir.h>

// include appropriate string libs
#if (defined(POOMA_KAI) || defined (TAU_STDCXXLIB))
#include <string>
using std::string;
#include <vector>
using std::vector;
#include <map>
using std::map;
#else
#define __BOOL_DEFINED 
#include "Profile/bstring.h"
#include <vector.h>
#include <map.h>
#endif /* POOMA_KAI */




/**** Callback routines ****/

extern void big_button_cb(FL_OBJECT *, long);
extern void hsbScroll_cb(FL_OBJECT *, long);
extern void vsbScroll_cb(FL_OBJECT *, long);
extern void resChanger_cb(FL_OBJECT*, long);
extern void timeout_callback(int, void*);

/**** Event Handlers ****/
extern int canvas_buttonPressHandler(FL_OBJECT*, Window, int, int,
                                     XEvent*, void *);
extern int NCTCanvas_exposeHandler(FL_OBJECT*, Window, int, int, XEvent*, 
                                   void *);
extern int funcLegend_exposeHandler(FL_OBJECT*, Window, int, int, XEvent*, 
                                    void *);
extern int funcInstance_exposeHandler(FL_OBJECT*, Window, int, int, XEvent*, 
                                      void *);


/**** Forms and Objects ****/

typedef struct {
	FL_FORM *frmMain;
	void *vdata;
	long ldata;
	FL_OBJECT *big_button;
} FD_frmMain;

extern FD_frmMain * create_form_frmMain(void);

// macros
#ifdef DEBUG
  #define DEBUG_PRINT(s) cout << s << endl
#else
  #define DEBUG_PRINT(s)
#endif 

#ifndef MIN
  #define MIN(a, b) a < b ? a : b
#endif

#ifndef MAX
  #define MAX(a, b) a > b ? a : b
#endif


/*---------------------------------------------------------------------
|
|  class FunctionInstance
|  ======================
|
*/
class FunctionInstance
{
  private:

    string *nameAndType;
    double funcIncl;
    double funcExcl;
    double profIncl;
    double profExcl;
    long         ncalls;
    long          nsubs;
    unsigned long color;    
    int yPos;
    int yHeight;
    int visible;
    
    FL_FORM   *form;
    FL_OBJECT *canvas;
    FL_OBJECT *box;

    Font  font;


  public:
    // constructors
    FunctionInstance();
    FunctionInstance(char *, long, long, double, double, double, double);
    FunctionInstance(FunctionInstance& f);
    // destructor
    ~FunctionInstance();   
 
    void buildDisplay();
    void paint();

    // get/set  methods
    string *getNameAndType()   {return nameAndType;};
    double getFuncIncl()  {return funcIncl;};
    double getFuncExcl()  {return funcExcl;};
    double getProfIncl()  {return profIncl;};
    double getProfExcl()  {return profExcl;};
    long   getNCalls()        {return ncalls;};
    long getNSubs()           {return nsubs;};
    unsigned long getColor()  {return color;};
    int getyPos()             {return yPos;};
    int getyHeight()           {return yHeight;};
    void setyPos(int y)       {yPos = y;};
    void setyHeight(int h)    {yHeight = h;};
    void setVisible(int b)    {visible = b;};
    int  getVisible()         {return visible;};
};


class NCT;

/*----------------------------------------------------------------------
|
|  class Dump
|  ==========
|  
|  This class exists for each specific callstack dump in a callstack.?.?.?
|  file.
*/
class Dump
{
  private:
    vector<FunctionInstance*> funcVctr;
    double  dumpTime;
    int xPos;
    int xWidth;
    int visible;          // true if this dump is visible, false otherwise
    NCT *myNCT;

  public:
    // constructors
    Dump();
    Dump(double t, NCT *myNCT);
    ~Dump();
    // geters and seters
    void   setDumpTime(double t) {dumpTime = t;};
    double getDumpTime()         {return dumpTime;};
    FunctionInstance* addFunc(char *nmType, long nc, long ns, double pit,
                                double pet, double fit, double fet);
    void setxPos(int x)   {xPos = x;};
    void setxWidth(int w) {xWidth = w;};
    int getxPos()         {return xPos;};
    int getxWidth()       {return xWidth;};

    void print();
    void paint(FL_OBJECT* canvas, int x, int canvasHeight, int width, 
               double vsbValue);

    const vector<FunctionInstance*>* getFuncVctr() {return &funcVctr;};
    int   getVisible() {return visible;};
    void  setVisible(int b) {visible = b;};
};

/*----------------------------------------------------------------------
|
|  class NCT
|  =========
|
|  This class exists for each of the callstack.?.?.? files.  NCT is a TLA
|  (three letter acronym) for Node-Context-Thread.
|
*/
class NCT
{
  private:
    vector<Dump*>     dumpVctr;
    char              fname[MAXSTRINGLENGTH];
    
    //drawing area stuff
    FL_Coord   canvasWidth;  // value set in buildDisplay(), buildDisplay() called by c'tor
    FL_Coord   canvasHeight; // value set in buildDisplay(), buildDisplay() called by c'tor
    FL_Coord   dumpMargin;
    FL_Coord   dumpWidth;
  
    FL_OBJECT *canvas;
    FL_OBJECT *hsbScroll;
    FL_OBJECT *vsbScroll;
    FL_OBJECT *textLabel;
    FL_OBJECT *resChanger;
    void buildDisplay();
    Font      font;

    static int lastCanvasY;  // allocate someplace else
    static int nCanvasesDrawn;

  public:  
    NCT(char* s);
    Dump* addDump(double dTime);
    vector<Dump> getDumps();
    void readFile(); 
    void print();
    void paint();
    FL_Coord   getCanvasHeight(){return canvasHeight;};
    FL_OBJECT* getResChanger() {return resChanger;};


    const vector<Dump*>* getDumpVctr() {return &dumpVctr;};
    Font   getFont() {return font;};

};


// STL map comparison function for strings
struct ltstr
{
    bool operator()(const char* s1, const char* s2) const
    {
      return strcmp(s1, s2) < 0;
    }
};


class GlobalFunctionInfo
{
  // keep a map between function names and colors used to display that function
   

 private:

 
  // STL map between function names and their color values
  map<const char*, unsigned long, ltstr> funcDB; 



  // display stuff
  FL_FORM  *form;
  FL_OBJECT *canvas;
  FL_OBJECT *box;
  Font      font;
 
  // private functions
  unsigned long nextColor();

 public:
   unsigned long addFunc(const char *s);
 

   void buildDisplay();
   void paint();

};

class GlobalFileInfo
{
  private:
    // STL map between filenames and their NCT objects
    map<const char*, NCT *, ltstr> filetoNCTMap;
    double minTime;   

  public:
    NCT*          addFile(const char *s); 
    void          checkForFiles();
    double          getMinTime(){return minTime;};
    void          findMinTime();

};


#endif /* FD_frmMain_h_ */
 
