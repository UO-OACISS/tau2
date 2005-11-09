
/* Form definition file generated with fdesign. */


#include "csUI.h"
#include <math.h>


// constants 
#define MAXSCROLLVALUE 10.0
#define DUMPWIDTH      40
#define DUMPMARGIN     20
#define CANVASHEIGHT   270

// max size of line in callstack.?.?.? file
#define SIZE_OF_LINE 64*1024

#define FONT_NCTDISPLAY       "-*-fixed-medium-*-normal-*-8-*-*-*-*-*-*-*"
#define FONT_FUNCTIONINSTANCE "-*-fixed-*-*-*-*-14-*-*-*-*-*-*-*"
#define FONT_FUNCTIONLEGEND   "-*-fixed-*-*-*-*-14-*-*-*-*-*-*-*"

#define WHITEPIXEL 0xFFFFFF

#define NCOLORPLANES    8     // number of color planes for our visual/colormap

/*--------------------------------------------------------------------
|
| global variables
| ================
| 
*/
FD_frmMain *fd_frmMain;
Display *dpy;

GlobalFunctionInfo globalFuncInfo;
GlobalFileInfo     globalFileInfo;

Colormap   globalSharedColormap;
Visual     globalSharedVisual;     // is it safe to share a visual across
                                   // multiple windows???? Will the visual
                                   // be destroyed when window destroyed???  

int NCT::lastCanvasY    = 0;
int NCT::nCanvasesDrawn = 0;

unsigned long GlobalFunctionInfo::nextColor()
{
  // return next color from our list

  static int ncolors   = 19;
  static int currColor = 0;
  unsigned long color;
  static unsigned long colormap[] = 
  {
    0x0011FFUL,
    0xFFFF00UL,
    0x00FFCCUL,
    0x66FF00UL,
    0x0033FFUL,
    0xFFCC00UL,
    0x00FF99UL,
    0x99FF00UL,
    0x0066FFUL,
    0xFF9900UL,
    0x00FF66UL,
    0xCCFF00UL,
    0x0099FFUL,
    0xFF6600UL,
    0x00FF00UL,
    0x00CCFFUL,
    0xFF3300UL,
    0x00FFFFUL,
    0xFF0000UL 
  };
  
  // get this color value
  color = colormap[currColor];
  // set next color value
  currColor = (++currColor) % ncolors;
  return color;
};


unsigned long GlobalFunctionInfo::addFunc(const char *s)
{
  // only add the function to funcDB if we haven't seen it before

  char *str;
  map<const char *, unsigned long, ltstr>::iterator it;  // funcDB iterator
  unsigned long color;

  // allocate some new space for the string
  str = new char[strlen(s)+1]; 
  strcpy(str, s);

  // see if we have found this function before
  if ((it = funcDB.find((const char *)str)) != funcDB.end())
  {
      // we've seen this func before so get its color
      color = (unsigned long)(*it).second;
  }
  else
  {
      // we havn't seen this func before so pick a color and put func in map
      color = nextColor();
      funcDB[(const char *)str] = color;
  }
 
  return color;
};



void GlobalFunctionInfo::buildDisplay()
{

  char **fontList;
  int    nfonts;


  DEBUG_PRINT("GlobalFunctionInfo::buildDisplay() started");

  if (form == NULL)
  {
    // we havn't built this window yet so build it
    
    form = fl_bgn_form(FL_NO_BOX, 420, 470);
    box = fl_add_box(FL_UP_BOX,0,0,420,470,"");
    canvas = fl_add_canvas(FL_NORMAL_CANVAS,10,10,400,450,"");

    fl_end_form();

    // show the form
    fl_show_form(form, FL_PLACE_CENTERFREE,FL_FULLBORDER,"Function Legend");
  
    // get the default font to use, store as member of GlobalFunctionInfo
    fontList = XListFonts(dpy, FONT_FUNCTIONLEGEND, 
                          10,&nfonts);
    if (!(nfonts > 0))
    {
      cout << "ERROR:  Unable to load font " << FONT_FUNCTIONLEGEND;
      cout << " in GlobalFunctionInfo::buildDisplay()" << endl;
      font = NULL;
    }
    else
    {
      // load font
      font = XLoadFont(dpy, fontList[0]); 
    }

    // add handler
    fl_add_canvas_handler(canvas, Expose,funcLegend_exposeHandler , this); 

  }

  DEBUG_PRINT("GlobalFunctionInfo::buildDisplay() ended");
};


NCT * GlobalFileInfo::addFile(const char *s)
{
  // add a file to our list and create it's NCT object if we havn't seen it 
  // before.  If we have seen it before, then just return it's NCT to the 
  // caller.

  char *str;
  map<const char *, NCT *, ltstr>::iterator it;
  NCT  *nct;

  DEBUG_PRINT("GlobalFileInfo::addFile() started");

  // allocate new space for the string
  str = new char[strlen(s) + 1];
  strcpy(str, s);


  if ((it = filetoNCTMap.find((const char *)str)) != filetoNCTMap.end())
  {
    // we've seen this file before and already have an NCT object for it
    nct = (NCT*)(*it).second;
  }
  else
  {
    // we havn't seen this file before so create an NCT object for it
    nct = new NCT(str);
    filetoNCTMap[(const char *)str] = nct;
  }

  return nct;

  DEBUG_PRINT("GlobalFileInfo::addFile() ended");  
};

void GlobalFileInfo::checkForFiles()
{

  // look in directory for all callstack

  DIR *dirp;
  struct dirent *direntp;
  NCT * nct;
  int   bFirstFileFound = TRUE;

  /*  NOTE!!!! readdir() returns a direct* on SGIs, but Solaris returns a
      dirent*.  This problem needs to be addressed!
  */

  DEBUG_PRINT("GlobalFileInfo::GlobalFileInfo() started");

  // hack!!!!!!!!!!!!!!
  // find the minimum time
  findMinTime();


  dirp = opendir( "." );
  while ( (direntp = readdir( dirp )) != NULL )
  { 
    // is this a callstack file?
    if (strncmp("callstack", direntp->d_name, 9) == 0)
    {
       // get this file's nct
       nct = globalFileInfo.addFile(direntp->d_name);
   
       // read data from this file
       nct->readFile();

       // repaint this nct
       nct->paint();
    }
  }
  (void)closedir( dirp );
  

  // add timeout again
  fl_add_timeout(MSECS_TOCHECKFORDATA, timeout_callback, NULL);

  DEBUG_PRINT("GlobalFileInfo::GlobalFileInfo() ended");
};


void GlobalFileInfo::findMinTime()
{

  // this is a HACK!!!!!!!!!!!!!!!!!
  // probably should be rewritten more generally
  
  // just find the first dump time in the first file, and subtract the incl.
  // time of func on bottom of first stack dump (func main).  Do this for 
  // every callstack.?.?.? file and take min of those.
    
  
  FILE *fp;
  char   line[SIZE_OF_LINE];
  char *fname;
  int nfilesfound = 0;
  DIR *dirp;
  struct dirent *direntp;
  double t1, t2;
  int j;
  

  DEBUG_PRINT("GlobalFileInfo::getMinTime() started");


 


  dirp = opendir( "." );
  while ( (direntp = readdir( dirp )) != NULL )
  { 
    // is this a callstack file?
    if (strncmp("callstack", direntp->d_name, 9) == 0)
    {

      fname = direntp->d_name;
      nfilesfound++;
      
      
      //open file
      if ((fp = fopen(fname, "r")) == NULL) 
      {
        sprintf(line, "Error in GlobalFileInfo::getMinTime(): Could not open file %s", fname);
        perror(line);
        fclose(fp);
        return;
      }
  
      // skip first two lines
      if (fgets(line, SIZE_OF_LINE, fp) == NULL) 
      {
          perror("Error in GlobalFileInfo::getMinTime(): : Cannot read function table in ");
        fclose(fp);
        return ;
      }
 
      if (fgets(line, SIZE_OF_LINE, fp) == NULL) 
      {
        perror("Error in GlobalFileInfo::getMinTime(): : Cannot read function table");
        fclose(fp);
        return;
      }

      // get time of first stack dump
      if (fgets(line, SIZE_OF_LINE, fp) == NULL)
      {
        perror("Error in GlobalFileInfo::getMinTime():  can't get dump time.");
      }
      else
      {
        sscanf(line, "%lG", &t1);

      }
      // now get incl time of bottom func on first stack dump. (assumed main())
      do
      {
        if (fgets(line, SIZE_OF_LINE, fp) == NULL)
          perror("Error in GlobalFileInfo::getMinTime(): parsing file.");
        else
          for (j=1; line[j] != '"'; j++);
          sscanf(&line[j+1], "%*ld %*ld %*lG %*lG %lG %*lG", &t2); 
      }while (line[0] == '"');

      if (nfilesfound > 1)
          minTime = MIN((t1-t2), minTime);
      else
          minTime = t1-t2;
      fclose(fp);
    }
  }
 
  DEBUG_PRINT("GlobalFileInfo::getMinTime() ended");

};

FunctionInstance::FunctionInstance()
{
  nameAndType  = new string("");
  funcExcl     = 0.0;
  funcIncl     = 0.0;
  profIncl     = 0.0;
  profExcl     = 0.0;
  ncalls       = 0;
  nsubs        = 0;
  
  canvas = NULL;
  form   = NULL;
  box    = NULL;

};


FunctionInstance::FunctionInstance(char *s, long nc, long ns, double pit,
                                    double pet, double fit, double fet)
{

  // fill in data
  nameAndType  = new string(s);
  funcExcl     = fet;
  funcIncl     = fit;
  profIncl     = pit;
  profExcl     = pet;
  ncalls       = nc;
  nsubs        = ns;  


  // get color by registering this func w/ global func database

  color = globalFuncInfo.addFunc(nameAndType->c_str());

  form   = NULL;
  canvas = NULL;
  box    = NULL;
  font   = NULL;

};


void GlobalFunctionInfo::paint()
{
  // print out all of our functions into this window

  map<const char*, unsigned long, ltstr>::iterator it;
  Window   win, oldWin;
  GC gc;
  XGCValues values;
  int y;
  int yheight = 25;
  int ydelta = 15;
  

  DEBUG_PRINT("GlobalFunctionInfo::paint() started");

  if (form != NULL)
  {
    oldWin = fl_winget();
    fl_winset(FL_ObjWin(canvas));
    win = fl_winget();

    // set up the font for the gc
    values.font = font;
    gc = XCreateGC(dpy, win, GCFont, &values);

    // set foreground color
    XSetForeground(dpy, gc, 0);
    XSetWindowBackground(dpy, win, WHITEPIXEL);
    XClearWindow(dpy, win);

    for (it = funcDB.begin(); it != funcDB.end(); it++)
    {
      y += ydelta + yheight;
      // draw a box of that color
      XSetForeground(dpy, gc, (*it).second);
      XFillRectangle(dpy, win, gc, 15, y - 15, 25, 25);
      XSetForeground(dpy, gc, 0);
      XDrawString(dpy, win ,gc, 50, y, (*it).first, strlen((*it).first));
    }
  }

  DEBUG_PRINT("GlobalFunctionInfo::paint() ended");
};

FunctionInstance::~FunctionInstance()
{
  DEBUG_PRINT("FunctionInstance::~FunctionInstance started ");

  delete nameAndType;

  DEBUG_PRINT("FunctionInstance::~FunctionInstance ended ");
}


FunctionInstance::FunctionInstance(FunctionInstance& f)
{

  DEBUG_PRINT("FunctionInstance::FunctionInstance(FunctionInstance) started");

  nameAndType  = new string(*(f.nameAndType));
  funcExcl     = f.funcExcl;
  funcIncl     = f.funcIncl;
  profIncl     = f.profIncl;
  profExcl     = f.profExcl;
  ncalls       = f.ncalls;
  nsubs        = f.nsubs;
  color        = f.color;  
  
  // UI stuff
  canvas = f.canvas; 
  form   = f.form;
  box    = f.box;
  font   = f.font;

  DEBUG_PRINT("FunctionInstance::FunctionInstance(FunctionInstance) ended");
}


void FunctionInstance::buildDisplay()
{

  GC gc;
  XGCValues    values;
  unsigned     long valueMask;
  char       **fontList;
  int          nfonts;
  Window       win;


  DEBUG_PRINT("FunctionInstance::buildDisplay() started"); 


  // build the display for the function if we havn't already
  if (form == NULL)
  {
    form = fl_bgn_form(FL_NO_BOX, 420, 470);

    box = fl_add_box(FL_UP_BOX,0,0,420,470,"");
    canvas = fl_add_canvas(FL_NORMAL_CANVAS,10,10,400,450,"");

    fl_end_form();

    // show the form
    fl_show_form(form, FL_PLACE_CENTERFREE,FL_FULLBORDER,"Function Instance Information");

    // set the canvas's GC's font member
    fontList = XListFonts(dpy, FONT_FUNCTIONINSTANCE, 
                          10,&nfonts);
    if (!(nfonts > 0))
    {
      cout << "ERROR:  Unable to allocate font";
      cout << FONT_FUNCTIONINSTANCE << endl;
      font = NULL;
    }
    else
    {
      // load font
      font = XLoadFont(dpy, fontList[0]); 
    }
   
   fl_add_canvas_handler(canvas, Expose,funcInstance_exposeHandler, this);

  }

  DEBUG_PRINT("FunctionInstance::buildDisplay()  ended");

};

void FunctionInstance::paint()
{
  // paint this fumnction 
  GC gc;
  XTextItem textItems[3];
  Window    win, oldWin;
  char      str1[MAXSTRINGLENGTH];
  char      str2[MAXSTRINGLENGTH];
  char      str3[MAXSTRINGLENGTH];
  char      str4[MAXSTRINGLENGTH]; 
  char      str5[MAXSTRINGLENGTH]; 
  char      str6[MAXSTRINGLENGTH]; 
  char      str7[MAXSTRINGLENGTH];
  XGCValues values;
  int  y;
  int  textWidth;
  XFontStruct *font_struct;


  DEBUG_PRINT("FunctionInstance::paint() started");

  if (form != NULL)
  {
    // get window
    oldWin = fl_winget();
    fl_winset(FL_ObjWin(canvas));
    win = fl_winget();

    // try to set the font for the gc
    values.font = font;
    gc = XCreateGC(dpy, win, GCFont, &values);


    XSetWindowBackground(dpy, win, WHITEPIXEL);
    XClearWindow(dpy, win);

    strcpy(str1, nameAndType->c_str());
    sprintf(str2, "Function Inclusive Time:  %.16G", funcIncl);
    sprintf(str3, "Function Exclusive Time:  %.16G", funcExcl);
    sprintf(str4, "Instance Inclusive Time:  %.16G", profIncl);
    sprintf(str5, "Instance Exclusive Time:  %.16G", profExcl);
    sprintf(str6, "        Number of calls:  %ld",    ncalls);
    sprintf(str7, "  Number of Subroutines:  %ld",    nsubs);


    y = 50;
    font_struct = XQueryFont(dpy, font);
    textWidth = XTextWidth(font_struct, str1, strlen(str1));  
    XSetForeground(dpy, gc, color);
    XFillRectangle(dpy, win, gc, 50, y - 15, textWidth, 25);
    // set foreground color
    XSetForeground(dpy, gc, 0);
    XDrawString(dpy, win ,gc, 50, y, str1, strlen(str1));
    XDrawString(dpy, win, gc, 50, y = y + 25, str2, strlen(str2));
    XDrawString(dpy, win, gc, 50, y = y + 25, str3, strlen(str3));
    XDrawString(dpy, win, gc, 50, y = y + 25, str4, strlen(str4));    
    XDrawString(dpy, win, gc, 50, y = y + 25, str5, strlen(str5));    
    XDrawString(dpy, win, gc, 50, y = y + 25, str6, strlen(str6));    
    XDrawString(dpy, win, gc, 50, y = y + 25, str7, strlen(str7));

    fl_winset(oldWin);

  }

  DEBUG_PRINT("FunctionInstance::paint() ended" );
}


Dump::Dump()
{
  DEBUG_PRINT("Dump::Dump() started");

  dumpTime = 0.0;
  myNCT    = NULL;

  DEBUG_PRINT("Dump::Dump() ended");
};


Dump::Dump(double t, NCT *nct)
{

  DEBUG_PRINT("Dump::Dump(double, NCT) started");

  dumpTime = t;
  myNCT    = nct;

  DEBUG_PRINT("Dump::Dump(double, NCT) ended");

};

Dump::~Dump()
{
  // clean up

  DEBUG_PRINT("Dump::~Dump() started");

  DEBUG_PRINT("Dump::~Dump() ended");
};

void Dump::print()
{
  // get iterator at begining of dump
  vector<FunctionInstance*>::iterator it;

  cout << dumpTime << endl;

  // cycle through functions and print them out
  for (it = funcVctr.begin(); it != funcVctr.end(); it++)
  {
    cout << *((*it)->getNameAndType()) << " "; 
    cout << (*it)->getNCalls()      << " ";
    cout << (*it)->getNSubs()       << " ";
    cout << (*it)->getProfIncl()    << " ";
    cout << (*it)->getProfExcl()    << " ";
    cout << (*it)->getFuncIncl()    << " ";
    cout << (*it)->getFuncExcl()    << " " << endl;
  }
};


void Dump::paint(FL_OBJECT* canvas, int x, int canvasHeight, int width, 
                 double vsbValue)
{ 
  //   paint this dump in drawObj

  /*
    virtual coordinate system is standard basis {e1, e2} (ie, the origin is 
    at the lower left hand corner of the window with y axis increasing upward
    and the x axis in increasing to the right.  The X system uses basis with
    origin at the *top* left corner of the window, y increasing *downward* 
    and x increasing to the right.  We compute the coords to
    draw with respect to our virtual coordinate system (i.e., standard 
    basis) and then transform coordinates to the Xwindow coord. system, 
    checking if they are visible at that time.  
 
    uses transformation matrix of:   1  0                0
                                     0 -1   canvasHeight + vsbValue - bottom
                                     0  0                1
  */


  DEBUG_PRINT("Dump::paint() started");


  GC        gc;  
  Window    oldWin, win;
  int       height;
  int       actual_y1, actual_y2;       // actual coords that we draw 
  int       virtual_y1, virtual_y2;     // coords in our space-unrestraned win
  double    displayTm;                  // time value that we are displaying  
  vector<FunctionInstance*>::reverse_iterator revIt;
  int       bottom = 20;
  int       graphPixels;               // # of pixels for printing stack bars

  Font        font;
  XTextItem   textItem;
  XFontStruct *font_struct;
  char        str[MAXSTRINGLENGTH];
  int         textWidth;

  // time value to pixel conversion.  "pptu" = "pixels per time unit"
  const double    pptuConst  = .01;   // .005 lets you see all
  double          pptu;


  // first get pptu
  pptu = pptuConst * fl_get_slider_value(myNCT->getResChanger());

  // get the old current window
  oldWin = fl_winget();
  fl_winset(FL_ObjWin(canvas));
  win = fl_winget();

  gc = fl_state[fl_vmode].gc[8];

  graphPixels  = canvasHeight- bottom; // # of pixels in canvas to draw bars

  virtual_y1 = 0;   // start drawing at 10, leave room at bottom for 
                         // text  

  for (revIt = funcVctr.rbegin(); revIt != funcVctr.rend(); revIt++)
  {
 

    //DEBUG_PRINT("-------iterating function " << *(*revIt)->getNameAndType());

      displayTm  =  (*revIt)->getProfExcl();
  
      //DEBUG_PRINT("       function's display time = " << displayTm);

      height  = displayTm * pptu;
      
      if (height < 2)  height = 2;

      virtual_y2  = virtual_y1;
      virtual_y1 += height;

      
    // only paint if virtual coord is >= scrollbar value
    if ((virtual_y1 >= vsbValue)  && (virtual_y2 <= graphPixels + vsbValue))
    {
  
      //DEBUG_PRINT("-------painting function " << *(*revIt)->getNameAndType());
   
      // transform virtual coordinates to screen coords, xcoord will remain
      // unchanged so ignore it & only compute y
      actual_y1 = -virtual_y1 + graphPixels + vsbValue;     
      actual_y2 = actual_y1 + height;
        
      if (actual_y1 < 0)
      {
          height = actual_y1 + height;
          actual_y1 =0;
          actual_y2 = height;
      }
      if (actual_y2 > graphPixels)
      {
          height = height - actual_y2 + graphPixels;
      }

      // set y position and height in funciton - used to resolve mouse clicks
      (*revIt)->setyPos(actual_y1);
      (*revIt)->setyHeight(height);
      (*revIt)->setVisible(TRUE);

      // set foreground color 
      XSetForeground(dpy, gc,  (*revIt)->getColor());
      // now draw a rectangle
      XFillRectangle(dpy, win, gc, x, actual_y1, width, height);     

      /*
   DEBUG_PRINT("-------painted function " << *(*revIt)->getNameAndType()); 
   DEBUG_PRINT("           at x = " << x);
   DEBUG_PRINT("            & y = " << actual_y1);
   DEBUG_PRINT("         height = " << height);
      */
    }
    else
      (*revIt)->setVisible(FALSE);
  } 


  // print out the label for this dump
  if ((font = myNCT->getFont()) != NULL)  
  {
    // set foreground color
    XSetForeground(dpy, gc, 0);
    sprintf(str,"%4.4lG", dumpTime);
    // set up XTextItem struct
    textItem.chars  = str;
    textItem.nchars = strlen(str);
    textItem.delta  = 0;
    textItem.font   = font;

    // get width of string
    font_struct = XQueryFont(dpy, font);
    textWidth = XTextWidth(font_struct, str, strlen(str));  

    // draw the string 
    XDrawText(dpy, win, gc, x + (width/2) - (textWidth/2), 
              canvasHeight-5, &textItem, 1);
  }

  // reset to old window
  fl_winset(oldWin);

  DEBUG_PRINT("Dump::paint() ended");
}


/*---------------------------------------------------------------------
|
|  call Dump::addFunc() to add a new FunctionInstance to Dump's 
|  function vector
|
*/
FunctionInstance* Dump::addFunc(char *nmType, long nc, long ns, double pit,
                                double pet, double fit, double fet)
{
  FunctionInstance* f;

  DEBUG_PRINT("Dump::addFunc() started");

  f = new FunctionInstance(nmType, nc, ns, pit, pet, fit , fet);
  funcVctr.push_back(f);

  DEBUG_PRINT("Dump::addFunc() ended");

  return f;



};




NCT::NCT(char *s)
{
  DEBUG_PRINT("NCT::NCT(char*) started");

  strcpy(fname, s);

  // set up UI sizes
  dumpMargin = DUMPMARGIN;
  dumpWidth  = DUMPWIDTH;
  buildDisplay();


  DEBUG_PRINT("NCT::NCT(char*) ended");
};



void NCT::readFile()
{

  char line[SIZE_OF_LINE], func[SIZE_OF_LINE];
  FILE *fp;
  int     i = 0, j         = 0;
  int     stat      = 0;
  int     lines     = 0;
  int     pos       = 0;
  double  dumpTime  = 0.0;
  double  funcExcl  = 0.0;
  double  funcIncl  = 0.0;
  double  profExcl  = 0.0;
  double  profIncl  = 0.0;
  long    ncalls    = 0;
  long    nsubs     = 0;
  Dump *currDump;
  
   DEBUG_PRINT("NCT::readFile() started");

  if ((fp = fopen(fname, "r")) == NULL) 
  {
    sprintf(line,"Error: Could not open file %s", fname);
    perror(line);
    fclose(fp);
    return;
  }

  if (fgets(line, SIZE_OF_LINE, fp) == NULL) 
  {
        perror("Error in fgets: Cannot read function table");
        fclose(fp);
        return ;
  }
 
  if (fgets(line, SIZE_OF_LINE, fp) == NULL) 
  {
        perror("Error in fgets: Cannot read function table");
        fclose(fp);
        return ;
  }

  while (TRUE) 
  {
    if (fgets(line, SIZE_OF_LINE, fp) == NULL) 
    {
      // EOF
        fclose(fp);
        return;
    }

    if(line[0] != '"') 
    { 
      /* NULL line in each block */
      sscanf(line, "%lG", &dumpTime);
      if (isdigit(line[0])) 
      { 
        currDump = this->addDump(dumpTime - globalFileInfo.getMinTime());
      }
    } 
    else { 
  
      
      for (j=1; line[j] != '"'; j++) {
          func[j-1] = line[j];
      }
      func[j-1] = '\0'; // null terminate the string
      // At this point line[j] is '"' and the has a blank after that, so
      // line[j+1] corresponds to the beginning of other data.
  
      sscanf(&line[j+1], "%ld %ld %lG %lG %lG %lG",  &ncalls, &nsubs, 
             &profIncl, &profExcl, &funcIncl, &funcExcl);
      /*
        DeBuG
      printf(" ncalls    = %ld \n", ncalls);
      printf(" nsubs     = %ld \n", nsubs);
      printf(" profIncl  = %lG \n", profIncl);
      printf(" profExcl  = %lG \n", profExcl);
      printf(" funcIncl  = %lG \n", funcIncl);
      printf(" funcExcl  = %lG \n", funcExcl);
      */

      // add new FunctionInstance to 
      currDump->addFunc(func, ncalls, nsubs, profIncl, profExcl, funcIncl,
                      funcExcl); 
   }  
 }

  DEBUG_PRINT("NCT::readFile() ended");
};


Dump* NCT::addDump(double dTime)
{
  Dump *d;

  DEBUG_PRINT("NCT::addDump() started");


  d = new Dump(dTime, this);
  dumpVctr.push_back(d);


  DEBUG_PRINT("NCT::addDump() ended");

  return d; 
};

void NCT::print()
{
  vector<Dump*>::iterator it;
  
  for (it=dumpVctr.begin();  it != dumpVctr.end(); it++)
  {
    (*it)->print();
  }
}

void NCT::paint()
{
 

  DEBUG_PRINT("NCT::paint() called");


  int x      = 100;
  int dumpNum = 0;  
  int leftMostDump, rightMostDump;
  Window win;
  vector<Dump*>::iterator it;
  double vsbValue;

  leftMostDump = fl_get_scrollbar_value(hsbScroll);
  rightMostDump = canvasWidth / (dumpWidth + dumpMargin) + leftMostDump;

  // clear out canvas
  win = fl_get_canvas_id(canvas);
  // background color will always be white
  XSetWindowBackground(dpy, win, WHITEPIXEL);
  XClearWindow(dpy, win);

  // get vertical scrollbar's value
  vsbValue = fl_get_scrollbar_value(vsbScroll);

  for(it = dumpVctr.begin(); it != dumpVctr.end(); it++)
  {
    if ((dumpNum >= leftMostDump) && (dumpNum <= rightMostDump))
    {
      // this dump is visible set it's x position and width
      (*it)->setxPos(x);
      (*it)->setxWidth(dumpWidth);
      // and paint it
      (*it)->paint(canvas, x, canvasHeight, dumpWidth, vsbValue);
      x += dumpWidth + dumpMargin;
      (*it)->setVisible(TRUE);
    }
    else
      (*it)->setVisible(FALSE);

    dumpNum++;
  }

  DEBUG_PRINT("NCT::paint() ended");

};


void NCT::buildDisplay()
{
  // build the canvas and scrollbars for this NCT to display its stuff in

 
  DEBUG_PRINT("started NCT::buildDisplay()");
  DEBUG_PRINT("   building display for " << fname);

  // position stuff
  FL_Coord canvas_xoff   =  30;
  FL_Coord canvas_yoff   =  20;
  FL_Coord vsb_width     =  20;
  FL_Coord hsb_height    =  20;
  FL_Coord formExtraY    =  100;
  FL_Coord textLabelHeight = 20;
  FL_Coord resChangerWidth = 20;

  // font stuff
  int       nfonts;
  char    **fontList;
  
  canvasWidth  = 650 - 2*canvas_xoff - vsb_width;
  canvasHeight = CANVASHEIGHT;

  // call so we can add objects dynamically to form
  fl_addto_form(fd_frmMain->frmMain);

  // now resize the form so we can see this new object
  ++nCanvasesDrawn;
  fl_set_form_size(fd_frmMain->frmMain,
                  canvas_xoff*2 + 10 + canvasWidth+resChangerWidth+vsb_width, 
                   nCanvasesDrawn *(canvasHeight + canvas_yoff + textLabelHeight)+formExtraY);

  /* add label */
  textLabel = fl_add_text(FL_NORMAL_TEXT, canvas_xoff, 
                          lastCanvasY + canvas_yoff, canvasWidth, 
                          textLabelHeight, fname);
  fl_set_object_boxtype(textLabel, FL_DOWN_BOX);

  /* add canvas */
  canvas = fl_add_canvas(FL_NORMAL_CANVAS, canvas_xoff, 
                         lastCanvasY + canvas_yoff + textLabelHeight,
                         canvasWidth, canvasHeight, "");


#if FALSE  
  // this block doesn't work so it is #if'd out 

  // set up canvas colormap and visual
   XVisualInfo  vTemplate;
   XVisualInfo  *visualList;
   int          visualsMatched;
   int          screen;
   Colormap     colormap;
   Visual       visual;
   screen = DefaultScreen(dpy);
   vTemplate.screen = screen;
   vTemplate.depth  = NCOLORPLANES;
   visualList = XGetVisualInfo(dpy, VisualScreenMask | VisualDepthMask,
                               &vTemplate, &visualsMatched);
   if (visualsMatched == 0)
   {
     cout << "ERROR:  Unable to locate 8 plane visual for X display.";
     cout << "  Using default visual and colormap" << endl; 
   }
   else
   {
     /* 
     // the Xlib way to create a colormap
     globalSharedColormap = XCreateColormap(dpy, RootWindow(dpy, screen), 
                            visualList[0].visual, AllocNone);
     */

     /*
     // xforms way to create a colormap
     colormap = fl_create_colormap(&(visualList[0]),NCOLORPLANES);
     // save our visual
     visual   = *(visualList[0].visual);
     // free visual list
     XFree(visualList);
     */
   }

   /*
  cout << "setting visual" << endl;
  //fl_set_canvas_visual(canvas, &globalSharedVisual);
  fl_set_canvas_visual(canvas, &visual);
  cout << "setting depth" << endl;
  fl_set_canvas_depth(canvas, NCOLORPLANES);
  cout << "setting colormap"<<endl;;
  //fl_share_canvas_colormap(canvas, globalSharedColormap);
  fl_set_canvas_colormap(canvas, colormap);
  cout << "colormap set" << endl;
   */
#endif


  // add scroll bars
  hsbScroll = fl_add_scrollbar(FL_HOR_SCROLLBAR,canvas_xoff,
                   lastCanvasY + canvas_yoff + canvasHeight + textLabelHeight,
                               canvasWidth, hsb_height,"");
  vsbScroll = fl_add_scrollbar(FL_VERT_SCROLLBAR, canvas_xoff + canvasWidth,
                               canvas_yoff+lastCanvasY+textLabelHeight, 
                               vsb_width, canvasHeight,"");

  resChanger = fl_add_slider(FL_VERT_BROWSER_SLIDER, 
                             canvas_xoff + vsb_width + canvasWidth + 10, 
                             canvas_yoff + lastCanvasY+textLabelHeight, 
                             resChangerWidth, canvasHeight, "");


  // remember bottom y value of canvas to position next canvas
  lastCanvasY = lastCanvasY + canvasHeight + canvas_yoff + 
                hsb_height + textLabelHeight;


  // add scroll bar callbacks
  fl_set_object_callback(hsbScroll, hsbScroll_cb, 0);
  fl_set_object_callback(vsbScroll, vsbScroll_cb, 0);
  // initialize some scrollbar stuff
  fl_set_scrollbar_return(hsbScroll, FL_RETURN_END_CHANGED); 
  fl_set_scrollbar_return(vsbScroll, FL_RETURN_END_CHANGED);
  // set up scroll bar value rounding
  fl_set_scrollbar_step(hsbScroll, 1.0);
  fl_set_scrollbar_step(vsbScroll, 1.0);
  // set scrollbar's bounds
  fl_set_scrollbar_bounds(hsbScroll, 0.0, MAXSCROLLVALUE);
  fl_set_scrollbar_bounds(vsbScroll, (double)canvasHeight*3, 0.0);
  fl_set_scrollbar_value(vsbScroll, 0.0);
  fl_set_scrollbar_value(hsbScroll, 0.0);
  // set scrollbar's increments
  fl_set_scrollbar_increment(hsbScroll, 1.0, 1.0);
  fl_set_scrollbar_increment(vsbScroll, 10.0, 50.0);
  // set up these scrollbar's FL_OBJECT's u_vdata * to point to this NCT
  // we will be able to access this NCT object via the callback function now
  hsbScroll->u_vdata = this;
  vsbScroll->u_vdata = this;

  // set up resChanger
  resChanger->u_vdata = this;
  fl_set_object_callback(resChanger, resChanger_cb, 0);

  // add canvas handler
  fl_add_canvas_handler(canvas, ButtonPress, canvas_buttonPressHandler, this);
  fl_add_canvas_handler(canvas, Expose, NCTCanvas_exposeHandler, this); 

  // set up resize and gravity
  fl_set_object_resize(canvas, FL_RESIZE_NONE);
  fl_set_object_resize(hsbScroll, FL_RESIZE_NONE);
  fl_set_object_resize(vsbScroll, FL_RESIZE_NONE);
  fl_set_object_resize(textLabel, FL_RESIZE_NONE);
  fl_set_object_resize(resChanger, FL_RESIZE_NONE);
  fl_set_object_gravity(canvas, FL_NorthWest,    FL_NoGravity);
  fl_set_object_gravity(hsbScroll, FL_NorthWest, FL_NoGravity );
  fl_set_object_gravity(vsbScroll, FL_NorthWest, FL_NoGravity);
  fl_set_object_gravity(textLabel, FL_NorthWest, FL_NoGravity);
  fl_set_object_gravity(resChanger, FL_NorthWest, FL_NoGravity);


  fl_end_form();

  // now set up the default font to use for text in the canvas
  // get font list
  fontList = XListFonts(dpy, FONT_NCTDISPLAY, 10,
                        &nfonts);
  if (nfonts == 0)
  {
    cout << "ERROR:  Unable to allocate font";
    cout << FONT_NCTDISPLAY << endl;
    font = NULL;
  }
  else
  {
    // load font
    font = XLoadFont(dpy, fontList[0]); 
  }

  DEBUG_PRINT("NCT::buildDisplay() ended");

};




FD_frmMain *create_form_frmMain(void)
{
  FL_OBJECT *obj;
  FD_frmMain *fdui = (FD_frmMain *) fl_calloc(1, sizeof(*fdui));


  fdui->frmMain = fl_bgn_form(FL_NO_BOX, 650, 360);
  obj = fl_add_box(FL_UP_BOX,0,0,650,360,"");

  fl_end_form();

  fdui->frmMain->fdui = fdui;

  return fdui;
}
/*---------------------------------------*/

/*----------------------------------------------------------------------
|
|  main()
|  =====
*/
int main(int argc, char *argv[])
{


   XVisualInfo  vTemplate;
   XVisualInfo  *visualList;
   int          visualsMatched;
   int          screen;

   DEBUG_PRINT("main() started");

   dpy = fl_initialize(&argc, argv, 0, 0, 0);
   fd_frmMain = create_form_frmMain();

   /* fill-in form initialization code */

   // we don't want a main form
   fl_set_app_nomainform(TRUE);

   // set up visual and colormap stuff
   // first set up vTemplate so that it returns a list of all the visuals 
   // of depth 8 in defined on the current screen by the X server
   screen = DefaultScreen(dpy);
   vTemplate.screen = screen;
   vTemplate.depth  = NCOLORPLANES;
   visualList = XGetVisualInfo(dpy, VisualScreenMask | VisualDepthMask,
                               &vTemplate, &visualsMatched);
   if (visualsMatched == 0)
   {
     cout << "ERROR:  Unable to locate 8 plane visual for X display.";
     cout << "  Using default visual and colormap" << endl; 
   }
   else
   {
     /* 
     // the Xlib way to create a colormap
     globalSharedColormap = XCreateColormap(dpy, RootWindow(dpy, screen), 
                            visualList[0].visual, AllocNone);
     */

     // xforms way to create a colormap
     globalSharedColormap = fl_create_colormap(&(visualList[0]),NCOLORPLANES);
     // save our visual
     globalSharedVisual   = *(visualList[0].visual);
     // free visual list
     XFree(visualList);
   }

   // ---register any desired timer callbacks here---


   /* show the first form */
   fl_show_form(fd_frmMain->frmMain,FL_PLACE_FREE,FL_FULLBORDER,"TAU Callstack Monitor");

   // read in files in directory
   globalFileInfo.checkForFiles();

   fl_do_forms();

   DEBUG_PRINT("main() ended");

   return 0;
}
