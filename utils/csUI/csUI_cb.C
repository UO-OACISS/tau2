#include "forms.h"
#include "csUI.h"

extern Display   *dpy;
extern GlobalFunctionInfo globalFuncInfo;



/*---------------------------------------------------------------------
| 
|  C A L L B A C K    R O U T I N E S
|  = = = = = = = =    = = = = = = = = 
|
*/
void big_button_cb(FL_OBJECT *ob, long data)
{
   NCT *nct;
  
   /* read file and output results */
   nct = new NCT("cs.dat");
   nct->readFile();
 
   nct->paint();
}

void hsbScroll_cb(FL_OBJECT *obj, long data)
{

  NCT *nct;
  
  DEBUG_PRINT("hsbScroll_cb started");

  // repaint this nct object
  nct = (NCT*) obj->u_vdata;
  nct->paint();
 
  DEBUG_PRINT("hsbScroll_cb ended"); 
}

void vsbScroll_cb(FL_OBJECT *obj, long data)
{
  DEBUG_PRINT("vsbScroll_cb started");

  NCT *nct;  

  nct = (NCT*) obj->u_vdata;
  nct->paint();
 
  DEBUG_PRINT("vsbScroll_cb ended");


}

void resChanger_cb(FL_OBJECT *obj, long data)
{
  // data is a pointer to NCT

  NCT *nct;

  DEBUG_PRINT("resChanger_cb started");

  nct = (NCT*) obj->u_vdata;
  nct->paint();


  DEBUG_PRINT("resChanger_cb ended");
}

void timeout_callback(int intarg, void *data)
{
  // this function is the one that keep on looking for data files from the 
  // callstack trace




}

/*---------------------------------------------------------------------
| 
|  E V E N T    H A N D L E R S
|  = = = = =    = = = = = = = = 
|
*/
int canvas_buttonPressHandler(FL_OBJECT *obj, Window win, int win_width, 
                          int win_height, XEvent *xev, void *user_data)
{
  // handle the button press in a canvas
  // user_data is a pointer to the NCT object whose canvas recieved event

  int x, y;
  int xPos, width, yPos, height;
  NCT *nct;
  Dump *dump         = NULL;
  const vector<FunctionInstance*>           *funcVctr;
  const vector<Dump*>                       *dumpVctr;
  vector<FunctionInstance*>::const_iterator funcIt;
  vector<Dump*>::const_iterator             dumpIt;
  XButtonEvent                              *btnEvent;


  DEBUG_PRINT("canvas_buttonPressHandler() started");

  nct = (NCT*)user_data;  
  btnEvent = (XButtonEvent*)xev;

  x = xev->xbutton.x;
  y = xev->xbutton.y;

  dumpVctr = nct->getDumpVctr();

  // cycle through  dumps, see if they are visible
  for (dumpIt = dumpVctr->begin(); dumpIt != dumpVctr->end(); dumpIt++)
  {
     if ((*dumpIt)->getVisible())        
     {  
       xPos = (*dumpIt)->getxPos();
       width = (*dumpIt)->getxWidth();
       // dump is visible     
       if ((x >= xPos) && (x <= (width+xPos)))
       {
         dump = (*dumpIt);
         break;
       }
     }
  }
  if (dump != NULL)
  {
    funcVctr = dump->getFuncVctr();
    // we clicked on a dump, find the function we clicked on
    for (funcIt = funcVctr->begin(); funcIt != funcVctr->end(); funcIt++)
    {
      if ((*funcIt)->getVisible())
      {
        yPos = (*funcIt)->getyPos();
        height = (*funcIt)->getyHeight();
        if (( y > yPos) && (y < (yPos + height)))
        {  
          cout << "function " << *(*funcIt)->getNameAndType() << " clicked";
          cout << " at (" << x << ", " << y << ")" <<endl;


          if (xev->xbutton.button == Button1)
	  {
            // build display for function view
            (*funcIt)->buildDisplay();
            (*funcIt)->paint();
          }
          else if (xev->xbutton.button == Button3)
	  {
            globalFuncInfo.buildDisplay();
            globalFuncInfo.paint();
	  }
          break;
        }
      }
    }

  }
  
  DEBUG_PRINT("canvas_buttonPressHandler() ended");

}


int NCTCanvas_exposeHandler(FL_OBJECT *obj, Window win, int win_width, 
                          int win_height, XEvent *xev, void *user_data)
{
  // repaint canvas.  user_data is a pointer to NCT object, just call
  // its paint() method and everything should be cool.
 
  NCT *nct;

  DEBUG_PRINT( "NCTCanvas_exposeHandler() started");

  nct = (NCT*)user_data;
  nct->paint();
  
  DEBUG_PRINT("NCTCanvas_exposeHandler() ended");
}

int funcLegend_exposeHandler(FL_OBJECT *obj, Window win, int win_width, 
                          int win_height, XEvent *xev, void *user_data)
{
  GlobalFunctionInfo *gfi;
 
  DEBUG_PRINT("funcLegend_exposeHandler() started");

  gfi = (GlobalFunctionInfo*)user_data;
  gfi->paint();

  DEBUG_PRINT("funcLegend_exposeHandler() ended");
}

int funcInstance_exposeHandler(FL_OBJECT *obj, Window win, int win_width, 
                          int win_height, XEvent *xev, void *user_data)
{
  FunctionInstance *fi;

  DEBUG_PRINT("funcInstance_exposeHandler() started");
  
  fi = (FunctionInstance*)user_data;
  fi->paint();

  DEBUG_PRINT("funcInstance_exposeHandler() ended");
}
