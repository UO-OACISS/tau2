/* 
  Title:      DBConfigure.java
  Author:     Robert Bell
  Description: A GUI wrapper around Kevin's configure program.
*/


import java.util.*;
import java.lang.*;
import java.io.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.border.*;
import javax.swing.event.*;

public class DBConfigure extends JFrame implements ActionListener
{
    
    //Some statics.
    static boolean debugIsOn = false;
    private static String USAGE = "DBConfigure (help | debug)";
    
    
    public DBConfigure() 
    {
	try {
	    UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
	} 
	catch (Exception e) { 
	}

	//******************************
	//Window Stuff.
	//******************************
	setTitle("DBConfigure");
	
	int windowWidth = 640;
	int windowHeight = 480;
	setSize(new java.awt.Dimension(windowWidth, windowHeight));
	
	//There is really no need to resize this window.
	setResizable(false);
	
    
	//Grab the screen size.
	Toolkit tk = Toolkit.getDefaultToolkit();
	Dimension screenDimension = tk.getScreenSize();
	int screenHeight = screenDimension.height;
	int screenWidth = screenDimension.width;
	
	//Set the window to come up in the center of the screen.
	int xPosition = 0;
	int yPosition = 0;
	
	//Center the window if required.
	if((screenHeight > windowHeight) && (screenWidth > windowWidth)){
	    xPosition = (int) ((screenWidth - windowWidth)/2);
	    yPosition = (int) ((screenHeight - windowHeight)/2);
	}

	setLocation(xPosition, yPosition);

	//******************************
	//End - Window Stuff.
	//******************************

	
    //******************************
    //Code to generate the menus.
    //******************************
    
    JMenuBar mainMenu = new JMenuBar();
    
    //******************************
    //File menu.
    //******************************
    JMenu fileMenu = new JMenu("File");
    
    //Add a menu item.
    JMenuItem saveItem = new JMenuItem("Save");
    saveItem.addActionListener(this);
    fileMenu.add(saveItem);
    
    //Add a menu item.
    JMenuItem loadItem = new JMenuItem("Load");
    loadItem.addActionListener(this);
    fileMenu.add(loadItem);
     
    //Add a menu item.
    JMenuItem exitItem = new JMenuItem("Exit DBConfigure!");
    exitItem.addActionListener(this);
    fileMenu.add(exitItem);
    //******************************
    //End - File menu.
    //******************************
     
    //Now, add all the menus to the main menu.
    mainMenu.add(fileMenu);
     
    setJMenuBar(mainMenu);
    
    //******************************
    //End - Code to generate the menus.
    //******************************



    }
    
    public void actionPerformed(ActionEvent evt){

	Object EventSrc = evt.getSource();
	String arg = evt.getActionCommand();
	
	
	if(EventSrc instanceof JMenuItem){
	    if(arg.equals("Save")){
	    }
	    else if(arg.equals("Load")){
	    }
	    else if(arg.equals("Exit DBConfigure!")){
		setVisible(false);
		dispose();
		System.exit(0);
	    }
	}
    }
    
    
    // Main entry point
    static public void main(String[] args) 
    {
	int numberOfArguments = 0;
	String argument;
	
	while (numberOfArguments < args.length) {
	    argument = args[numberOfArguments++];
	    if (argument.equalsIgnoreCase("HELP")) {
		System.err.println(USAGE);
                   System.exit(-1);
	    }
	    if (argument.equalsIgnoreCase("DEBUG")) {
		DBConfigure.debugIsOn = true;
		continue;
	    }
	}

	DBConfigure dBConfigure = new DBConfigure();
	dBConfigure.show();
    }


    //Instance data.
    //This object does all the actual non-gui work.
    //Configure = new Configure();

}
