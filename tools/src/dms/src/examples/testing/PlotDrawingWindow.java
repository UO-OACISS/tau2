import java.awt.*;
import java.awt.event.*;
import javax.swing.*;

public class PlotDrawingWindow extends JFrame {
	
	public PlotDrawingWindow(String arg0) {

		super(arg0);

		// int winWidth = 600;
		// int winHeight = 550;	
		int winWidth = 310;
		int winHeight = 300;	
		setSize(new java.awt.Dimension(winWidth, winHeight));

		//get the screen size.
		Toolkit tk = Toolkit.getDefaultToolkit();
		Dimension screenSize = tk.getScreenSize();
		int screenHeight = screenSize.height;
		int screenWidth = screenSize.width;

		//Set the window location.
		int xPosition = (screenWidth - winWidth) / 2;
		int yPosition = (screenHeight - winHeight) / 2;      
		setLocation(xPosition, yPosition);			    			    
	}	
}

