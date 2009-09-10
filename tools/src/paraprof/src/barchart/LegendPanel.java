package edu.uoregon.tau.paraprof.barchart;

import java.awt.*;

import javax.swing.JPanel;

import edu.uoregon.tau.paraprof.ParaProf;
import edu.uoregon.tau.paraprof.ParaProfUtils;

public class LegendPanel extends JPanel {

    private LegendModel model;

    private int xPanelSize;
    private int yPanelSize;

    private int leftMargin = 3;
    private int rightMargin = 50;
    
    public LegendPanel(LegendModel model) {
        this.model = model;
        setBackground(Color.white);
    }

    public void paintComponent(Graphics g) {
        try {
            super.paintComponent(g);
            draw((Graphics2D)g);
        } catch (Exception e) {
            ParaProfUtils.handleException(e);
        }
    }

    
    private void draw(Graphics2D g2D) {
        

        //Now safe to grab spacing and bar heights.
        int barHeight = ParaProf.preferencesWindow.getFontSize();

        Font font = ParaProf.preferencesWindow.getFont();
        g2D.setFont(font);
        FontMetrics fontMetrics = g2D.getFontMetrics(font);

        int barVerticalSpacing = fontMetrics.getHeight() - barHeight;
        

        int y = barVerticalSpacing;
        for (int i = 0; i < model.getNumElements(); i++) {
            String label = model.getLabel(i);
            Color color = model.getColor(i);
            
            g2D.setColor(color);
            g2D.fillRect(leftMargin, y, barHeight, barHeight);
            g2D.setColor(Color.black);
            g2D.drawRect(leftMargin, y, barHeight, barHeight);
            g2D.drawString(label, leftMargin + barHeight+5,y+barHeight);
            
            y = y + barHeight + barVerticalSpacing;
        }
        
        if (y + barVerticalSpacing > yPanelSize) {
            
            xPanelSize = 0;
            for (int i = 0; i < model.getNumElements(); i++) {
                String label = model.getLabel(i);
                xPanelSize = Math.max(xPanelSize, fontMetrics.stringWidth(label)+barHeight+rightMargin);
            }
            yPanelSize = y + barVerticalSpacing;
            setMinimumSize(new Dimension(xPanelSize, yPanelSize));
            setPreferredSize(new Dimension(xPanelSize, yPanelSize));
            this.revalidate();
        }
        
    }

    public void setModel(LegendModel model) {
        this.model = model;
    }
    
}
