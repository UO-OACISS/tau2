package edu.uoregon.tau.perfdmf.database;

/**
 * This class attempts to erase characters echoed to the console.
 */

class MaskingThread extends Thread {
   private boolean stop = false;
   //private int index;
   private String prompt;


  /**
   *@param prompt The prompt displayed to the user
   */
   public MaskingThread(String prompt) {
      this.prompt = prompt;
   }


  /**
   * Begin masking until asked to stop.
   */
   public void run() {
      while(!stop) {
         try {
            // attempt masking at this rate
            Thread.sleep(1);
         }catch (InterruptedException iex) {
            iex.printStackTrace();
         }
         if (!stop) {
            System.out.print("\r" + prompt + " \r" + prompt);
         }
         System.out.flush();
      }
   }


  /**
   * Instruct the thread to stop masking.
   */
   public void stopMasking() {
      this.stop = true;
   }
}
