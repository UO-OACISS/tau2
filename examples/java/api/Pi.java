//Simple program that claculates pi
//by integrating 4/(1+x^2) between 0 and 1.
//
//Takes an optional command line argument of 
//the number on increments between 0 and 1.
//If a value is not specified then it assumes 100.

import TAU.*;

public class Pi
{
    static TAU.Profile t1 = new TAU.Profile("Tau Timer", "(sleep for 5 secs)",
	"TAU_DEFAULT", TAU.Profile.TAU_DEFAULT);

    public static void main(String[] args)
    {
	int Steps = 0;

	if(args.length > 0)
	    Steps = Integer.parseInt(args[0]);
	else
	    Steps = 100;

	double Value = 0;


	//Create an instance of PiCalculateValue.
	PiCalculateValue pi = new PiCalculateValue();

	Value = pi.getPi(Steps);

	System.out.println("The value of pi is: " + Value);

	t1.Start();
 	long then = System.currentTimeMillis();

        while (System.currentTimeMillis() - then < 5000) {}
	t1.Stop();

    }
}

class PiCalculateValue
{
    //Default contructor is fine.

    public void loop(int StepsIn)
    { 

	for(int i = 0; i < StepsIn; i++)
	    {
		Xvalue = i*Increment;
		Yvalue = 4/(1+(Xvalue*Xvalue));
		Sum = (Sum + (Yvalue*Increment));
		//System.out.println("The value of YValue is: " + Yvalue);
	    }

    }
    public double getPi(int StepsIn)
    {
	Increment = 1/(float)StepsIn;
	System.out.println("The value of Increment is: " + Increment);
	
        loop(StepsIn);
	return Sum;
    }
    double Increment = 1;
    double Xvalue = 0;
    double Yvalue = 0;
    double Sum = 0;
}







