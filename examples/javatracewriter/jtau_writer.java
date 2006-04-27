import edu.uoregon.tau.tf_writer.*;
class jtau_writer{
public static void main(String[] args) {
		System.out.println("Begin!\n");
		int MAIN=1;
		int FOO=2;
		int BAR=3;
		int USER_EVENT_1=4;
		//TAU_tf_writer tau = new TAU_tf_writer();
		Ttf_file file = TAU_tf_writer.Ttf_OpenFileForOutput("tau.trc","tau.edf");

		if (file == null) {
			System.out.println("Error openging trace for output");
			return;// -1;
		}

		TAU_tf_writer.Ttf_DefThread(file, 0, 0, "node 0");
		TAU_tf_writer.Ttf_DefThread(file, 1, 0, "node 1");
		TAU_tf_writer.Ttf_DefStateGroup(file, "TAU_DEFAULT", 1);

		TAU_tf_writer.Ttf_DefState(file, MAIN, "main", 1);
		TAU_tf_writer.Ttf_DefState(file, FOO, "foo", 1);
		TAU_tf_writer.Ttf_DefState(file, BAR, "bar", 1);

		TAU_tf_writer.Ttf_DefUserEvent(file, USER_EVENT_1, "User Event 1", 1);

		//double s = 1e6;
		int s=1;
		TAU_tf_writer.Ttf_EnterState(file, 1*s, 0, 0, MAIN);
		TAU_tf_writer.Ttf_EnterState(file, 2*s, 0, 0, FOO);
		TAU_tf_writer.Ttf_EnterState(file, 3*s, 0, 0, BAR);
		TAU_tf_writer.Ttf_EventTrigger(file, 4*s, 0, 0, USER_EVENT_1, 500);
		TAU_tf_writer.Ttf_EventTrigger(file, 5*s, 0, 0, USER_EVENT_1, 1000);

		TAU_tf_writer.Ttf_EnterState(file, 4*s, 1, 0, MAIN);
		TAU_tf_writer.Ttf_SendMessage(file, 4*s,
				1, 0, // from 1,0
				0, 0, // to 0,0
				500,  // length
				42,   // tag
				0);   // communicator
		TAU_tf_writer.Ttf_EnterState(file, 6*s, 1, 0, FOO);
		TAU_tf_writer.Ttf_EnterState(file, 7*s, 1, 0, BAR);
		TAU_tf_writer.Ttf_LeaveState(file, 8*s, 1, 0, BAR);
		TAU_tf_writer.Ttf_LeaveState(file, 9*s, 1, 0, FOO);
		TAU_tf_writer.Ttf_LeaveState(file, 10*s, 1, 0, MAIN);


		TAU_tf_writer.Ttf_LeaveState(file, 11*s, 0, 0, BAR);
		TAU_tf_writer.Ttf_RecvMessage(file, 12*s,
				1, 0, // from 1,0
				0, 0, // to 0,0
				500,  // length
				42,   // tag
				0);   // communicator
		TAU_tf_writer.Ttf_LeaveState(file, 13*s, 0, 0, FOO);
		TAU_tf_writer.Ttf_LeaveState(file, 14*s, 0, 0, MAIN);

		TAU_tf_writer.Ttf_CloseOutputFile(file);
		return ;
	}}
